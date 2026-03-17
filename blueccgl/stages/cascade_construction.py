"""
Stage: cascade_construction
Reconstructs repost cascade trees from conflict posts, assigns parent nodes
via the priority heuristic, and labels edge types using the follower graph.

Reads:
  {output_dir}/nlp_filter/conflict_posts.parquet
  {raw_dir}/user_posts.tar.gz  OR  {sample_dir}/*.jsonl  (to find reposters)
  {raw_dir}/followers.csv.gz  (optional — for edge-type labelling)

Writes:
  {output_dir}/cascade_construction/cascades.parquet        (edge list)
  {output_dir}/cascade_construction/cascade_meta.parquet     (per-cascade metadata)
  {output_dir}/cascade_construction/construction_stats.json
"""

from __future__ import annotations

import json
import re
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from blueccgl.data.cascade_builder import (
    EDGE_FOLLOWER, EDGE_NON_FOLLOWER, EDGE_UNKNOWN,
    Cascade, CascadeNode, build_cascade_tree,
)
from blueccgl.data.events import parse_date
from blueccgl.data.follower_graph import FollowerGraph


def run(cfg: dict, output_dir: Path) -> None:
    out_dir = output_dir / "cascade_construction"
    out_dir.mkdir(parents=True, exist_ok=True)

    min_size = cfg["data"]["cascade_min_size"]
    max_size = cfg["data"]["cascade_max_size"]
    use_sample = cfg["nlp_filter"].get("use_sample_dir", True)

    # ── Step 1: Load conflict posts and identify cascade roots ───────────
    filter_path = output_dir / "nlp_filter" / "conflict_posts.parquet"
    print(f"[cascade] Loading conflict posts from {filter_path}")
    conflict_df = pq.read_table(filter_path).to_pandas()

    # Roots = original (non-repost) conflict posts
    roots_df = conflict_df[~conflict_df["is_repost"]].copy()
    root_post_ids = set(roots_df["post_id"].tolist())

    print(f"[cascade] Total conflict posts: {len(conflict_df):,}")
    print(f"[cascade] Cascade roots (originals): {len(roots_df):,}")

    # Build root lookup: post_id → root info
    root_lookup: dict[int, dict] = {}
    for _, row in roots_df.iterrows():
        root_lookup[int(row["post_id"])] = {
            "user_id":      int(row["user_id"]),
            "timestamp":    row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
            "sent_label":   int(row["sent_label"]),
            "sent_score":   float(row["sent_score"]),
            "text":         str(row["text"]),
            "event_window": str(row["event_window"]),
        }

    # Also index reposts in conflict_posts that reference known roots
    # (reposters who are themselves conflict posters)
    conflict_reposters: dict[int, list[dict]] = defaultdict(list)
    for _, row in conflict_df[conflict_df["is_repost"]].iterrows():
        rfrom = row["repost_from"]
        if rfrom is not None and int(rfrom) in root_post_ids:
            conflict_reposters[int(rfrom)].append({
                "user_id":    int(row["user_id"]),
                "timestamp":  row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
                "sent_label": int(row["sent_label"]),
                "sent_score": float(row["sent_score"]),
            })

    # ── Step 2: Find ALL reposters of our roots ───────────────────────
    print(f"[cascade] Finding reposters of {len(root_post_ids):,} roots...")
    reposters_by_root: dict[int, list[CascadeNode]] = defaultdict(list)

    # First, add any conflict reposters we already know about
    for root_pid, rlist in conflict_reposters.items():
        for r in rlist:
            reposters_by_root[root_pid].append(CascadeNode(
                user_id=r["user_id"],
                timestamp=r["timestamp"],
                sent_label=r["sent_label"],
                sent_score=r["sent_score"],
            ))

    # Preferred: pre-extracted conflict_window_posts.parquet (fast, no tar re-scan)
    conflict_window_parquet = Path(cfg["data"].get(
        "conflict_window_parquet", "Data/conflict_window_posts.parquet"
    ))

    reposts_found = 0

    if conflict_window_parquet.exists():
        reposts_found = _scan_parquet_for_reposters(
            conflict_window_parquet, root_post_ids, reposters_by_root
        )
    elif use_sample:
        # Fallback: scan sample JSONL directory
        sample_dir = Path(cfg["data"]["sample_dir"])
        reposts_found = _scan_jsonl_for_reposters(
            _iter_sample_dir(sample_dir), root_post_ids, reposters_by_root,
            total=len(list(sample_dir.glob("*.jsonl"))),
        )
    else:
        # Fallback: stream through the tar (slow but works)
        tar_path = Path(cfg["data"]["raw_dir"]) / "user_posts.tar.gz"
        if tar_path.exists():
            print(f"[cascade] WARNING: No conflict_window_posts.parquet found.")
            print(f"  Falling back to streaming {tar_path} (slow).")
            print(f"  Run 'python scripts/extract_conflict_window.py' first for 10x speedup.")
            reposts_found = _scan_jsonl_for_reposters(
                _iter_tar(tar_path), root_post_ids, reposters_by_root, total=None,
            )
        else:
            print(f"[cascade] WARNING: No data source found for reposter discovery.")
            pbar.set_postfix({"reposts": reposts_found})

    print(f"[cascade] Found {reposts_found:,} repost links across user_posts scan")
    print(f"[cascade] Roots with ≥1 reposter: {sum(1 for v in reposters_by_root.values() if len(v) >= 1):,}")

    # ── Step 3: Load follower graph (if available) ──────────────────────
    follower_graph = _load_follower_graph(cfg)

    # ── Step 4: Build cascade trees ─────────────────────────────────────
    print(f"[cascade] Building cascade trees (min_size={min_size}, max_size={max_size})...")

    cascades: list[Cascade] = []
    total_priority = {"priority_1": 0, "priority_2": 0, "priority_3": 0}
    skipped_small = 0
    skipped_large = 0

    for root_pid, root_info in tqdm(root_lookup.items(), desc="cascade build", unit="root"):
        reposters = reposters_by_root.get(root_pid, [])

        # +1 for root node
        cascade_size = len(reposters) + 1
        if cascade_size < min_size:
            skipped_small += 1
            continue
        if cascade_size > max_size:
            # Trim to max_size (keep earliest reposters)
            reposters = sorted(reposters, key=lambda n: n.timestamp)[:max_size - 1]
            skipped_large += 1

        root_node = CascadeNode(
            user_id=root_info["user_id"],
            timestamp=root_info["timestamp"],
            sent_label=root_info["sent_label"],
            sent_score=root_info["sent_score"],
            post_id=root_pid,
        )

        edges, stats = build_cascade_tree(root_node, reposters, follower_graph)
        for k, v in stats.items():
            total_priority[k] += v

        # Collect all nodes (root + unique reposters in edge list)
        node_uids = {root_node.user_id: root_node}
        for r in reposters:
            if r.user_id not in node_uids:
                node_uids[r.user_id] = r
        all_nodes = [root_node] + [n for uid, n in node_uids.items() if uid != root_node.user_id]

        cascade = Cascade(
            cascade_id=root_pid,
            root=root_node,
            event_window=root_info["event_window"],
            root_text=root_info["text"],
            nodes=all_nodes,
            edges=edges,
        )
        cascades.append(cascade)

    if follower_graph is not None:
        follower_graph.close()

    print(f"[cascade] Built {len(cascades):,} cascades")
    print(f"[cascade] Skipped: {skipped_small:,} too small (<{min_size}), {skipped_large:,} trimmed to {max_size}")

    # ── Step 5: Write outputs ───────────────────────────────────────────
    _write_edge_parquet(cascades, out_dir / "cascades.parquet")
    _write_meta_parquet(cascades, out_dir / "cascade_meta.parquet")

    # Stats
    edge_types = defaultdict(int)
    for c in cascades:
        for e in c.edges:
            edge_types[e.edge_type] += 1

    stats_out = {
        "total_cascades": len(cascades),
        "total_edges": sum(len(c.edges) for c in cascades),
        "total_nodes": sum(c.size for c in cascades),
        "skipped_small": skipped_small,
        "skipped_large": skipped_large,
        "parent_assignment": total_priority,
        "edge_types": {
            "FOLLOWER": edge_types.get(EDGE_FOLLOWER, 0),
            "NON_FOLLOWER": edge_types.get(EDGE_NON_FOLLOWER, 0),
            "UNKNOWN": edge_types.get(EDGE_UNKNOWN, 0),
        },
        "cascade_sizes": _size_stats([c.size for c in cascades]) if cascades else {},
        "by_event": _event_counts(cascades),
        "follower_graph_available": follower_graph is not None,
    }
    (out_dir / "construction_stats.json").write_text(json.dumps(stats_out, indent=2))

    _print_summary(stats_out)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_follower_graph(cfg: dict) -> FollowerGraph | None:
    csv_gz = Path(cfg["data"]["raw_dir"]) / "followers.csv.gz"
    db_path = Path(cfg["cascade"].get("follower_db_path", "Data/followers.duckdb"))

    if db_path.exists():
        print(f"[cascade] Loading follower graph from {db_path}")
        return FollowerGraph(csv_gz, db_path)
    elif csv_gz.exists():
        print(f"[cascade] Building follower graph DuckDB from {csv_gz}")
        return FollowerGraph(csv_gz, db_path)
    else:
        print(f"[cascade] WARNING: No follower data found.")
        print(f"  Looked for: {csv_gz}")
        print(f"         and: {db_path}")
        print(f"  Running in DEGRADED MODE — all edges will be UNKNOWN.")
        print(f"  Download followers.csv.gz and re-run to get edge-type labels.")
        return None


def _scan_parquet_for_reposters(
    parquet_path: Path,
    root_post_ids: set[int],
    reposters_by_root: dict[int, list[CascadeNode]],
) -> int:
    """Scan the pre-extracted conflict_window_posts.parquet for reposters. Fast path."""
    print(f"[cascade] Scanning pre-extracted Parquet: {parquet_path}")
    pf = pq.ParquetFile(parquet_path)
    found = 0

    for batch in tqdm(pf.iter_batches(batch_size=500_000), desc="parquet scan", unit="batch"):
        df = batch.to_pandas()
        # Filter to rows that are reposts of our roots
        reposts = df[df["repost_from"].isin(root_post_ids)]

        for _, row in reposts.iterrows():
            root_pid = int(row["repost_from"])
            ts = row["timestamp"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            reposters_by_root[root_pid].append(CascadeNode(
                user_id=int(row["user_id"]),
                timestamp=ts,
                sent_label=int(row["sent_label"]),
                sent_score=float(row["sent_score"]),
                post_id=int(row["post_id"]),
            ))
            found += 1

    print(f"[cascade] Found {found:,} repost links from Parquet")
    print(f"[cascade] Roots with ≥1 reposter: "
          f"{sum(1 for v in reposters_by_root.values() if len(v) >= 1):,}")
    return found


def _scan_jsonl_for_reposters(
    file_iter,
    root_post_ids: set[int],
    reposters_by_root: dict[int, list[CascadeNode]],
    total: int | None = None,
) -> int:
    """Scan JSONL files (from sample dir or tar) for reposters. Slow path."""
    found = 0
    with tqdm(total=total, unit="file", desc="cascade scan") as pbar:
        for fname, lines in file_iter:
            pbar.update(1)
            for line in lines:
                post = _safe_parse(line)
                if post is None:
                    continue
                rfrom = post.get("repost_from")
                if rfrom is None or int(rfrom) not in root_post_ids:
                    continue

                dt = parse_date(post.get("date"))
                if dt is None:
                    continue

                reposters_by_root[int(rfrom)].append(CascadeNode(
                    user_id=int(post["user_id"]),
                    timestamp=dt,
                    sent_label=int(post.get("sent_label", 1)),
                    sent_score=float(post.get("sent_score", 0.5)),
                    post_id=int(post.get("post_id", 0)),
                ))
                found += 1
            pbar.set_postfix({"reposts": found})

    print(f"[cascade] Found {found:,} repost links from JSONL scan")
    print(f"[cascade] Roots with ≥1 reposter: "
          f"{sum(1 for v in reposters_by_root.values() if len(v) >= 1):,}")
    return found


def _iter_sample_dir(sample_dir: Path):
    for fpath in sorted(sample_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            yield fpath.name, f.readlines()


def _iter_tar(tar_path: Path):
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".jsonl"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            yield member.name, f.read().decode("utf-8", errors="replace").splitlines(keepends=True)


def _safe_parse(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        parts = re.split(r"\}\s*\{", line)
        if len(parts) > 1:
            try:
                return json.loads(parts[0] + "}")
            except json.JSONDecodeError:
                pass
        return None


def _write_edge_parquet(cascades: list[Cascade], path: Path) -> None:
    if not cascades:
        print(f"[cascade] No cascades to write — skipping {path}")
        return

    rows = []
    for c in cascades:
        for e in c.edges:
            rows.append({
                "cascade_id":   c.cascade_id,
                "source_uid":   e.source_uid,
                "target_uid":   e.target_uid,
                "edge_type":    e.edge_type,
                "source_ts":    e.source_ts,
                "target_ts":    e.target_ts,
                "event_window": c.event_window,
            })

    schema = pa.schema([
        pa.field("cascade_id",   pa.int64()),
        pa.field("source_uid",   pa.int64()),
        pa.field("target_uid",   pa.int64()),
        pa.field("edge_type",    pa.int8()),
        pa.field("source_ts",    pa.timestamp("s")),
        pa.field("target_ts",    pa.timestamp("s")),
        pa.field("event_window", pa.string()),
    ])
    table = pa.table({k: [r[k] for r in rows] for k in schema.names}, schema=schema)
    pq.write_table(table, path)
    print(f"[cascade] Wrote {len(rows):,} edges to {path}")


def _write_meta_parquet(cascades: list[Cascade], path: Path) -> None:
    if not cascades:
        return

    rows = []
    for c in cascades:
        n_follower = sum(1 for e in c.edges if e.edge_type == EDGE_FOLLOWER)
        n_total = len(c.edges) if c.edges else 1

        rows.append({
            "cascade_id":       c.cascade_id,
            "root_user_id":     c.root.user_id,
            "root_timestamp":   c.root.timestamp,
            "cascade_size":     c.size,
            "depth":            c.depth,
            "event_window":     c.event_window,
            "follower_ratio":   n_follower / n_total,
            "root_text":        c.root_text[:300],
        })

    schema = pa.schema([
        pa.field("cascade_id",     pa.int64()),
        pa.field("root_user_id",   pa.int64()),
        pa.field("root_timestamp", pa.timestamp("s")),
        pa.field("cascade_size",   pa.int32()),
        pa.field("depth",          pa.int32()),
        pa.field("event_window",   pa.string()),
        pa.field("follower_ratio", pa.float32()),
        pa.field("root_text",      pa.string()),
    ])
    table = pa.table({k: [r[k] for r in rows] for k in schema.names}, schema=schema)
    pq.write_table(table, path)
    print(f"[cascade] Wrote {len(rows):,} cascade metadata rows to {path}")


def _size_stats(sizes: list[int]) -> dict:
    if not sizes:
        return {}
    sizes = sorted(sizes)
    n = len(sizes)
    return {
        "count": n,
        "min": sizes[0],
        "median": sizes[n // 2],
        "mean": round(sum(sizes) / n, 1),
        "p95": sizes[int(n * 0.95)] if n > 20 else sizes[-1],
        "max": sizes[-1],
    }


def _event_counts(cascades: list[Cascade]) -> dict:
    counts: dict[str, int] = {}
    for c in cascades:
        counts[c.event_window] = counts.get(c.event_window, 0) + 1
    return dict(sorted(counts.items()))


def _print_summary(stats: dict) -> None:
    print(f"\n[cascade] ── Summary ──────────────────────────────────────")
    print(f"  Cascades built        : {stats['total_cascades']:,}")
    print(f"  Total edges           : {stats['total_edges']:,}")
    print(f"  Total nodes           : {stats['total_nodes']:,}")
    print(f"  Skipped (too small)   : {stats['skipped_small']:,}")
    print(f"  Skipped (trimmed)     : {stats['skipped_large']:,}")
    print(f"  Follower graph        : {'yes' if stats['follower_graph_available'] else 'NO (degraded)'}")
    print(f"\n  Parent assignment priority:")
    pa_stats = stats["parent_assignment"]
    for k in ["priority_1", "priority_2", "priority_3"]:
        label = {"priority_1": "P1 (followed)", "priority_2": "P2 (temporal)", "priority_3": "P3 (root)"}[k]
        print(f"    {label:<20} {pa_stats.get(k, 0):>6,}")
    print(f"\n  Edge types:")
    et = stats["edge_types"]
    for k in ["FOLLOWER", "NON_FOLLOWER", "UNKNOWN"]:
        print(f"    {k:<15} {et.get(k, 0):>6,}")
    if stats.get("cascade_sizes"):
        cs = stats["cascade_sizes"]
        print(f"\n  Cascade size: min={cs['min']}  median={cs['median']}  "
              f"mean={cs['mean']}  p95={cs['p95']}  max={cs['max']}")
    if stats.get("by_event"):
        print(f"\n  By event:")
        for event, count in stats["by_event"].items():
            print(f"    {event:<28} {count:>5,}")
    print(f"[cascade] ──────────────────────────────────────────────────\n")
