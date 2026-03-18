"""
Stage: build_pyg_dataset

Converts cascade edge-list Parquet into PyG Data objects saved as .pt.

Reads:
  {output_dir}/cascade_construction/cascades.parquet
  {output_dir}/cascade_construction/cascade_meta.parquet
  {output_dir}/nlp_filter/conflict_posts.parquet           (sentiment, tier 1)
  Data/conflict_window_posts.parquet                       (sentiment, tier 2)

Writes:
  {output_dir}/build_pyg_dataset/cascade_dataset.pt
  {output_dir}/build_pyg_dataset/dataset_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from blueccgl.data.pyg_dataset import EVENT_TO_IDX, edges_to_pyg


def run(cfg: dict, output_dir: Path, device: str) -> None:
    cascade_dir = output_dir / "cascade_construction"
    out_dir = output_dir / "build_pyg_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load cascade data ─────────────────────────────────────────────
    print("[build_pyg] Loading cascade data...")
    edges_df = pd.read_parquet(cascade_dir / "cascades.parquet")
    meta_df = pd.read_parquet(cascade_dir / "cascade_meta.parquet")
    print(f"[build_pyg] {len(edges_df):,} edges across {len(meta_df):,} cascades")

    meta_lookup = dict(zip(meta_df["cascade_id"], meta_df["event_window"]))

    # ── 2. Build sentiment lookup ────────────────────────────────────────
    sentiment = _build_sentiment_lookup(edges_df, output_dir, cfg)

    # ── 3. Precompute epoch seconds ──────────────────────────────────────
    edges_df = edges_df.assign(
        source_epoch=edges_df["source_ts"].values.astype("datetime64[s]").astype("float64"),
        target_epoch=edges_df["target_ts"].values.astype("datetime64[s]").astype("float64"),
    )

    # ── 4. Convert each cascade ──────────────────────────────────────────
    print("[build_pyg] Converting to PyG Data objects...")
    data_list: list = []
    skipped = 0

    for cascade_id, grp in tqdm(
        edges_df.groupby("cascade_id"), desc="build PyG", unit="cascade"
    ):
        event = meta_lookup.get(cascade_id, "baseline")
        try:
            data = edges_to_pyg(
                cascade_id=int(cascade_id),
                source_uids=grp["source_uid"].values,
                target_uids=grp["target_uid"].values,
                edge_types=grp["edge_type"].values,
                source_epoch=grp["source_epoch"].values,
                target_epoch=grp["target_epoch"].values,
                event_window=event,
                sentiment=sentiment,
            )
            data_list.append(data)
        except Exception as e:
            print(f"[build_pyg] WARNING: cascade {cascade_id} skipped: {e}")
            skipped += 1

    print(f"[build_pyg] Built {len(data_list):,} Data objects ({skipped} skipped)")

    # ── 5. Save ──────────────────────────────────────────────────────────
    out_path = out_dir / "cascade_dataset.pt"
    torch.save(data_list, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[build_pyg] Saved {out_path} ({size_mb:.1f} MB)")

    # ── 6. Stats + validation ────────────────────────────────────────────
    stats = _compute_stats(data_list)
    stats["skipped"] = skipped
    stats["file_size_mb"] = round(size_mb, 1)
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    _print_summary(stats)


# ── Sentiment recovery ───────────────────────────────────────────────────────


def _build_sentiment_lookup(
    edges_df: pd.DataFrame, output_dir: Path, cfg: dict,
) -> dict[int, tuple[int, float]]:
    """Build uid → (sent_label, sent_score) from available data sources.

    Pass 1: conflict_posts.parquet   (1.9 M rows — roots + conflict reposters)
    Pass 2: conflict_window_posts.parquet via DuckDB  (172 M rows — all others)
    Unresolved UIDs default to neutral (1, 0.5) at feature-construction time.
    """
    all_uids = (
        set(edges_df["source_uid"].unique().tolist())
        | set(edges_df["target_uid"].unique().tolist())
    )
    print(f"[build_pyg] Unique users in cascades: {len(all_uids):,}")

    lookup: dict[int, tuple[int, float]] = {}

    # ── Pass 1: conflict_posts.parquet (small, fast) ──────────────────────
    filter_path = output_dir / "nlp_filter" / "conflict_posts.parquet"
    if filter_path.exists():
        cp = pd.read_parquet(
            filter_path, columns=["user_id", "sent_label", "sent_score"],
        )
        cp = cp.drop_duplicates(subset="user_id", keep="first")
        for uid, sl, ss in zip(cp["user_id"], cp["sent_label"], cp["sent_score"]):
            uid_int = int(uid)
            if uid_int in all_uids:
                lookup[uid_int] = (int(sl), float(ss))

    remaining = all_uids - set(lookup.keys())
    print(
        f"[build_pyg] Sentiment pass 1 (conflict_posts): "
        f"{len(lookup):,} resolved, {len(remaining):,} remaining"
    )

    # ── Pass 2: conflict_window_posts.parquet via DuckDB ──────────────────
    if remaining:
        cwp = Path(
            cfg["data"].get(
                "conflict_window_parquet", "Data/conflict_window_posts.parquet"
            )
        )
        if cwp.exists():
            found = _duckdb_sentiment_lookup(cwp, remaining, lookup)
            still_missing = len(all_uids) - len(lookup)
            print(
                f"[build_pyg] Sentiment pass 2 (conflict_window_posts): "
                f"{found:,} resolved, {still_missing:,} will default to neutral"
            )
        else:
            print(
                f"[build_pyg] WARNING: {cwp} not found — "
                f"{len(remaining):,} users will default to neutral"
            )

    return lookup


def _duckdb_sentiment_lookup(
    parquet_path: Path,
    uids: set[int],
    lookup: dict[int, tuple[int, float]],
) -> int:
    """Query conflict_window_posts.parquet for sentiment of remaining UIDs."""
    import duckdb

    con = duckdb.connect()
    uid_df = pd.DataFrame({"uid": np.array(list(uids), dtype=np.int64)})
    con.register("target_uids", uid_df)

    result = con.execute(f"""
        SELECT p.user_id,
               FIRST(p.sent_label) AS sent_label,
               FIRST(p.sent_score) AS sent_score
        FROM read_parquet('{parquet_path}') p
        WHERE p.user_id IN (SELECT uid FROM target_uids)
        GROUP BY p.user_id
    """).fetchdf()
    con.close()

    found = 0
    for _, row in result.iterrows():
        lookup[int(row["user_id"])] = (int(row["sent_label"]), float(row["sent_score"]))
        found += 1
    return found


# ── Stats + validation ────────────────────────────────────────────────────────


def _compute_stats(data_list: list) -> dict:
    if not data_list:
        return {"total_cascades": 0}

    sizes = [d.cascade_size.item() for d in data_list]
    depths = [d.cascade_depth.item() for d in data_list]
    fol_ratios = [d.follower_edge_ratio.item() for d in data_list]
    wiener_vals = [d.wiener_index.item() for d in data_list]

    # Event distribution
    idx_to_event = {v: k for k, v in EVENT_TO_IDX.items()}
    event_counts: dict[str, int] = {}
    for d in data_list:
        ev = idx_to_event.get(d.event_window.item(), "unknown")
        event_counts[ev] = event_counts.get(ev, 0) + 1

    # Avg sentiment composition
    sent_neg, sent_neu, sent_pos = 0.0, 0.0, 0.0
    for d in data_list:
        sc = d.sentiment_composition
        sent_neg += sc[0].item()
        sent_neu += sc[1].item()
        sent_pos += sc[2].item()
    n = len(data_list)

    # Feature ranges (sample first 200 cascades)
    sample_x = torch.cat([d.x for d in data_list[:200]], dim=0)
    feature_names = [
        "temporal_delta", "sent_label", "sent_score",
        "depth", "in_degree", "out_degree",
    ]

    # Edge type distribution
    all_etypes = torch.cat([d.edge_type for d in data_list])
    etype_counts = {
        "FOLLOWER": (all_etypes == 0).sum().item(),
        "NON_FOLLOWER": (all_etypes == 1).sum().item(),
        "UNKNOWN": (all_etypes == 2).sum().item(),
    }

    return {
        "total_cascades": n,
        "total_nodes": sum(sizes),
        "total_edges": sum(d.edge_index.shape[1] for d in data_list),
        "node_feature_dim": data_list[0].x.shape[1],
        "size": _dist_stats(sizes),
        "depth": _dist_stats(depths),
        "follower_edge_ratio": _dist_stats(fol_ratios),
        "wiener_index": _dist_stats(wiener_vals),
        "event_distribution": dict(sorted(event_counts.items())),
        "edge_type_distribution": etype_counts,
        "avg_sentiment_composition": {
            "negative": round(sent_neg / n, 4),
            "neutral": round(sent_neu / n, 4),
            "positive": round(sent_pos / n, 4),
        },
        "feature_ranges": {
            feature_names[i]: {
                "min": round(float(sample_x[:, i].min()), 4),
                "max": round(float(sample_x[:, i].max()), 4),
            }
            for i in range(sample_x.shape[1])
        },
    }


def _dist_stats(values: list) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "median": s[n // 2],
        "mean": round(sum(s) / n, 2),
        "p95": s[int(n * 0.95)] if n > 20 else s[-1],
        "max": s[-1],
    }


def _print_summary(stats: dict) -> None:
    print(f"\n[build_pyg] ── Dataset Summary ──────────────────────────")
    print(f"  Cascades      : {stats['total_cascades']:,}")
    print(f"  Total nodes   : {stats['total_nodes']:,}")
    print(f"  Total edges   : {stats['total_edges']:,}")
    print(f"  Feature dim   : {stats.get('node_feature_dim', '?')}")
    if stats.get("size"):
        cs = stats["size"]
        print(
            f"  Size: min={cs['min']}  median={cs['median']}  "
            f"mean={cs['mean']}  p95={cs['p95']}  max={cs['max']}"
        )
    if stats.get("depth"):
        d = stats["depth"]
        print(
            f"  Depth: min={d['min']}  median={d['median']}  "
            f"mean={d['mean']}  max={d['max']}"
        )
    if stats.get("avg_sentiment_composition"):
        sc = stats["avg_sentiment_composition"]
        print(
            f"  Avg sentiment: neg={sc['negative']:.3f}  "
            f"neu={sc['neutral']:.3f}  pos={sc['positive']:.3f}"
        )
    if stats.get("edge_type_distribution"):
        et = stats["edge_type_distribution"]
        print(
            f"  Edge types: FOLLOWER={et['FOLLOWER']:,}  "
            f"NON_FOLLOWER={et['NON_FOLLOWER']:,}  UNKNOWN={et['UNKNOWN']:,}"
        )
    if stats.get("event_distribution"):
        print(f"  Events: {stats['event_distribution']}")
    print(f"  File size     : {stats.get('file_size_mb', '?')} MB")
    print(f"[build_pyg] ────────────────────────────────────────────\n")
