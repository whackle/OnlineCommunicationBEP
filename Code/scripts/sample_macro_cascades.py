"""
Sample 20 cascades per macro-archetype and render the full cascade tree for verification.

Each cascade is shown as:
  - Header: macro, cluster, structural stats
  - Root text (full)
  - ASCII tree: every node labelled edge-type (FOL/NON/UNK) + sentiment (neg/neu/pos)
    Quote-post nodes (text ≠ root text) have their added commentary printed inline.

Reads:
  {output_dir}/cluster/cluster_assignments.parquet
  {output_dir}/extract_embeddings/cascade_index.parquet
  {output_dir}/hierarchical/hierarchical_results.json
  {cascades}       (outputs/cascade_construction/cascades.parquet)
  {cascade-meta}   (outputs/cascade_construction/cascade_meta.parquet)
  {conflict-posts} (outputs/nlp_filter/conflict_posts.parquet)

Writes:
  {out_dir}/cascade_samples.csv      — flat table, all 100 rows
  {out_dir}/verification_report.md   — human-readable trees, grouped by macro

Usage:
  python scripts/sample_macro_cascades.py \\
    --output-dir Outputs_17_04 \\
    --n 20 --seed 42 --out-dir verification/
"""

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd

MACRO_NAMES = {
    0: "Pure-negative micro-cascades",
    1: "Mixed negative discourse",
    2: "Large mixed-sentiment cascades",
    3: "Follower-driven neutral cascades",
    4: "Neutral follower spreading",
}
# sent_label stored as int8: 0=neg, 1=neu, 2=pos
SENT_LABEL = {0: "neg", 1: "neu", 2: "pos",
              "negative": "neg", "neutral": "neu", "positive": "pos"}
EDGE_LABEL = {0: "FOL", 1: "NON", 2: "UNK"}

# Per-macro: key structural/sentiment properties to sanity-check against sampled cascades
MACRO_VERIFY = {
    0: (
        "- Sentiment should be predominantly `neg`\n"
        "- Trees small (≤ 8 nodes) and shallow (depth ≤ 3)\n"
        "- High `FOL` edge ratio (follower-to-follower repost paths)"
    ),
    1: (
        "- Mix of `neg` / `neu` sentiment; minimal `pos`\n"
        "- C8 cascades: wider breadth + elevated `NON` edges (cross-community amplification)\n"
        "- Trees larger than M0; depth and breadth both moderate-to-high"
    ),
    2: (
        "- All sentiments present (`neg` / `neu` / `pos`)\n"
        "- Largest trees in the dataset (high cascade_size, high depth)\n"
        "- Mixed edge types — both `FOL` and `NON` present"
    ),
    3: (
        "- Predominantly `neu` sentiment\n"
        "- High `FOL` ratio — tight follower-network propagation\n"
        "- Moderate size; low-to-moderate depth"
    ),
    4: (
        "- Predominantly `neu` sentiment\n"
        "- High `FOL` ratio; small-to-moderate cascade size\n"
        "- Compact, follower-driven spreading pattern"
    ),
}
EVENT_NAMES = {
    "E1": "E1 Oct4–10",  "E2": "E2 Oct14–20", "E3": "E3 Nov21–Dec3",
    "E4": "E4 Jan23–29", "E5": "E5 Feb22–28", "E6": "E6 Feb26–Mar3",
    "baseline": "baseline",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",    default="Outputs_17_04")
    p.add_argument("--cascades",      default="outputs/cascade_construction/cascades.parquet")
    p.add_argument("--cascade-meta",  default="outputs/cascade_construction/cascade_meta.parquet")
    p.add_argument("--conflict-posts",default="outputs/nlp_filter/conflict_posts.parquet")
    p.add_argument("--n",    type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="verification")
    p.add_argument("--max-nodes", type=int, default=60,
                   help="Max tree nodes rendered per cascade (rest truncated)")
    return p.parse_args()


# ── Sampling ──────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, group_col: str, n: int, seed: int) -> pd.DataFrame:
    groups = sorted(df[group_col].unique())
    k = len(groups)
    base, rem = divmod(n, k)
    top_groups = set(df.groupby(group_col).size().sort_values(ascending=False).index[:rem])
    parts = []
    for g in groups:
        sub = df[df[group_col] == g]
        quota = min(base + (1 if g in top_groups else 0), len(sub))
        parts.append(sub.sample(n=quota, random_state=seed))
    return pd.concat(parts, ignore_index=True)


# ── Tree rendering (iterative pre-order DFS) ─────────────────────────────────

def build_children(edges: pd.DataFrame) -> tuple[dict, int]:
    """Return (children_dict, root_uid).
    children_dict: {parent_uid: [(child_uid, edge_type), ...]} sorted by target_ts.
    root_uid: the node that appears as source but never as target.
    """
    sources = set(edges["source_uid"].astype(int))
    targets = set(edges["target_uid"].astype(int))
    roots   = sources - targets
    root_uid = int(roots.pop()) if roots else int(
        edges.sort_values("source_ts")["source_uid"].iloc[0]
    )
    children: dict[int, list] = {}
    for _, row in edges.sort_values("target_ts").iterrows():
        src, tgt, et = int(row["source_uid"]), int(row["target_uid"]), int(row["edge_type"])
        children.setdefault(src, []).append((tgt, et))
    return children, root_uid


def render_tree(
    children: dict,
    root_uid: int,
    node_props: dict,      # uid -> {sent, is_quote, text}
    root_text: str,
    max_nodes: int = 60,
    wrap_width: int = 86,
) -> list[str]:
    """Iterative pre-order DFS; returns list of text lines for a code block."""
    lines: list[str] = []
    rendered = 0

    # Stack: (uid, etype_from_parent, prefix_string, is_last_sibling)
    # Push children in REVERSE original order so DFS pops them in original order.
    stack = [(root_uid, None, "", True)]

    while stack:
        uid, etype, prefix, is_last = stack.pop()
        rendered += 1

        props     = node_props.get(uid, {})
        sent      = props.get("sent", "?")
        is_qt     = props.get("is_quote", False)
        qt_text   = props.get("text", "")

        if etype is None:                        # root node
            line         = f"ROOT  uid={uid}  sent={sent}"
            child_prefix = ""
        else:
            connector    = "└── " if is_last else "├── "
            etype_str    = EDGE_LABEL.get(etype, "?")
            line         = f"{prefix}{connector}uid={uid}  {etype_str}  {sent}"
            child_prefix = prefix + ("    " if is_last else "│   ")
            if is_qt:
                line += "  ← quote post"

        lines.append(line)

        # Inline quote-post added text
        if is_qt and qt_text:
            indent = child_prefix + "  "
            for seg in textwrap.wrap(qt_text, width=wrap_width - len(indent)):
                lines.append(f"{indent}| {seg}")

        if rendered >= max_nodes:
            remaining = _subtree_size(children, uid, set(uid for _, _, _, _ in stack))
            if remaining:
                lines.append(f"{child_prefix}… ({remaining} more nodes not shown)")
            break

        kids = children.get(uid, [])
        # Push reversed so first child is on top of stack (processed first)
        for i, (child_uid, child_etype) in enumerate(reversed(kids)):
            is_last_child = (i == 0)   # i==0 in reversed → last in original order
            stack.append((child_uid, child_etype, child_prefix, is_last_child))

    return lines


def _subtree_size(children: dict, uid: int, exclude: set) -> int:
    """Count descendants of uid not already in exclude."""
    count = 0
    q = list(children.get(uid, []))
    seen = set()
    while q:
        child, _ = q.pop()
        if child in seen or child in exclude:
            continue
        seen.add(child)
        count += 1
        q.extend(children.get(child, []))
    return count


# ── Markdown report ───────────────────────────────────────────────────────────

def format_sent(neg: float, neu: float, pos: float) -> str:
    return f"{neg*100:.0f}% neg / {neu*100:.0f}% neu / {pos*100:.0f}% pos"


def wrap_quote(text: str, width: int = 90) -> str:
    if not isinstance(text, str) or not text.strip():
        return "> *[no text available]*"
    return "\n".join("> " + l for l in textwrap.wrap(text, width=width - 2)) or "> *[empty]*"


def build_markdown(
    samples: pd.DataFrame,
    macro_profiles: dict,
    cascade_edges: pd.DataFrame,
    node_lookup: dict,          # (uid, ts) → {sent_label, text, is_repost}
    root_text_map: dict,
    max_nodes: int,
    args_n: int = 20,
) -> str:
    lines = [
        "# Macro-Archetype Full-Cascade Verification Report",
        "",
        "Each cascade: root text + full ASCII tree.",
        "Node labels: `ROOT/└──/├──  uid=<id>  <edge_type>  <sent>`",
        "Edge: `FOL`=FOLLOWER · `NON`=NON-FOLLOWER · `UNK`=UNKNOWN",
        "Sent: `neg`=0 · `neu`=1 · `pos`=2 (int8)",
        "",
    ]

    for macro_id in sorted(samples["macro_label"].unique()):
        name    = MACRO_NAMES.get(macro_id, f"Macro {macro_id}")
        profile = macro_profiles.get(str(macro_id), {})
        members = profile.get("member_clusters", [])
        n_total = profile.get("n_cascades", "?")

        lines += ["---", "",
                  f"## Macro {macro_id} — {name}", "",
                  f"**Clusters:** {', '.join(f'C{c}' for c in members)}  "
                  f"**Total cascades:** {n_total:,}", "",
                  "**Expected profile:**"]

        for prop, label in [("cascade_size", "Size (median)"), ("cascade_depth", "Depth"),
                             ("cascade_breadth", "Breadth"), ("follower_edge_ratio", "Follower ratio")]:
            if prop in profile:
                v = profile[prop]
                lines.append(f"- {label}: {v['median']:.1f}  (IQR {v['q25']:.1f}–{v['q75']:.1f})")
        if "sentiment_neg" in profile:
            lines.append(f"- Sentiment: " + format_sent(
                profile["sentiment_neg"]["median"],
                profile["sentiment_neu"]["median"],
                profile["sentiment_pos"]["median"],
            ))

        verify_text = MACRO_VERIFY.get(macro_id, "")
        if verify_text:
            lines += ["", "**What to verify:**", verify_text]
        lines.append("")

        macro_rows = samples[samples["macro_label"] == macro_id].sort_values("cluster_label")
        for i, (_, row) in enumerate(macro_rows.iterrows(), 1):
            cid      = int(row["cascade_id"])
            root_txt = root_text_map.get(cid, "[not available]")
            edges    = cascade_edges[cascade_edges["cascade_id"] == cid]
            ev       = EVENT_NAMES.get(str(row["event_window"]), str(row["event_window"]))
            sent_str = format_sent(row["sentiment_neg"], row["sentiment_neu"], row["sentiment_pos"])

            lines += [
                "---",
                "",
                f"### [{i}/{args_n}] Cascade `{cid}` — C{int(row['cluster_label'])}",
                (
                    f"Size={row['cascade_size']:.0f} · "
                    f"Depth={row['cascade_depth']:.0f} · "
                    f"Breadth={row['cascade_breadth']:.0f} · "
                    f"Follower={row['follower_edge_ratio']:.2f} · "
                    f"Sentiment: {sent_str} · "
                    f"Event: {ev}"
                ),
                "",
                "**Root text:**",
                wrap_quote(root_txt),
                "",
                "**Cascade tree:**",
                "```",
            ]

            if edges.empty:
                lines.append("  (no edges found)")
                lines += ["```", ""]
            else:
                children, root_uid = build_children(edges)

                # Per-node sentiment lookup
                node_ts: dict[int, pd.Timestamp] = {}
                for _, er in edges.iterrows():
                    node_ts[int(er["source_uid"])] = er["source_ts"]
                    node_ts[int(er["target_uid"])] = er["target_ts"]

                node_props: dict[int, dict] = {}
                for uid, ts in node_ts.items():
                    info       = node_lookup.get((uid, ts), {})
                    raw_sent   = info.get("sent_label", None)
                    sent       = SENT_LABEL.get(raw_sent, "?") if raw_sent is not None else "?"
                    node_text  = info.get("text", "")
                    is_quote   = (
                        isinstance(node_text, str)
                        and bool(node_text)
                        and node_text != root_txt
                        and not info.get("is_repost", True)
                    )
                    node_props[uid] = {"sent": sent, "is_quote": is_quote, "text": node_text}

                tree_lines = render_tree(children, root_uid, node_props, root_txt, max_nodes=max_nodes)
                lines.extend(tree_lines)

                # Edge-type tally
                etype_counts = edges["edge_type"].value_counts().to_dict()
                fol = int(etype_counts.get(0, 0))
                non = int(etype_counts.get(1, 0))
                unk = int(etype_counts.get(2, 0))
                total_e = fol + non + unk
                fol_pct = f"{fol / total_e:.0%}" if total_e > 0 else "N/A"
                lines += [
                    "```",
                    "",
                    f"> **Edge tally:** FOL={fol}  NON={non}  UNK={unk}  "
                    f"→  observed follower ratio: {fol_pct}",
                    "",
                ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    print("Loading cluster assignments + cascade index…")
    asgn = pd.read_parquet(output_dir / "cluster" / "cluster_assignments.parquet")
    ci   = pd.read_parquet(output_dir / "extract_embeddings" / "cascade_index.parquet")
    hier = json.loads((output_dir / "hierarchical" / "hierarchical_results.json").read_text())
    meta = pd.read_parquet(args.cascade_meta, columns=["cascade_id", "root_text"])

    cluster_to_macro = {int(k): v for k, v in hier["cluster_to_macro"].items()}
    macro_profiles   = hier["macro_profiles"]

    df = asgn.merge(ci, on="cascade_id").merge(meta, on="cascade_id")
    df = df[df["cluster_label"] >= 0].copy()
    df.loc[:, "macro_label"] = df["cluster_label"].map(cluster_to_macro)

    for m in sorted(df["macro_label"].unique()):
        sub = df[df["macro_label"] == m]
        cc  = sub.groupby("cluster_label").size().to_dict()
        print(f"  Macro {m}: {len(sub):,} — " + ", ".join(f"C{k}={v:,}" for k, v in sorted(cc.items())))

    samples = pd.concat([
        stratified_sample(df[df["macro_label"] == m], "cluster_label", args.n, args.seed)
        for m in sorted(df["macro_label"].unique())
    ], ignore_index=True)
    print(f"Sampled {len(samples)} cascades total ({args.n} per macro)\n")

    sample_ids    = set(samples["cascade_id"].astype(int))
    root_text_map = dict(zip(meta["cascade_id"], meta["root_text"]))

    print("Loading cascade edges for sampled cascades…")
    all_edges     = pd.read_parquet(args.cascades)
    cascade_edges = all_edges[all_edges["cascade_id"].isin(sample_ids)].copy()
    cascade_edges.loc[:, "source_ts"] = pd.to_datetime(cascade_edges["source_ts"])
    cascade_edges.loc[:, "target_ts"] = pd.to_datetime(cascade_edges["target_ts"])

    print("Loading per-node sentiment from conflict_posts…")
    all_uids = set(cascade_edges["source_uid"].astype(int)) | set(cascade_edges["target_uid"].astype(int))
    cp = pd.read_parquet(args.conflict_posts, columns=["user_id", "timestamp", "text", "sent_label", "is_repost"])
    cp = cp[cp["user_id"].isin(all_uids)].copy()
    cp.loc[:, "ts"] = pd.to_datetime(cp["timestamp"])

    node_lookup: dict = {}
    for _, row in cp.iterrows():
        key = (int(row["user_id"]), row["ts"])
        if key not in node_lookup:
            node_lookup[key] = {
                "sent_label": int(row["sent_label"]) if pd.notna(row["sent_label"]) else None,
                "text":       row["text"],
                "is_repost":  bool(row["is_repost"]),
            }
    print(f"  {len(node_lookup):,} entries for {len(all_uids):,} unique users")

    # Summary to stdout
    print()
    for m in sorted(samples["macro_label"].unique()):
        sub = samples[samples["macro_label"] == m]
        cc  = sub.groupby("cluster_label").size().to_dict()
        print(f"Macro {m} — {MACRO_NAMES[m]}")
        print(f"  " + ", ".join(f"C{c}(n={v})" for c, v in sorted(cc.items())))
        print(f"  Size={sub['cascade_size'].median():.0f}  "
              f"Depth={sub['cascade_depth'].median():.0f}  "
              f"Follower={sub['follower_edge_ratio'].median():.2f}  "
              f"Sentiment: {format_sent(sub['sentiment_neg'].median(), sub['sentiment_neu'].median(), sub['sentiment_pos'].median())}")
        print()

    # CSV
    csv_cols = ["macro_label","cluster_label","cascade_id","event_window",
                "cascade_size","cascade_depth","cascade_breadth",
                "follower_edge_ratio","sentiment_neg","sentiment_neu","sentiment_pos","root_text"]
    samples[csv_cols].sort_values(["macro_label","cluster_label"]).to_csv(
        out_dir / "cascade_samples.csv", index=False)
    print(f"Wrote {out_dir}/cascade_samples.csv")

    # Markdown with trees
    print("Rendering cascade trees…")
    md = build_markdown(samples, macro_profiles, cascade_edges, node_lookup, root_text_map, args.max_nodes, args.n)
    (out_dir / "verification_report.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out_dir}/verification_report.md")


if __name__ == "__main__":
    main()
