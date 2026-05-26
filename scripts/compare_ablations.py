"""
compare_ablations.py — Aggregate 2³ ablation results into a comparison table.

Reads  evaluate/metrics.json  and  train/training_log.json  from each variant
output directory and writes:

  {out_dir}/ablation_results.csv   — one row per variant, all numeric columns
  {out_dir}/ablation_results.md    — Markdown table with Δ columns vs. full CCGL

Usage:
  python scripts/compare_ablations.py [--baseline DIR] [--ablation-root DIR]
                                       [--out-dir DIR]

Defaults:
  --baseline       Outputs_17_04
  --ablation-root  Outputs_ablation
  --out-dir        Outputs_ablation
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# ── Variant registry ──────────────────────────────────────────────────────────
# Each dict fully specifies one cell of the 2³ grid.
# output_dir is resolved relative to --baseline / --ablation-root at runtime.

VARIANTS = [
    # name               encoder   aug           sent   is_baseline
    {"name": "rgcn_feed_sent",   "encoder": "R-GCN", "aug": "feed-aware", "sentiment": True,  "baseline": True},
    {"name": "gcn_feed_sent",    "encoder": "GCN",   "aug": "feed-aware", "sentiment": True,  "baseline": False},
    {"name": "rgcn_rand_sent",   "encoder": "R-GCN", "aug": "random",     "sentiment": True,  "baseline": False},
    {"name": "rgcn_feed_nosent", "encoder": "R-GCN", "aug": "feed-aware", "sentiment": False, "baseline": False},
    {"name": "gcn_rand_sent",    "encoder": "GCN",   "aug": "random",     "sentiment": True,  "baseline": False},
    {"name": "gcn_feed_nosent",  "encoder": "GCN",   "aug": "feed-aware", "sentiment": False, "baseline": False},
    {"name": "rgcn_rand_nosent", "encoder": "R-GCN", "aug": "random",     "sentiment": False, "baseline": False},
    {"name": "gcn_rand_nosent",  "encoder": "GCN",   "aug": "random",     "sentiment": False, "baseline": False},
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_metrics(output_dir: Path) -> dict | None:
    path = output_dir / "evaluate" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_training_log(output_dir: Path) -> dict | None:
    path = output_dir / "train" / "training_log.json"
    if not path.exists():
        return None
    log = json.loads(path.read_text())
    if not log:
        return None
    best = min(log, key=lambda e: e["loss"])
    last = log[-1]
    return {
        "best_loss":      round(best["loss"], 5),
        "best_epoch":     best["epoch"],
        "final_loss":     round(last["loss"], 5),
        "total_epochs":   last["epoch"],
        "total_time_min": round(last.get("time_s", 0) / 60, 1),
        "aug_fallback_rate": round(last.get("aug_fallback_rate", float("nan")), 3),
    }


def extract_row(variant: dict, output_dir: Path) -> dict:
    """Build one result row; missing values filled with None."""
    row = {
        "variant":   variant["name"],
        "encoder":   variant["encoder"],
        "aug":       variant["aug"],
        "sentiment": "yes" if variant["sentiment"] else "no",
    }

    metrics = load_metrics(output_dir)
    if metrics:
        for key, col in [("silhouette", "sil"), ("davies_bouldin", "db"),
                         ("calinski_harabasz", "ch"), ("dbcv", "dbcv")]:
            if key in metrics:
                row[f"{col}_mean"] = round(metrics[key]["mean"], 4)
                row[f"{col}_ci_lo"] = round(metrics[key]["ci_lo"], 4)
                row[f"{col}_ci_hi"] = round(metrics[key]["ci_hi"], 4)
            else:
                row[f"{col}_mean"] = row[f"{col}_ci_lo"] = row[f"{col}_ci_hi"] = None
        meta = metrics.get("meta", {})
        row["n_clusters"]       = meta.get("n_clusters")
        row["noise_pct"]        = round(meta.get("noise_fraction", 0) * 100, 2)
        row["n_clustered"]      = meta.get("n_cascades_clustered")
        row["persistence_mean"] = round(meta.get("mean_cluster_persistence", float("nan")), 4)
    else:
        for col in ["sil", "db", "ch", "dbcv"]:
            row[f"{col}_mean"] = row[f"{col}_ci_lo"] = row[f"{col}_ci_hi"] = None
        row["n_clusters"] = row["noise_pct"] = row["n_clustered"] = row["persistence_mean"] = None

    tlog = load_training_log(output_dir)
    if tlog:
        row.update(tlog)
    else:
        for k in ["best_loss", "best_epoch", "final_loss", "total_epochs",
                  "total_time_min", "aug_fallback_rate"]:
            row[k] = None

    row["output_dir"] = str(output_dir)
    row["status"]     = "done" if metrics else "missing"
    return row


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 4, missing: str = "—") -> str:
    if val is None:
        return missing
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _delta(v: float | None, ref: float | None, higher_is_better: bool = True) -> str:
    """Format a Δ value relative to the reference row."""
    if v is None or ref is None:
        return "—"
    d = v - ref
    sign = "+" if d >= 0 else ""
    indicator = ("↑" if d > 0 else "↓" if d < 0 else "=") if higher_is_better else \
                ("↓" if d > 0 else "↑" if d < 0 else "=")
    return f"{sign}{d:+.4f} {indicator}"


def build_markdown(rows: list[dict]) -> str:
    # Find baseline row for delta computation
    ref = next((r for r in rows if r["variant"] == "rgcn_feed_sent"), None)

    header_cols = [
        ("Variant",          "variant"),
        ("Encoder",          "encoder"),
        ("Aug",              "aug"),
        ("Sent",             "sentiment"),
        ("Silhouette ↑",     "sil_mean"),
        ("Δ Sil",            None),           # computed
        ("DB ↓",             "db_mean"),
        ("Δ DB",             None),
        ("CH ↑",             "ch_mean"),
        ("DBCV ↑",           "dbcv_mean"),
        ("Clusters",         "n_clusters"),
        ("Noise %",          "noise_pct"),
        ("Best Loss",        "best_loss"),
        ("Best Epoch",       "best_epoch"),
        ("Status",           "status"),
    ]

    def row_cells(r: dict) -> list[str]:
        is_ref = (r["variant"] == "rgcn_feed_sent")
        sil    = r.get("sil_mean")
        db     = r.get("db_mean")
        return [
            f"**{r['variant']}**" if is_ref else r["variant"],
            r["encoder"],
            r["aug"],
            "✓" if r["sentiment"] == "yes" else "✗",
            _fmt(sil, 4),
            ("baseline" if is_ref else _delta(sil, ref["sil_mean"] if ref else None, True)),
            _fmt(db, 4),
            ("baseline" if is_ref else _delta(db, ref["db_mean"] if ref else None, False)),
            _fmt(r.get("ch_mean"), 1),
            _fmt(r.get("dbcv_mean"), 4),
            _fmt(r.get("n_clusters"), 0),
            _fmt(r.get("noise_pct"), 2),
            _fmt(r.get("best_loss"), 5),
            _fmt(r.get("best_epoch"), 0),
            r.get("status", "?"),
        ]

    headers = [h for h, _ in header_cols]
    sep     = [":---"] + [":---:"] * (len(headers) - 2) + [":---:"]

    lines = [
        "# BlueCCGL Ablation Results — 2³ Factorial Grid",
        "",
        "Baseline: **rgcn_feed_sent** (R-GCN + feed-aware augmentation + sentiment ON) = `Outputs_17_04`",
        "",
        "Metrics: Silhouette (↑ better), Davies-Bouldin (↓ better), "
        "Calinski-Harabasz (↑ better), DBCV (↑ better).",
        "Δ columns show change vs. baseline; ↑/↓ indicates direction of improvement.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for r in rows:
        lines.append("| " + " | ".join(row_cells(r)) + " |")

    # Main-effects summary block (only if all 8 variants are present)
    done = [r for r in rows if r["status"] == "done"]
    if len(done) == 8:
        lines += ["", "## Main Effects (averaged over the other two axes)", ""]
        lines += _main_effects_table(rows)

    lines += [
        "",
        "## Notes",
        "",
        "- **Δ Sil** and **Δ DB** are computed relative to the full CCGL baseline.",
        "- Davies-Bouldin: lower is better — a negative Δ (↑) means improvement.",
        "- Silhouette: higher is better — a positive Δ (↑) means improvement.",
        "- `aug_fallback_rate` in the CSV records the fraction of batches where",
        "  augmentation fell back to the original graph (min-nodes constraint).",
    ]
    return "\n".join(lines)


def _main_effects_table(rows: list[dict]) -> list[str]:
    """Compute average silhouette for each level of each factor."""
    def avg(subset, col):
        vals = [r[col] for r in subset if r.get(col) is not None]
        return sum(vals) / len(vals) if vals else None

    effects = {
        "Encoder — R-GCN":      [r for r in rows if r["encoder"] == "R-GCN"],
        "Encoder — GCN":        [r for r in rows if r["encoder"] == "GCN"],
        "Aug — feed-aware":     [r for r in rows if r["aug"] == "feed-aware"],
        "Aug — random":         [r for r in rows if r["aug"] == "random"],
        "Sentiment — ON":       [r for r in rows if r["sentiment"] == "yes"],
        "Sentiment — OFF":      [r for r in rows if r["sentiment"] == "no"],
    }

    lines = ["| Factor | Silhouette (avg) | DB (avg) | DBCV (avg) |",
             "| :--- | :---: | :---: | :---: |"]
    for label, subset in effects.items():
        sil  = avg(subset, "sil_mean")
        db   = avg(subset, "db_mean")
        dbcv = avg(subset, "dbcv_mean")
        lines.append(f"| {label} | {_fmt(sil, 4)} | {_fmt(db, 4)} | {_fmt(dbcv, 4)} |")
    return lines


# ── CSV ───────────────────────────────────────────────────────────────────────

CSV_COLS = [
    "variant", "encoder", "aug", "sentiment", "status",
    "sil_mean", "sil_ci_lo", "sil_ci_hi",
    "db_mean",  "db_ci_lo",  "db_ci_hi",
    "ch_mean",  "ch_ci_lo",  "ch_ci_hi",
    "dbcv_mean","dbcv_ci_lo","dbcv_ci_hi",
    "n_clusters", "noise_pct", "n_clustered", "persistence_mean",
    "best_loss", "best_epoch", "final_loss", "total_epochs",
    "total_time_min", "aug_fallback_rate",
    "output_dir",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline",      default="Outputs_17_04",
                   help="Output directory of the full CCGL baseline run.")
    p.add_argument("--ablation-root", default="Outputs_ablation",
                   help="Root directory containing the 7 ablation variant subdirs.")
    p.add_argument("--out-dir",       default="Outputs_ablation",
                   help="Where to write ablation_results.csv and ablation_results.md.")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[str] = []

    for v in VARIANTS:
        if v["baseline"]:
            output_dir = Path(args.baseline)
        else:
            output_dir = Path(args.ablation_root) / v["name"]

        row = extract_row(v, output_dir)
        rows.append(row)

        if row["status"] == "missing":
            missing.append(f"  {v['name']}  ({output_dir})")

    # Always write what we have
    csv_path = out_dir / "ablation_results.csv"
    md_path  = out_dir / "ablation_results.md"

    write_csv(rows, csv_path)
    md = build_markdown(rows)
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    done  = sum(1 for r in rows if r["status"] == "done")
    total = len(rows)
    print(f"\n{done}/{total} variants complete.")

    if missing:
        print("\nMissing (not yet trained or no evaluate/metrics.json):")
        for m in missing:
            print(m)
        print("\nRun  bash scripts/run_ablations.sh  to train missing variants.")
        sys.exit(0)   # not an error — partial tables are still useful

    print("\nAll variants present — main-effects summary included in the Markdown.")


if __name__ == "__main__":
    main()
