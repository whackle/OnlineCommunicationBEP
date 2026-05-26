"""
plot_ablation_interactions.py — 2×2 interaction plots for the ablation grid.

Reads  Outputs_ablation/ablation_results.csv  (produced by compare_ablations.py)
and writes three figures to  Outputs_ablation/figures/ :

  interaction_encoder_x_aug.png    — (R-GCN vs GCN) × (feed-aware vs random)
  interaction_encoder_x_sent.png   — (R-GCN vs GCN) × (sent ON vs OFF)
  interaction_aug_x_sent.png       — (feed-aware vs random) × (sent ON vs OFF)

Each figure shows four panels (one per metric): Silhouette, Davies-Bouldin,
Calinski-Harabasz, DBCV.  Crossing lines signal an interaction effect.

Usage:
  python scripts/plot_ablation_interactions.py [--csv PATH] [--out-dir DIR]
"""

import argparse
import copy
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath


# ── Python 3.14 / matplotlib 3.10 compatibility patch ────────────────────────
# matplotlib.path.Path.__deepcopy__ calls copy.deepcopy(super(), memo) which
# crashes on Python 3.14 because deepcopy of super() proxy objects changed.
# Replace it with a direct __new__ + dict-copy that avoids the proxy issue.
def _path_deepcopy_fix(self, memo):
    p = mpath.Path.__new__(mpath.Path)
    p.__dict__.update(copy.deepcopy(self.__dict__, memo))
    return p

mpath.Path.__deepcopy__ = _path_deepcopy_fix
# ─────────────────────────────────────────────────────────────────────────────


# ── Config ────────────────────────────────────────────────────────────────────

METRICS = [
    ("sil_mean",  "Silhouette",         True),   # (col, label, higher_is_better)
    ("db_mean",   "Davies-Bouldin",     False),
    ("ch_mean",   "Calinski-Harabasz",  True),
    ("dbcv_mean", "DBCV",               True),
]

PALETTE = ["#2563eb", "#dc2626"]   # blue = first level, red = second level


# ── Plot helpers ──────────────────────────────────────────────────────────────

def interaction_plot(
    df: pd.DataFrame,
    x_col: str,
    x_levels: list[str],
    line_col: str,
    line_levels: list[str],
    x_label: str,
    line_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Draw a 2×2 interaction plot for all four metrics.
    x_col / line_col each have exactly 2 levels.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, (col, metric_label, higher_better) in zip(axes, METRICS):
        for color, line_val in zip(PALETTE, line_levels):
            subset = df[df[line_col] == line_val]
            # Average over the third (free) axis
            means = [
                subset[subset[x_col] == xv][col].mean()
                for xv in x_levels
            ]
            # Pass string x-values directly; matplotlib's categorical axis
            # handles tick placement without triggering the Python 3.14
            # deepcopy bug in _copy_tick_props.
            ax.plot(x_levels, means, marker="o", color=color,
                    linewidth=2, markersize=8,
                    label=f"{line_label}={line_val}")

            for xv, m in zip(x_levels, means):
                if pd.notna(m):
                    ax.annotate(f"{m:.3f}", (xv, m),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8, color=color)

        direction = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{metric_label}\n({direction})", fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",     default="Outputs_ablation/ablation_results.csv",
                   help="Path to ablation_results.csv from compare_ablations.py")
    p.add_argument("--out-dir", default="Outputs_ablation/figures",
                   help="Directory to write PNG figures into.")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            "Run  python scripts/compare_ablations.py  first."
        )

    df = pd.read_csv(csv_path)

    # Check all 8 variants are present and done
    done = df[df["status"] == "done"].copy()
    if len(done) < 8:
        missing = df[df["status"] != "done"]["variant"].tolist()
        print(f"Warning: {8 - len(done)} variant(s) missing — plots will use "
              f"available data only.\nMissing: {missing}")

    if done.empty:
        print("No complete variants found. Run the ablations first.")
        return

    df = done.assign(
        encoder=done["encoder"].str.strip(),
        aug=done["aug"].str.strip(),
        sentiment=done["sentiment"].str.strip().str.lower(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: encoder × augmentation ──────────────────────────────────────
    interaction_plot(
        df,
        x_col="encoder",      x_levels=["R-GCN", "GCN"],
        line_col="aug",        line_levels=["feed-aware", "random"],
        x_label="Encoder",
        line_label="Aug",
        title="Interaction: Encoder × Augmentation strategy\n(averaged over Sentiment ON/OFF)",
        out_path=out_dir / "interaction_encoder_x_aug.png",
    )

    # ── Plot 2: encoder × sentiment ──────────────────────────────────────────
    interaction_plot(
        df,
        x_col="encoder",      x_levels=["R-GCN", "GCN"],
        line_col="sentiment",  line_levels=["yes", "no"],
        x_label="Encoder",
        line_label="Sentiment",
        title="Interaction: Encoder × Sentiment enrichment\n(averaged over Aug strategy)",
        out_path=out_dir / "interaction_encoder_x_sent.png",
    )

    # ── Plot 3: augmentation × sentiment ─────────────────────────────────────
    interaction_plot(
        df,
        x_col="aug",           x_levels=["feed-aware", "random"],
        line_col="sentiment",  line_levels=["yes", "no"],
        x_label="Augmentation strategy",
        line_label="Sentiment",
        title="Interaction: Augmentation × Sentiment enrichment\n(averaged over Encoder type)",
        out_path=out_dir / "interaction_aug_x_sent.png",
    )

    print(f"\nAll interaction plots written to {out_dir}/")


if __name__ == "__main__":
    main()
