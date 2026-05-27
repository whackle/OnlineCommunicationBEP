"""
Generate three full-model run vs no-sentiment run comparison figures for thesis §4.8.1.

Outputs (saved to outputs_no_sentiment/figures/):
  - probe_comparison.png      : grouped horizontal bar chart of structural probe R²
  - umap_macro_comparison.png : side-by-side UMAP coloured by macro archetype
  - macro_sentiment_comparison.png : side-by-side stacked sentiment bars per macro
"""

import copy
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Python 3.14 / matplotlib 3.10 compatibility fix
def _path_deepcopy_fix(self, memo):
    p = mpath.Path.__new__(mpath.Path)
    p.__dict__.update(copy.deepcopy(self.__dict__, memo))
    return p

mpath.Path.__deepcopy__ = _path_deepcopy_fix

REPO = pathlib.Path(__file__).resolve().parent.parent
V8_DIR = REPO / "Outputs_17_04"
V10_DIR = REPO / "outputs_no_sentiment"
OUT_DIR = V10_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
V8_COLOUR = "#2B6CB0"   # blue
V10_COLOUR = "#C05621"  # orange-brown
NEG_C = "#C0392B"
NEU_C = "#7F8C8D"
POS_C = "#27AE60"

# ─────────────────────────────────────────────────────────────────────────────
# Figure B1 — Structural Probe R² Comparison
# ─────────────────────────────────────────────────────────────────────────────

def load_probe_r2(probe_json: pathlib.Path) -> dict:
    with open(probe_json) as f:
        d = json.load(f)
    out = {}
    for target, data in d.items():
        if target == "event_window":
            continue
        out[target] = data["global_score"]
    return out


def plot_probe_comparison():
    v8_r2 = load_probe_r2(V8_DIR / "explain" / "probe_results.json")
    v10_r2 = load_probe_r2(V10_DIR / "explain" / "probe_results.json")

    labels_raw = ["follower_edge_ratio", "cascade_size", "cascade_depth",
                  "wiener_index", "cascade_breadth"]
    display_labels = ["Follower edge ratio", "Cascade size", "Cascade depth",
                      "Wiener index", "Cascade breadth"]

    v8_vals = [v8_r2[k] for k in labels_raw]
    v10_vals = [v10_r2[k] for k in labels_raw]

    y = np.arange(len(labels_raw))
    bar_h = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(y + bar_h / 2, v8_vals, bar_h, label="Full-model run (with sentiment)",
            color=V8_COLOUR, alpha=0.85)
    ax.barh(y - bar_h / 2, v10_vals, bar_h, label="No-sentiment run",
            color=V10_COLOUR, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xlabel("Linear probe R²", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # annotate values
    for i, (v8v, v10v) in enumerate(zip(v8_vals, v10_vals)):
        ax.text(v8v + 0.01, i + bar_h / 2, f"{v8v:.3f}", va="center",
                fontsize=8, color=V8_COLOUR)
        ax.text(v10v + 0.01, i - bar_h / 2, f"{v10v:.3f}", va="center",
                fontsize=8, color=V10_COLOUR)

    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Structural probe R²: full-model run vs no-sentiment run", fontsize=11, pad=8)
    fig.tight_layout()
    out_path = OUT_DIR / "probe_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B2 — UMAP Macro Side-by-Side
# ─────────────────────────────────────────────────────────────────────────────

def load_umap_with_macro(base_dir: pathlib.Path) -> pd.DataFrame:
    umap = pd.read_parquet(base_dir / "evaluate" / "umap_coords.parquet")
    clusters = pd.read_parquet(base_dir / "cluster" / "cluster_assignments.parquet")
    taxonomy = pd.read_csv(base_dir / "hierarchical" / "archetype_taxonomy.csv")

    # cluster_id → macro_id
    cluster_to_macro = taxonomy.set_index("cluster_id")["macro_id"].to_dict()
    cluster_to_name = taxonomy.set_index("cluster_id")["macro_name"].to_dict()

    df = umap.merge(clusters[["cascade_id", "cluster_label"]], on="cascade_id", how="left")
    df["macro_id"] = df["cluster_label"].map(cluster_to_macro)
    # noise → -1
    df["macro_id"] = df["macro_id"].fillna(-1).astype(int)
    return df


def plot_umap_macro_comparison():
    v8_df = load_umap_with_macro(V8_DIR)
    v10_df = load_umap_with_macro(V10_DIR)

    # v8: 5 macros + noise; v10: 2 macros + noise
    v8_palette = {
        0: "#8E44AD", 1: "#2980B9", 2: "#27AE60", 3: "#E67E22", 4: "#C0392B", -1: "#D5D8DC"
    }
    v8_names = {
        0: "M0 Pure-neg micro", 1: "M1 Mixed neg", 2: "M2 Large mixed",
        3: "M3 Neutral follower", 4: "M4 Neutral spreading", -1: "Noise"
    }
    v10_palette = {0: "#2980B9", 1: "#C0392B", -1: "#D5D8DC"}
    v10_names = {0: "M0 Pure-neg micro\n(96.6%)", 1: "M1 Mixed neg\n(3.4%)", -1: "Noise"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, palette, names, title in [
        (axes[0], v8_df, v8_palette, v8_names, "Full-model run — 5 macro-archetypes (with sentiment)"),
        (axes[1], v10_df, v10_palette, v10_names, "No-sentiment run — 2 macro-archetypes"),
    ]:
        # plot noise first (background)
        noise = df[df.macro_id == -1]
        ax.scatter(noise.umap_x, noise.umap_y, s=0.5, c=palette[-1],
                   alpha=0.3, rasterized=True)

        macro_ids = sorted([m for m in df.macro_id.unique() if m >= 0])
        for mid in macro_ids:
            sub = df[df.macro_id == mid]
            ax.scatter(sub.umap_x, sub.umap_y, s=0.8,
                       c=palette.get(mid, "#999999"),
                       alpha=0.6, label=names.get(mid, f"M{mid}"),
                       rasterized=True)

        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.tick_params(labelsize=8)
        handles = [
            mpatches.Patch(color=palette.get(m, "#999999"), label=names.get(m, f"M{m}"))
            for m in macro_ids
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                  markerscale=4, framealpha=0.8)

    fig.suptitle("UMAP embedding coloured by macro-archetype: full-model run vs no-sentiment run",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / "umap_macro_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B3 — Macro Sentiment Comparison
# ─────────────────────────────────────────────────────────────────────────────

def load_macro_sentiment(hier_json: pathlib.Path, taxonomy_csv: pathlib.Path):
    with open(hier_json) as f:
        d = json.load(f)
    taxonomy = pd.read_csv(taxonomy_csv)
    macro_names = taxonomy.groupby("macro_id")["macro_name"].first().to_dict()
    macro_counts = taxonomy.groupby("macro_id")["n_cascades"].sum().to_dict()

    rows = []
    for mid_str, profile in d["macro_profiles"].items():
        mid = int(mid_str)
        rows.append({
            "macro_id": mid,
            "name": macro_names.get(mid, f"Macro {mid}"),
            "n": macro_counts.get(mid, 0),
            "neg": profile["sentiment_neg"]["median"],
            "neu": profile["sentiment_neu"]["median"],
            "pos": profile["sentiment_pos"]["median"],
        })
    return pd.DataFrame(rows).sort_values("macro_id")


def plot_macro_sentiment_comparison():
    v8 = load_macro_sentiment(
        V8_DIR / "hierarchical" / "hierarchical_results.json",
        V8_DIR / "hierarchical" / "archetype_taxonomy.csv",
    )
    v10 = load_macro_sentiment(
        V10_DIR / "hierarchical" / "hierarchical_results.json",
        V10_DIR / "hierarchical" / "archetype_taxonomy.csv",
    )

    def short_name(row):
        name = row["name"]
        n = row["n"]
        pct = 100 * n / (v8["n"].sum() if row["n"] in v8["n"].values else v10["n"].sum())
        return f"M{row['macro_id']}\n({pct:.0f}%)"

    total_v8 = v8["n"].sum()
    total_v10 = v10["n"].sum()
    v8["label"] = v8.apply(lambda r: f"Full-model M{r.macro_id}\n({100*r.n/total_v8:.0f}%)", axis=1)
    v10["label"] = v10.apply(lambda r: f"No-sent M{r.macro_id}\n({100*r.n/total_v10:.0f}%)", axis=1)

    all_labels = list(v8["label"]) + list(v10["label"])
    neg_vals = list(v8["neg"]) + list(v10["neg"])
    neu_vals = list(v8["neu"]) + list(v10["neu"])
    pos_vals = list(v8["pos"]) + list(v10["pos"])

    x = np.arange(len(all_labels))
    bar_w = 0.65
    n_v8 = len(v8)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    b_neg = ax.bar(x, neg_vals, bar_w, label="Negative", color=NEG_C, alpha=0.85)
    b_neu = ax.bar(x, neu_vals, bar_w, bottom=neg_vals, label="Neutral",
                   color=NEU_C, alpha=0.85)
    b_pos = ax.bar(x, pos_vals, bar_w,
                   bottom=[n + u for n, u in zip(neg_vals, neu_vals)],
                   label="Positive", color=POS_C, alpha=0.85)

    # divider between full-model and no-sentiment
    ax.axvline(n_v8 - 0.5, color="black", linewidth=1.2, linestyle="--")
    ax.text(n_v8 / 2 - 0.5, 1.06, "Full-model run  (5 macros)", ha="center",
            fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(n_v8 + len(v10) / 2 - 0.5, 1.06, "No-sentiment run  (2 macros)", ha="center",
            fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8.5)
    ax.set_ylabel("Median sentiment proportion", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Macro-archetype sentiment composition: full-model run vs no-sentiment run", fontsize=11, pad=18)

    fig.tight_layout()
    out_path = OUT_DIR / "macro_sentiment_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating full-model run vs no-sentiment run comparison figures...")
    plot_probe_comparison()
    plot_umap_macro_comparison()
    plot_macro_sentiment_comparison()
    print("Done.")
