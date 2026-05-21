"""
--stage hierarchical

Reads:
  {output_dir}/cluster/umap_embeddings_cluster.h5
  {output_dir}/cluster/cluster_assignments.parquet
  {output_dir}/extract_embeddings/cascade_index.parquet

Writes:
  {output_dir}/hierarchical/hierarchical_results.json
  {output_dir}/hierarchical/figures/dendrogram.png
  {output_dir}/hierarchical/figures/macro_profiles.png

Runs Ward agglomerative clustering on the 11 HDBSCAN cluster centroids
(in 10D UMAP space) to reveal a two-level taxonomy: macro-archetypes
(3-4 groups) that each decompose into fine-grained HDBSCAN clusters.
The natural cut height is chosen as the largest gap in the linkage
distance sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def run(cfg: dict, output_dir: Path) -> None:
    import blueccgl.utils.mpl_compat  # noqa: F401 — Python 3.14 deepcopy patch
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist

    out_dir = output_dir / "hierarchical"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    with h5py.File(output_dir / "cluster" / "umap_embeddings_cluster.h5", "r") as f:
        umap_emb = f["embeddings"][:]                # [N, 10]

    assignments = pd.read_parquet(output_dir / "cluster" / "cluster_assignments.parquet")
    cascade_index = pd.read_parquet(output_dir / "extract_embeddings" / "cascade_index.parquet")

    labels = assignments["cluster_label"].values
    cluster_ids = sorted(k for k in np.unique(labels) if k >= 0)   # exclude noise (-1)
    n_clusters = len(cluster_ids)

    print(f"[hierarchical] {n_clusters} HDBSCAN clusters, {(labels == -1).sum()} noise points excluded")

    # ── Compute centroids in 10D UMAP space ──────────────────────────────────
    centroids = np.stack([umap_emb[labels == k].mean(axis=0) for k in cluster_ids])  # [K, 10]

    # ── Ward linkage on centroid distance matrix ──────────────────────────────
    Z = linkage(pdist(centroids, metric="euclidean"), method="ward")

    # Natural cut: largest gap between consecutive linkage merge distances
    merge_dists = Z[:, 2]
    gaps = np.diff(merge_dists)
    cut_idx = int(np.argmax(gaps))          # index into gaps → merge step with biggest jump
    cut_height = float((merge_dists[cut_idx] + merge_dists[cut_idx + 1]) / 2)
    n_macro = n_clusters - cut_idx - 1     # clusters remaining after that merge

    # Clamp to sensible range [2, 6]
    if n_macro < 2:
        n_macro = 2
        cut_height = float(merge_dists[-(n_clusters - n_macro)] - 1e-6)
    elif n_macro > 6:
        n_macro = 4
        cut_height = float(merge_dists[-(n_clusters - n_macro)] - 1e-6)

    macro_labels_per_cluster = fcluster(Z, t=n_macro, criterion="maxclust")  # [K] 1-indexed
    cluster_to_macro = {int(cluster_ids[i]): int(macro_labels_per_cluster[i] - 1)
                        for i in range(n_clusters)}

    print(f"[hierarchical] natural cut → {n_macro} macro-archetypes (cut height={cut_height:.3f})")
    for macro_id in range(n_macro):
        members = [k for k, m in cluster_to_macro.items() if m == macro_id]
        print(f"  Macro {macro_id}: HDBSCAN clusters {members}")

    # ── Characterise macro-archetypes ────────────────────────────────────────
    df = cascade_index.merge(assignments, on="cascade_id")
    df = df[df["cluster_label"] >= 0].copy()
    df["macro_label"] = df["cluster_label"].map(cluster_to_macro)

    profile_props = [
        "cascade_depth", "cascade_breadth", "cascade_size",
        "follower_edge_ratio", "wiener_index",
    ]
    # Add sentiment columns if present as separate fields or nested
    for col in ["sentiment_neg", "sentiment_neu", "sentiment_pos"]:
        if col in df.columns:
            profile_props.append(col)

    macro_profiles: dict[int, dict] = {}
    for macro_id in range(n_macro):
        sub = df[df["macro_label"] == macro_id]
        member_clusters = [k for k, m in cluster_to_macro.items() if m == macro_id]
        profile: dict = {
            "n_cascades": int(len(sub)),
            "member_clusters": member_clusters,
        }
        for prop in profile_props:
            if prop in sub.columns:
                profile[prop] = {
                    "mean":   float(sub[prop].mean()),
                    "median": float(sub[prop].median()),
                    "std":    float(sub[prop].std()),
                    "q25":    float(sub[prop].quantile(0.25)),
                    "q75":    float(sub[prop].quantile(0.75)),
                }
        macro_profiles[macro_id] = profile

    macro_colors = plt.cm.tab10(np.arange(n_macro) / 10)

    # ── Figure 1: Dendrogram ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        Z,
        labels=[f"C{k}" for k in cluster_ids],
        color_threshold=cut_height,
        ax=ax,
        leaf_font_size=11,
    )
    ax.axhline(cut_height, color="crimson", linestyle="--", linewidth=1.2,
               label=f"cut → {n_macro} macro-archetypes")
    ax.set_title("Ward Dendrogram of HDBSCAN Cascade Archetypes", fontsize=13)
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Ward linkage distance", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[hierarchical] saved dendrogram.png")

    # ── Figure 2: Macro-archetype structural profiles (median + IQR) ──────────
    display_props = [p for p in ["cascade_depth", "follower_edge_ratio",
                                  "cascade_size", "cascade_breadth"]
                     if p in df.columns]

    fig, axes = plt.subplots(1, len(display_props), figsize=(4 * len(display_props), 4),
                              sharey=False)
    if len(display_props) == 1:
        axes = [axes]

    x = np.arange(n_macro)
    for ax, prop in zip(axes, display_props):
        medians = [macro_profiles[m][prop]["median"] for m in range(n_macro)]
        q25s    = [macro_profiles[m][prop]["q25"]    for m in range(n_macro)]
        q75s    = [macro_profiles[m][prop]["q75"]    for m in range(n_macro)]
        err_lo  = [med - q25 for med, q25 in zip(medians, q25s)]
        err_hi  = [q75 - med for med, q75 in zip(medians, q75s)]
        ax.bar(x, medians, yerr=[err_lo, err_hi], color=macro_colors, capsize=4,
               edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Macro {i}" for i in range(n_macro)], fontsize=9)
        ax.set_title(prop.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("median + IQR", fontsize=9)

    fig.suptitle("Macro-Archetype Structural Profiles", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "macro_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[hierarchical] saved macro_profiles.png")

    # ── Figure 3: Sentiment composition stacked bar ───────────────────────────
    sent_cols = [c for c in ["sentiment_neg", "sentiment_neu", "sentiment_pos"]
                 if c in df.columns]
    if sent_cols:
        sent_data = {col: [macro_profiles[m][col]["median"] for m in range(n_macro)]
                     for col in sent_cols}
        fig, ax = plt.subplots(figsize=(7, 4))
        bottoms = np.zeros(n_macro)
        sent_colors = {"sentiment_neg": "#d62728", "sentiment_neu": "#aec7e8",
                       "sentiment_pos": "#2ca02c"}
        sent_labels = {"sentiment_neg": "Negative", "sentiment_neu": "Neutral",
                       "sentiment_pos": "Positive"}
        for col in sent_cols:
            vals = np.array(sent_data[col])
            ax.bar(x, vals, bottom=bottoms, color=sent_colors.get(col, "grey"),
                   label=sent_labels.get(col, col), edgecolor="white")
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"Macro {i}\n(n={macro_profiles[i]['n_cascades']:,})"
                            for i in range(n_macro)], fontsize=9)
        ax.set_ylabel("Median sentiment composition", fontsize=10)
        ax.set_title("Sentiment Composition per Macro-Archetype", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(fig_dir / "macro_sentiment.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[hierarchical] saved macro_sentiment.png")

    # ── Figure 4: UMAP scatter coloured by macro label ────────────────────────
    umap_path = output_dir / "evaluate" / "umap_coords.parquet"
    if umap_path.exists():
        umap_coords = pd.read_parquet(umap_path)
        plot_df = umap_coords.merge(assignments, on="cascade_id")
        plot_df = plot_df[plot_df["cluster_label"] >= 0].copy()
        plot_df["macro_label"] = plot_df["cluster_label"].map(cluster_to_macro)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: coloured by macro
        for mid in range(n_macro):
            sub = plot_df[plot_df["macro_label"] == mid]
            axes[0].scatter(sub["umap_x"], sub["umap_y"], s=1.5, alpha=0.4,
                            color=macro_colors[mid],
                            label=f"Macro {mid} (n={len(sub):,})")
        noise = umap_coords.merge(assignments[assignments["cluster_label"] == -1],
                                  on="cascade_id")
        if len(noise):
            axes[0].scatter(noise["umap_x"], noise["umap_y"], s=1, alpha=0.2,
                            color="lightgrey", label="noise")
        axes[0].set_title("UMAP — by Macro-Archetype", fontsize=12)
        axes[0].legend(markerscale=5, fontsize=8, loc="best")
        axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

        # Panel B: coloured by fine-grained HDBSCAN cluster
        fine_colors = plt.cm.tab20(np.arange(len(cluster_ids)) / len(cluster_ids))
        for i, k in enumerate(cluster_ids):
            sub = plot_df[plot_df["cluster_label"] == k]
            axes[1].scatter(sub["umap_x"], sub["umap_y"], s=1.5, alpha=0.4,
                            color=fine_colors[i], label=f"C{k}")
        if len(noise):
            axes[1].scatter(noise["umap_x"], noise["umap_y"], s=1, alpha=0.2,
                            color="lightgrey", label="noise")
        axes[1].set_title("UMAP — by HDBSCAN Cluster", fontsize=12)
        axes[1].legend(markerscale=5, fontsize=7, loc="best", ncol=2)
        axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

        fig.suptitle("Two-Level Cascade Archetype Structure", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "umap_macro.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[hierarchical] saved umap_macro.png")

    # ── Figure 5: Cascade count per macro ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [macro_profiles[m]["n_cascades"] for m in range(n_macro)]
    bars = ax.bar(x, counts, color=macro_colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], fontsize=9, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Macro {i}\n({', '.join(f'C{k}' for k in macro_profiles[i]['member_clusters'])})"
                        for i in range(n_macro)], fontsize=9)
    ax.set_ylabel("Number of cascades", fontsize=10)
    ax.set_title("Cascade Count per Macro-Archetype", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "macro_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[hierarchical] saved macro_counts.png")

    # ── Load fine-grained cluster profiles from explain stage ─────────────────
    cluster_profile_path = output_dir / "explain" / "cluster_profiles.json"
    per_cluster_stats: dict = {}
    if cluster_profile_path.exists():
        with open(cluster_profile_path) as f:
            cp = json.load(f)
        per_cluster_stats = cp.get("per_cluster", {})
    else:
        print("[hierarchical] WARNING: explain/cluster_profiles.json not found — "
              "skipping dendrogram+heatmap and taxonomy outputs")

    if per_cluster_stats:
        # Archetype names derived from dominant sentiment + structural character
        MACRO_NAMES = {
            0: "Pure-negative\nmicro-cascades",
            1: "Mixed negative\ndiscourse",
            2: "Large mixed-\nsentiment cascades",
            3: "Follower-driven\nneutral cascades",
            4: "Neutral follower\nspreading",
        }

        heatmap_props = [
            ("Size",          "cascade_size"),
            ("Depth",         "cascade_depth"),
            ("Breadth",       "cascade_breadth"),
            ("Follower\nratio","follower_edge_ratio"),
            ("Neg.",          "sentiment_neg"),
            ("Neu.",          "sentiment_neu"),
            ("Pos.",          "sentiment_pos"),
        ]

        # Leaf order from the dendrogram (aligns heatmap rows to dendrogram leaves)
        leaf_order = dendrogram(Z, no_plot=True)["leaves"]
        ordered_clusters = [cluster_ids[i] for i in leaf_order]

        # Raw values matrix [n_clusters × n_props]
        raw = np.array([
            [float(per_cluster_stats.get(str(k), {}).get(prop, {}).get("median", 0))
             for _, prop in heatmap_props]
            for k in ordered_clusters
        ])

        # Rank-normalise each column to [0, 1] — avoids ties from min-max
        from scipy.stats import rankdata
        n_cls = raw.shape[0]
        norm = np.apply_along_axis(
            lambda col: (rankdata(col, method="average") - 1) / max(n_cls - 1, 1),
            axis=0, arr=raw
        )

        # ── Figure 6: Dendrogram + heatmap sidebar ────────────────────────────
        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], wspace=0.05)
        ax_dendro = fig.add_subplot(gs[0])
        ax_heat   = fig.add_subplot(gs[1])

        # Horizontal dendrogram (leaves on the right side)
        ddata_h = dendrogram(
            Z,
            labels=[f"C{k}  [M{cluster_to_macro[k]}]" for k in cluster_ids],
            orientation="right",
            color_threshold=cut_height,
            ax=ax_dendro,
            leaf_font_size=10,
        )
        ax_dendro.axvline(cut_height, color="crimson", linestyle="--", linewidth=1.2,
                          label=f"cut → {n_macro} macros")
        ax_dendro.set_xlabel("Ward linkage distance", fontsize=10)
        ax_dendro.set_title("Dendrogram", fontsize=11)
        ax_dendro.legend(fontsize=8)

        # Heatmap — rows ordered by leaf_order, bottom→top to match right-oriented dendro
        im = ax_heat.imshow(norm, aspect="auto", cmap="YlOrRd",
                            vmin=0, vmax=1, origin="lower")
        ax_heat.set_xticks(range(len(heatmap_props)))
        ax_heat.set_xticklabels([lbl for lbl, _ in heatmap_props], fontsize=8,
                                 rotation=30, ha="right")
        ax_heat.set_yticks(range(len(ordered_clusters)))
        ax_heat.set_yticklabels([f"C{k}" for k in ordered_clusters], fontsize=9)
        ax_heat.yaxis.set_label_position("right")
        ax_heat.yaxis.tick_right()
        ax_heat.set_title("Cluster properties\n(normalised median)", fontsize=10)

        # Annotate cells with normalised values so numbers match the color scale
        for row_i in range(len(ordered_clusters)):
            for col_i in range(len(heatmap_props)):
                v = norm[row_i, col_i]
                txt_color = "white" if v > 0.6 else "black"
                ax_heat.text(col_i, row_i, f"{v:.2f}", ha="center", va="center",
                             fontsize=7, color=txt_color)
        fig.suptitle("Two-Level Cascade Archetype Taxonomy", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "dendrogram_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[hierarchical] saved dendrogram_heatmap.png")

        # ── Figure 7: Bubble chart — structural positioning of 11 clusters ────
        fig, ax = plt.subplots(figsize=(8, 6))
        for k in cluster_ids:
            stats  = per_cluster_stats.get(str(k), {})
            x_val  = float(stats.get("follower_edge_ratio", {}).get("median", 0))
            y_val  = float(stats.get("cascade_depth",       {}).get("median", 0))
            sz_val = float(stats.get("cascade_size",        {}).get("median", 1))
            mid    = cluster_to_macro[k]
            ax.scatter(x_val, y_val, s=sz_val * 30, color=macro_colors[mid],
                       alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(f"C{k}", (x_val, y_val), fontsize=9, ha="center", va="center",
                        fontweight="bold", color="white", zorder=4)

        # Legend: macro colour patches + size guide
        from matplotlib.lines import Line2D
        macro_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=macro_colors[m],
                   markersize=10, label=f"Macro {m}: {MACRO_NAMES.get(m,'').replace(chr(10),' ')}")
            for m in range(n_macro)
        ]
        size_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=np.sqrt(s * 30) * 0.5, label=f"size={s}")
            for s in [7, 13, 16]
        ]
        ax.legend(handles=macro_handles + size_handles, fontsize=8,
                  loc="upper left", framealpha=0.8)
        ax.set_xlabel("Follower edge ratio (median)", fontsize=11)
        ax.set_ylabel("Cascade depth (median)", fontsize=11)
        ax.set_title("Structural Positioning of Cascade Archetypes\n"
                     "(bubble size = cascade size)", fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "bubble_chart.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[hierarchical] saved bubble_chart.png")

        # ── Taxonomy CSV ──────────────────────────────────────────────────────
        import csv
        rows = []
        for k in cluster_ids:
            stats = per_cluster_stats.get(str(k), {})
            mid   = cluster_to_macro[k]
            rows.append({
                "cluster_id":          k,
                "macro_id":            mid,
                "macro_name":          MACRO_NAMES.get(mid, "").replace("\n", " "),
                "n_cascades":          int((labels == k).sum()),
                "cascade_size_median": stats.get("cascade_size",         {}).get("median", ""),
                "cascade_depth_median":stats.get("cascade_depth",        {}).get("median", ""),
                "cascade_breadth_median":stats.get("cascade_breadth",    {}).get("median", ""),
                "follower_ratio_median":stats.get("follower_edge_ratio",  {}).get("median", ""),
                "sentiment_neg_median":stats.get("sentiment_neg",        {}).get("median", ""),
                "sentiment_neu_median":stats.get("sentiment_neu",        {}).get("median", ""),
                "sentiment_pos_median":stats.get("sentiment_pos",        {}).get("median", ""),
            })
        rows.sort(key=lambda r: (r["macro_id"], r["cluster_id"]))
        taxonomy_path = out_dir / "archetype_taxonomy.csv"
        with open(taxonomy_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[hierarchical] saved archetype_taxonomy.csv ({len(rows)} clusters)")

    # ── Save JSON checkpoint ──────────────────────────────────────────────────
    results = {
        "n_hdbscan_clusters": n_clusters,
        "n_macro_archetypes": n_macro,
        "cut_height": cut_height,
        "cluster_to_macro": {str(k): v for k, v in cluster_to_macro.items()},
        "macro_profiles": {str(k): v for k, v in macro_profiles.items()},
    }
    (out_dir / "hierarchical_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"[hierarchical] done — {n_macro} macro-archetypes over {n_clusters} HDBSCAN clusters")
