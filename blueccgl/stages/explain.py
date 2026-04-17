"""Stage: explain — Multi-layer XAI for BlueCCGL cascade archetypes.

Sub-steps
---------
3.1  Probing classifiers: what properties are linearly decodable from embeddings?
3.2  DiffPool assignment extraction: which structural roles does the model learn?
3.3  Cluster statistical profiles: archetype characterisation + RQ-aligned analyses.
3.4  Augmentation sensitivity: which structural signal defines each archetype?
3.5  Representative cascade visualisation: per-archetype tree plots.

Reads:
  {output_dir}/train/checkpoints/best.pt
  {output_dir}/build_pyg_dataset/cascade_dataset.pt
  {output_dir}/extract_embeddings/embeddings.h5
  {output_dir}/extract_embeddings/cascade_index.parquet
  {output_dir}/cluster/cluster_assignments.parquet

Writes:
  {output_dir}/explain/explanation_report.json        (checkpoint)
  {output_dir}/explain/probe_results.json
  {output_dir}/explain/diffpool_assignments.h5
  {output_dir}/explain/cluster_profiles.json
  {output_dir}/explain/augmentation_sensitivity.json
  {output_dir}/explain/figures/probe_heatmap.png
  {output_dir}/explain/figures/diffpool_analysis.png
  {output_dir}/explain/figures/cluster_radar_profiles.png
  {output_dir}/explain/figures/archetype_event_heatmap.png
  {output_dir}/explain/figures/follower_ratio_violins.png
  {output_dir}/explain/figures/temporal_evolution.png
  {output_dir}/explain/figures/augmentation_sensitivity_bars.png
  {output_dir}/explain/figures/archetypes/cluster_{k}_rep_{i}.png  (optional)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import blueccgl.utils.mpl_compat  # noqa: F401 — patches matplotlib Path.__deepcopy__ for Python 3.14


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from blueccgl.models.blueccgl import BlueCCGL

# ---------------------------------------------------------------------------
# Event window index → human-readable label
# ---------------------------------------------------------------------------
EVENT_LABELS = {
    0: "Baseline",
    1: "E1: Oct7 Attack",
    2: "E2: Al-Ahli",
    3: "E3: Ceasefire",
    4: "E4: ICJ Ruling",
    5: "E5: Bushnell",
    6: "E6: Flour Massacre",
}

# Edge type index → label
EDGE_TYPE_LABELS = {0: "Follower", 1: "Non-Follower", 2: "Unknown"}


# ===========================================================================
# Entry point
# ===========================================================================


def run(cfg: dict, output_dir: Path, device: str) -> None:
    xcfg = cfg.get("explain", {})
    out_dir = output_dir / "explain"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "archetypes").mkdir(exist_ok=True)

    dev = torch.device(device)

    # ── Load shared inputs ────────────────────────────────────────────────
    print("[explain] Loading shared inputs ...")

    with h5py.File(output_dir / "extract_embeddings" / "embeddings.h5", "r") as f:
        embeddings: np.ndarray = f["embeddings"][:]

    meta_df = pd.read_parquet(
        output_dir / "extract_embeddings" / "cascade_index.parquet"
    )
    assign_df = pd.read_parquet(
        output_dir / "cluster" / "cluster_assignments.parquet"
    )
    df = meta_df.merge(assign_df, on="cascade_id").reset_index(drop=True)
    assert len(df) == embeddings.shape[0], "Embedding/index count mismatch"

    labels = df["cluster_label"].values
    cluster_ids = sorted(set(labels) - {-1})
    print(
        f"[explain] {len(df):,} cascades | {len(cluster_ids)} clusters | "
        f"{(labels == -1).sum():,} noise"
    )

    # Load model (needed for 3.2 and 3.4)
    model = _load_model(cfg, output_dir, dev)

    # Load full dataset (needed for 3.2 and 3.4)
    data_list: list[Data] = torch.load(
        output_dir / "build_pyg_dataset" / "cascade_dataset.pt",
        weights_only=False,
    )
    # Build cascade_id → data lookup
    cid_to_data: dict[int, Data] = {
        int(d.cascade_id.item()): d for d in data_list
    }

    report: dict = {
        "n_cascades": int(len(df)),
        "n_clusters": len(cluster_ids),
        "noise_fraction": float((labels == -1).sum() / len(df)),
        "model_checkpoint": str(output_dir / "train" / "checkpoints" / "best.pt"),
    }

    # ── Sub-step 3.1: Probing classifiers ────────────────────────────────
    if xcfg.get("run_probes", True):
        print("\n[explain] ── 3.1 Probing classifiers ──")
        probe_results = _run_probes(
            embeddings, df, labels, cluster_ids, xcfg, out_dir, fig_dir
        )
        report["probe_summary"] = probe_results
        print("[explain] Probing done.")

    # ── Sub-step 3.2: DiffPool assignment extraction ──────────────────────
    if xcfg.get("run_diffpool_analysis", True) and model.use_diffpool:
        print("\n[explain] ── 3.2 DiffPool assignment extraction ──")
        diffpool_summary = _run_diffpool_analysis(
            model, cid_to_data, df, labels, cluster_ids, dev, out_dir, fig_dir
        )
        report["diffpool_summary"] = diffpool_summary
        print("[explain] DiffPool analysis done.")
    else:
        print("[explain] Skipping DiffPool analysis (disabled or mean-pool model).")

    # ── Sub-step 3.3: Cluster statistical profiles ────────────────────────
    if xcfg.get("run_cluster_profiles", True):
        print("\n[explain] ── 3.3 Cluster statistical profiles ──")
        cluster_profiles = _run_cluster_profiles(
            embeddings, df, labels, cluster_ids, out_dir, fig_dir
        )
        report["cluster_profiles"] = cluster_profiles
        print("[explain] Cluster profiles done.")

    # ── Sub-step 3.4: Augmentation sensitivity ────────────────────────────
    if xcfg.get("run_augmentation_sensitivity", True):
        print("\n[explain] ── 3.4 Augmentation sensitivity ──")
        aug_sensitivity = _run_augmentation_sensitivity(
            model, cid_to_data, df, labels, cluster_ids, cfg, dev, xcfg, fig_dir
        )
        report["augmentation_sensitivity"] = aug_sensitivity
        print("[explain] Augmentation sensitivity done.")

    # ── Sub-step 3.5: Representative cascade visualisations ───────────────
    if xcfg.get("run_representative_viz", True):
        print("\n[explain] ── 3.5 Representative cascade visualisations ──")
        _run_representative_viz(
            embeddings, cid_to_data, df, labels, cluster_ids, fig_dir
        )
        print("[explain] Visualisations done.")

    # ── Write consolidated report (checkpoint) ────────────────────────────
    report_path = out_dir / "explanation_report.json"
    report_path.write_text(json.dumps(report, indent=2, cls=_NumpyEncoder))
    print(f"\n[explain] Explanation report → {report_path}")


# ===========================================================================
# 3.1 Probing Classifiers
# ===========================================================================


def _run_probes(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    xcfg: dict,
    out_dir: Path,
    fig_dir: Path,
) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    probe_props = xcfg.get(
        "probe_properties",
        ["cascade_depth", "cascade_breadth", "wiener_index", "follower_edge_ratio",
         "cascade_size", "event_window"],
    )
    n_boot = xcfg.get("probe_bootstrap_n", 200)
    small_cluster_thresh = 100  # use CV for clusters below this size

    # Categorical properties → classification; others → regression
    categorical_props = {"event_window"}

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    results: dict = {}
    heatmap_data: dict[str, dict[str, float]] = {}  # prop → {global + per-cluster}

    for prop in probe_props:
        if prop not in df.columns:
            print(f"[explain/probe]   '{prop}' not in metadata, skipping")
            continue

        y = df[prop].values.astype(float)
        is_cat = prop in categorical_props

        print(f"[explain/probe]   {prop} ({'classification' if is_cat else 'regression'}) ...", end=" ", flush=True)

        # Global score
        if is_cat:
            y_int = y.astype(int)
            clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
            try:
                global_score = float(
                    np.mean(
                        cross_val_score(clf, emb_scaled, y_int, cv=5, scoring="balanced_accuracy")
                    )
                )
            except Exception:
                global_score = float("nan")
            metric_name = "balanced_accuracy"
        else:
            reg = Ridge(alpha=1.0)
            try:
                global_score = float(
                    np.mean(cross_val_score(reg, emb_scaled, y, cv=5, scoring="r2"))
                )
            except Exception:
                global_score = float("nan")
            metric_name = "r2"

        # Bootstrap CI on global score
        rng = np.random.default_rng(42)
        boot_scores: list[float] = []
        for _ in range(n_boot):
            idx = rng.choice(len(emb_scaled), size=len(emb_scaled), replace=True)
            try:
                if is_cat:
                    clf_b = LogisticRegression(max_iter=2000, random_state=0, n_jobs=-1)
                    cv_b = cross_val_score(clf_b, emb_scaled[idx], y_int[idx], cv=3, scoring="balanced_accuracy")
                else:
                    reg_b = Ridge(alpha=1.0)
                    cv_b = cross_val_score(reg_b, emb_scaled[idx], y[idx], cv=3, scoring="r2")
                boot_scores.append(float(np.mean(cv_b)))
            except Exception:
                pass
        boot_arr = np.array(boot_scores) if boot_scores else np.array([global_score])
        ci = (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)))

        # Per-cluster scores
        per_cluster: dict[str, float] = {}
        for k in cluster_ids:
            mask = labels == k
            n_k = mask.sum()
            if n_k < 20:
                per_cluster[str(k)] = float("nan")
                continue
            emb_k = emb_scaled[mask]
            y_k = y[mask]
            n_splits = 5 if n_k >= small_cluster_thresh else 3
            try:
                if is_cat:
                    # Need at least 2 classes in cluster
                    if len(set(y_k.astype(int))) < 2:
                        per_cluster[str(k)] = float("nan")
                        continue
                    clf_k = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
                    skf = StratifiedKFold(n_splits=n_splits)
                    scores_k = cross_val_score(clf_k, emb_k, y_k.astype(int), cv=skf, scoring="balanced_accuracy")
                else:
                    reg_k = Ridge(alpha=1.0)
                    scores_k = cross_val_score(reg_k, emb_k, y_k, cv=n_splits, scoring="r2")
                per_cluster[str(k)] = float(np.mean(scores_k))
            except Exception:
                per_cluster[str(k)] = float("nan")

        results[prop] = {
            "metric": metric_name,
            "global_score": global_score,
            "global_ci_lo": ci[0],
            "global_ci_hi": ci[1],
            "per_cluster": per_cluster,
        }
        heatmap_data[prop] = {"global": global_score, **{f"C{k}": v for k, v in per_cluster.items()}}
        print(f"global={global_score:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Save probe results
    probe_path = out_dir / "probe_results.json"
    probe_path.write_text(json.dumps(results, indent=2, cls=_NumpyEncoder))

    # Heatmap figure
    _plot_probe_heatmap(results, cluster_ids, fig_dir)

    return results


def _plot_probe_heatmap(results: dict, cluster_ids: list[int], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        props = list(results.keys())
        cols = ["Global"] + [f"C{k}" for k in cluster_ids]
        mat = np.full((len(props), len(cols)), np.nan)

        for i, prop in enumerate(props):
            mat[i, 0] = results[prop]["global_score"]
            for j, k in enumerate(cluster_ids):
                mat[i, j + 1] = results[prop]["per_cluster"].get(str(k), np.nan)

        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.7), max(4, len(props) * 0.6)))
        sns.heatmap(
            mat, annot=True, fmt=".2f", xticklabels=cols,
            yticklabels=props, cmap="RdYlGn", vmin=-0.1, vmax=1.0,
            ax=ax, linewidths=0.5,
        )
        ax.set_title("Probing Classifier Scores (R² / Balanced Accuracy)")
        plt.tight_layout()
        fig.savefig(fig_dir / "probe_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/probe] WARNING: heatmap failed ({e})")


# ===========================================================================
# 3.2 DiffPool Assignment Extraction
# ===========================================================================


def _run_diffpool_analysis(
    model: BlueCCGL,
    cid_to_data: dict[int, Data],
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    dev: torch.device,
    out_dir: Path,
    fig_dir: Path,
) -> dict:
    from torch_geometric.data import Batch

    model.eval()

    # Per-cascade aggregates (avoid storing raw assignment matrices)
    per_cascade_entropy: list[float] = []  # mean node assignment entropy at Level 1
    per_cascade_follower_in_dominant_pool: list[float] = []  # fraction of follower nodes in dominant super-node
    cascade_ids_processed: list[int] = []

    print(f"[explain/diffpool]   Processing {len(df):,} cascades (batch_size=1) ...")

    with torch.no_grad():
        for i, row in df.iterrows():
            cid = int(row["cascade_id"])
            data = cid_to_data.get(cid)
            if data is None:
                per_cascade_entropy.append(float("nan"))
                per_cascade_follower_in_dominant_pool.append(float("nan"))
                cascade_ids_processed.append(cid)
                continue

            batch = Batch.from_data_list([data]).to(dev)
            try:
                _, h, s1, s2, mask = model.explain(batch)
            except Exception:
                per_cascade_entropy.append(float("nan"))
                per_cascade_follower_in_dominant_pool.append(float("nan"))
                cascade_ids_processed.append(cid)
                continue

            if s1 is None:
                per_cascade_entropy.append(float("nan"))
                per_cascade_follower_in_dominant_pool.append(float("nan"))
                cascade_ids_processed.append(cid)
                continue

            # s1: [1, N_max, k1], mask: [1, N_max]
            n_nodes = int(mask[0].sum().item())
            s1_real = s1[0, :n_nodes, :]  # [n_nodes, k1]

            # Assignment probabilities + entropy
            s1_prob = torch.softmax(s1_real, dim=-1)  # [n_nodes, k1]
            eps = 1e-10
            entropy = -(s1_prob * (s1_prob + eps).log()).sum(dim=-1)  # [n_nodes]
            mean_entropy = float(entropy.mean().item())

            # Hard assignment: which super-node does each real node go to?
            assignments = s1_prob.argmax(dim=-1).cpu().numpy()  # [n_nodes]

            # Compute fraction of follower-typed edges whose target ends up in dominant pool
            edge_type = data.edge_type.numpy()  # [E]
            edge_tgt = data.edge_index[1].numpy()  # [E] — target node indices
            # Only consider targets within node range (should always hold for single graph)
            valid = edge_tgt < n_nodes
            if valid.sum() > 0:
                is_follower = edge_type[valid] == 0
                pool_ids = assignments[edge_tgt[valid]]
                dominant_pool = int(np.bincount(assignments).argmax())
                follower_in_dominant = float(
                    (is_follower & (pool_ids == dominant_pool)).sum()
                    / max(1, is_follower.sum())
                )
            else:
                follower_in_dominant = float("nan")

            per_cascade_entropy.append(mean_entropy)
            per_cascade_follower_in_dominant_pool.append(follower_in_dominant)
            cascade_ids_processed.append(cid)

            if (i + 1) % 500 == 0:
                print(f"[explain/diffpool]   {i + 1:,}/{len(df):,} ...", flush=True)

    # Save per-cascade arrays to HDF5
    h5_path = out_dir / "diffpool_assignments.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("cascade_ids", data=np.array(cascade_ids_processed, dtype=np.int64))
        f.create_dataset("mean_entropy_l1", data=np.array(per_cascade_entropy, dtype=np.float32))
        f.create_dataset(
            "follower_in_dominant_pool",
            data=np.array(per_cascade_follower_in_dominant_pool, dtype=np.float32),
        )
    print(f"[explain/diffpool]   Saved HDF5 → {h5_path}")

    # Per-cluster aggregation
    per_cluster: dict[str, dict] = {}
    ent_arr = np.array(per_cascade_entropy)
    fol_arr = np.array(per_cascade_follower_in_dominant_pool)

    for k in cluster_ids:
        mask_k = labels == k
        ent_k = ent_arr[mask_k]
        fol_k = fol_arr[mask_k]
        valid_ent = ent_k[~np.isnan(ent_k)]
        valid_fol = fol_k[~np.isnan(fol_k)]
        per_cluster[str(k)] = {
            "mean_entropy": float(np.mean(valid_ent)) if len(valid_ent) else None,
            "std_entropy": float(np.std(valid_ent)) if len(valid_ent) else None,
            "mean_follower_in_dominant_pool": float(np.mean(valid_fol)) if len(valid_fol) else None,
        }

    summary = {"per_cluster": per_cluster}

    # Plot
    _plot_diffpool_analysis(per_cluster, cluster_ids, fig_dir)

    return summary


def _plot_diffpool_analysis(per_cluster: dict, cluster_ids: list[int], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        means_ent = [per_cluster[str(k)]["mean_entropy"] or 0.0 for k in cluster_ids]
        stds_ent = [per_cluster[str(k)]["std_entropy"] or 0.0 for k in cluster_ids]
        fol_dom = [per_cluster[str(k)]["mean_follower_in_dominant_pool"] or 0.0 for k in cluster_ids]

        x = np.arange(len(cluster_ids))
        labels_x = [f"C{k}" for k in cluster_ids]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(x, means_ent, yerr=stds_ent, capsize=4, color="steelblue", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels_x, rotation=45, ha="right")
        axes[0].set_ylabel("Mean Node Assignment Entropy (Level 1)")
        axes[0].set_title("DiffPool Assignment Certainty per Archetype\n(lower = more decisive pooling)")

        axes[1].bar(x, fol_dom, color="coral", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels_x, rotation=45, ha="right")
        axes[1].set_ylabel("Fraction of Follower Edges to Dominant Super-node")
        axes[1].set_title("Follower Edge Concentration in Dominant Pool\n(higher = model groups follower nodes together)")
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        fig.savefig(fig_dir / "diffpool_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/diffpool] WARNING: plot failed ({e})")


# ===========================================================================
# 3.3 Cluster Statistical Profiles
# ===========================================================================


def _run_cluster_profiles(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    out_dir: Path,
    fig_dir: Path,
) -> dict:
    structural_props = [
        "cascade_depth", "cascade_breadth", "wiener_index",
        "follower_edge_ratio", "cascade_size",
        "sentiment_neg", "sentiment_neu", "sentiment_pos",
    ]
    # Only use props actually in df
    structural_props = [p for p in structural_props if p in df.columns]

    non_noise_mask = labels != -1
    df_clean = df[non_noise_mask].copy()
    labs_clean = labels[non_noise_mask]
    emb_clean = embeddings[non_noise_mask]

    # ── Per-cluster descriptive statistics ───────────────────────────────
    per_cluster: dict[str, dict] = {}
    for k in cluster_ids:
        mask_k = labs_clean == k
        df_k = df_clean[mask_k]
        stats: dict = {"n": int(mask_k.sum())}
        for prop in structural_props:
            vals = df_k[prop].dropna().values.astype(float)
            stats[prop] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
            }
        per_cluster[str(k)] = stats

    # ── Kruskal-Wallis between-cluster tests ─────────────────────────────
    significance: dict[str, dict] = {}
    for prop in structural_props:
        groups = [
            df_clean[labs_clean == k][prop].dropna().values.astype(float)
            for k in cluster_ids
        ]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue
        try:
            h_stat, p_val = scipy_stats.kruskal(*groups)
            significance[prop] = {
                "kruskal_H": float(h_stat),
                "p_value": float(p_val),
                "significant_p05": bool(p_val < 0.05),
            }
        except Exception:
            pass

    # ── Pairwise Mann-Whitney U with Bonferroni correction ────────────────
    n_pairs = len(cluster_ids) * (len(cluster_ids) - 1) // 2
    pairwise: dict[str, list[dict]] = {}
    for prop in structural_props:
        pairs: list[dict] = []
        for i, ki in enumerate(cluster_ids):
            for kj in cluster_ids[i + 1:]:
                a = df_clean[labs_clean == ki][prop].dropna().values.astype(float)
                b = df_clean[labs_clean == kj][prop].dropna().values.astype(float)
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    u_stat, p_raw = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
                    p_bonf = min(1.0, float(p_raw) * n_pairs)
                    pairs.append({
                        "cluster_a": ki, "cluster_b": kj,
                        "U": float(u_stat),
                        "p_raw": float(p_raw),
                        "p_bonferroni": p_bonf,
                        "significant_bonf": p_bonf < 0.05,
                    })
                except Exception:
                    pass
        pairwise[prop] = pairs

    # ── Archetype × event_window cross-tabulation ─────────────────────────
    # SRQ1 + SRQ2 analysis
    cross_tab: dict = {}
    if "event_window" in df_clean.columns:
        ct = pd.crosstab(labs_clean, df_clean["event_window"].values)
        cross_tab["counts"] = ct.to_dict()

        # Normalised by event (column): what fraction of each event's cascades go to each archetype?
        ct_norm_event = (ct.div(ct.sum(axis=0), axis=1)).fillna(0)
        cross_tab["fraction_per_event"] = ct_norm_event.to_dict()

        # Chi-squared test for independence
        try:
            chi2, p_chi, dof, expected = scipy_stats.chi2_contingency(ct.values)
            # Flag cells with expected < 5
            low_expected = int((expected < 5).sum())
            cross_tab["chi2"] = {
                "statistic": float(chi2),
                "p_value": float(p_chi),
                "dof": int(dof),
                "n_cells_expected_lt5": low_expected,
                "interpretation": (
                    "archetypes and event windows are NOT independent"
                    if p_chi < 0.05
                    else "no significant association detected"
                ),
            }
        except Exception as e:
            cross_tab["chi2"] = {"error": str(e)}

    # ── PCA discriminative dimensions ─────────────────────────────────────
    pca_result: dict = {}
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        centroids = np.stack([emb_clean[labs_clean == k].mean(axis=0) for k in cluster_ids])
        scaler_pca = StandardScaler()
        centroids_scaled = scaler_pca.fit_transform(centroids)
        pca = PCA(n_components=min(5, len(cluster_ids) - 1, centroids.shape[1]))
        pca.fit(centroids_scaled)
        # Correlate PCA dims with structural properties
        pca_proj = pca.transform(scaler_pca.transform(emb_clean))
        prop_corrs: dict[str, list[float]] = {}
        for prop in structural_props:
            vals = df_clean[prop].values.astype(float)
            corrs = []
            for d in range(pca_proj.shape[1]):
                valid = ~np.isnan(vals)
                if valid.sum() > 10:
                    r, _ = scipy_stats.pearsonr(pca_proj[valid, d], vals[valid])
                    corrs.append(float(r))
                else:
                    corrs.append(float("nan"))
            prop_corrs[prop] = corrs
        pca_result = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "property_correlations_with_pca_dims": prop_corrs,
        }
    except Exception as e:
        print(f"[explain/profiles] WARNING: PCA failed ({e})")

    profiles = {
        "per_cluster": per_cluster,
        "significance": significance,
        "pairwise_mannwhitney": pairwise,
        "archetype_event_crosstab": cross_tab,
        "pca_discriminative": pca_result,
    }
    (out_dir / "cluster_profiles.json").write_text(json.dumps(profiles, indent=2, cls=_NumpyEncoder))

    # ── Figures ───────────────────────────────────────────────────────────
    _plot_radar_profiles(per_cluster, structural_props, cluster_ids, fig_dir)
    _plot_follower_violins(df_clean, labs_clean, cluster_ids, fig_dir)
    _plot_archetype_event_heatmap(cross_tab, cluster_ids, fig_dir)
    _plot_temporal_evolution(df_clean, labs_clean, cluster_ids, fig_dir)

    return profiles


def _plot_radar_profiles(
    per_cluster: dict, props: list[str], cluster_ids: list[int], fig_dir: Path
) -> None:
    """Radar chart: normalised mean per structural property per cluster."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Normalise each property to [0, 1] across clusters
        means = {
            prop: np.array([per_cluster[str(k)][prop]["mean"] for k in cluster_ids if prop in per_cluster[str(k)]])
            for prop in props
            if all(prop in per_cluster[str(k)] for k in cluster_ids)
        }
        valid_props = [p for p in props if p in means and len(means[p]) == len(cluster_ids)]
        if not valid_props:
            return

        for prop in valid_props:
            lo, hi = means[prop].min(), means[prop].max()
            rng = hi - lo if hi != lo else 1.0
            means[prop] = (means[prop] - lo) / rng

        N = len(valid_props)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        colours = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))

        for k, col in zip(cluster_ids, colours):
            vals = [means[p][cluster_ids.index(k)] for p in valid_props]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=1.5, color=col, label=f"C{k}")
            ax.fill(angles, vals, alpha=0.1, color=col)

        ax.set_thetagrids(np.degrees(angles[:-1]), valid_props)
        ax.set_title("Normalised Archetype Profiles", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        fig.savefig(fig_dir / "cluster_radar_profiles.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/profiles] WARNING: radar plot failed ({e})")


def _plot_follower_violins(
    df_clean: pd.DataFrame, labels: np.ndarray, cluster_ids: list[int], fig_dir: Path
) -> None:
    """Violin plot of follower_edge_ratio per archetype (primary SRQ3 figure)."""
    if "follower_edge_ratio" not in df_clean.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data_by_cluster = [
            df_clean[labels == k]["follower_edge_ratio"].dropna().values
            for k in cluster_ids
        ]
        data_by_cluster = [d for d in data_by_cluster if len(d) > 1]
        if not data_by_cluster:
            return

        fig, ax = plt.subplots(figsize=(max(8, len(cluster_ids) * 0.8), 5))
        parts = ax.violinplot(data_by_cluster, positions=range(len(data_by_cluster)), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(data_by_cluster)))
        ax.set_xticklabels([f"C{k}" for k in cluster_ids[: len(data_by_cluster)]])
        ax.set_ylabel("Follower Edge Ratio")
        ax.set_title(
            "Follower Edge Ratio per Archetype\n"
            "(high = organic follower-driven spread; low = algorithmic/cross-community)"
        )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        fig.savefig(fig_dir / "follower_ratio_violins.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/profiles] WARNING: violin plot failed ({e})")


def _plot_archetype_event_heatmap(cross_tab: dict, cluster_ids: list[int], fig_dir: Path) -> None:
    """Heatmap: fraction of each event's cascades assigned to each archetype (SRQ1, SRQ2)."""
    if not cross_tab or "fraction_per_event" not in cross_tab:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        frac = cross_tab["fraction_per_event"]
        # frac: {event_window: {cluster_id: fraction, ...}, ...}
        event_cols = sorted(frac.keys())
        mat = pd.DataFrame(
            {ev: [frac[ev].get(k, 0.0) for k in cluster_ids] for ev in event_cols},
            index=[f"C{k}" for k in cluster_ids],
        )
        # Rename columns to human-readable event labels
        mat.columns = [EVENT_LABELS.get(int(e), str(e)) for e in mat.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(mat.columns) * 1.2), max(5, len(cluster_ids) * 0.7)))
        sns.heatmap(
            mat, annot=True, fmt=".2f", cmap="Blues", ax=ax,
            linewidths=0.5, vmin=0, vmax=1,
        )
        ax.set_title(
            "Archetype Distribution per Event Window\n"
            "(fraction of each event's cascades assigned to each archetype)"
        )
        ax.set_xlabel("Event Window")
        ax.set_ylabel("Archetype")
        plt.tight_layout()
        fig.savefig(fig_dir / "archetype_event_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/profiles] WARNING: event heatmap failed ({e})")


def _plot_temporal_evolution(
    df_clean: pd.DataFrame, labels: np.ndarray, cluster_ids: list[int], fig_dir: Path
) -> None:
    """Archetype prevalence by event_window over the conflict timeline (SRQ1)."""
    if "event_window" not in df_clean.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ew = df_clean["event_window"].values.astype(int)
        event_order = sorted(set(ew))
        event_labels_ordered = [EVENT_LABELS.get(e, str(e)) for e in event_order]

        # Fraction of each event's cascades per archetype
        mat = np.zeros((len(cluster_ids), len(event_order)))
        for j, e in enumerate(event_order):
            mask_e = ew == e
            total = max(1, mask_e.sum())
            for i, k in enumerate(cluster_ids):
                mat[i, j] = float((labels[mask_e] == k).sum()) / total

        fig, ax = plt.subplots(figsize=(12, 5))
        colours = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))
        for i, (k, col) in enumerate(zip(cluster_ids, colours)):
            ax.plot(event_labels_ordered, mat[i], "o-", label=f"C{k}", color=col, linewidth=1.5)

        ax.set_xlabel("Event Window (conflict timeline)")
        ax.set_ylabel("Fraction of Event's Cascades")
        ax.set_title("Archetype Prevalence Over the Conflict Timeline")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(fig_dir / "temporal_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/profiles] WARNING: temporal evolution plot failed ({e})")


# ===========================================================================
# 3.4 Augmentation Sensitivity
# ===========================================================================


def _run_augmentation_sensitivity(
    model: BlueCCGL,
    cid_to_data: dict[int, Data],
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    cfg: dict,
    dev: torch.device,
    xcfg: dict,
    fig_dir: Path,
) -> dict:
    from torch_geometric.data import Batch
    from blueccgl.augmentations.follower_subgraph import augment as aug_A
    from blueccgl.augmentations.temporal_prefix import augment as aug_B
    from blueccgl.augmentations.bridge_removal import augment as aug_C

    acfg = cfg.get("augmentation", {})
    n_reps = xcfg.get("n_representatives_per_cluster", 20)
    min_nodes = acfg.get("min_nodes_after_aug", 3)
    fol_ratio = acfg.get("follower_retain_ratio", 0.5)
    temp_ratio = acfg.get("temporal_prefix_ratio", 0.4)
    bridge_ratio = acfg.get("bridge_remove_ratio", 0.5)

    aug_fns = {
        "follower_subgraph": lambda d, seed: _seeded_aug(aug_A, d, seed, retain_ratio=fol_ratio, min_nodes=min_nodes),
        "temporal_prefix":   lambda d, seed: _seeded_aug(aug_B, d, seed, prefix_ratio=temp_ratio, min_nodes=min_nodes),
        "bridge_removal":    lambda d, seed: _seeded_aug(aug_C, d, seed, remove_ratio=bridge_ratio, min_nodes=min_nodes),
    }

    results: dict[str, dict] = {}
    model.eval()

    with torch.no_grad():
        for k in cluster_ids:
            mask_k = labels == k
            candidate_ids = df[mask_k]["cascade_id"].values

            # Sample n_reps; prefer non-noise
            rng = np.random.default_rng(42 + k)
            sampled_ids = rng.choice(
                candidate_ids, size=min(n_reps, len(candidate_ids)), replace=False
            )

            per_aug: dict[str, list[float]] = {name: [] for name in aug_fns}

            for idx_in_sample, cid in enumerate(sampled_ids):
                data = cid_to_data.get(int(cid))
                if data is None:
                    continue

                # Original embedding
                batch_orig = Batch.from_data_list([data]).to(dev)
                emb_orig = model.encode(batch_orig).cpu().numpy()[0]  # [64]

                seed_cascade = int(cid) * 137 % (2 ** 31)

                for aug_name, aug_fn in aug_fns.items():
                    aug_data = aug_fn(data, seed_cascade)
                    if aug_data is None:
                        continue  # augmentation returned None (too small)
                    try:
                        batch_aug = Batch.from_data_list([aug_data]).to(dev)
                        emb_aug = model.encode(batch_aug).cpu().numpy()[0]
                        dist = float(np.linalg.norm(emb_orig - emb_aug))
                        per_aug[aug_name].append(dist)
                    except Exception:
                        pass

            cluster_result: dict = {}
            for aug_name, dists in per_aug.items():
                if dists:
                    cluster_result[aug_name] = {
                        "mean": float(np.mean(dists)),
                        "std": float(np.std(dists)),
                        "n": len(dists),
                    }
                else:
                    cluster_result[aug_name] = {"mean": None, "std": None, "n": 0}

            results[str(k)] = cluster_result
            aug_summary = "  ".join(
                f"{n}={cluster_result[n]['mean']:.3f}" if cluster_result[n]["mean"] else f"{n}=N/A"
                for n in aug_fns
            )
            print(f"[explain/aug_sensitivity]   C{k}: {aug_summary}")

    aug_path = fig_dir.parent / "augmentation_sensitivity.json"
    aug_path.write_text(json.dumps(results, indent=2, cls=_NumpyEncoder))

    _plot_aug_sensitivity(results, cluster_ids, list(aug_fns.keys()), fig_dir)

    return results


def _seeded_aug(aug_fn, data: Data, seed: int, **kwargs):
    """Call aug_fn with a fixed per-cascade torch + random seed."""
    import random as _random
    torch.manual_seed(seed)
    _random.seed(seed)
    return aug_fn(data, **kwargs)


def _plot_aug_sensitivity(
    results: dict, cluster_ids: list[int], aug_names: list[str], fig_dir: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(cluster_ids))
        width = 0.25
        colours = ["#4C72B0", "#DD8452", "#55A868"]

        fig, ax = plt.subplots(figsize=(max(10, len(cluster_ids)), 5))
        for i, (aug_name, col) in enumerate(zip(aug_names, colours)):
            means = [
                results[str(k)][aug_name]["mean"] or 0.0
                for k in cluster_ids
            ]
            errs = [
                results[str(k)][aug_name]["std"] or 0.0
                for k in cluster_ids
            ]
            ax.bar(x + i * width, means, width, yerr=errs, label=aug_name.replace("_", " ").title(),
                   color=col, alpha=0.85, capsize=3)

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"C{k}" for k in cluster_ids], rotation=45, ha="right")
        ax.set_ylabel("||embed(original) - embed(augmented)||₂")
        ax.set_title(
            "Augmentation Sensitivity per Archetype\n"
            "(higher = archetype strongly relies on that structural signal)"
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / "augmentation_sensitivity_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[explain/aug_sensitivity] WARNING: plot failed ({e})")


# ===========================================================================
# 3.5 Representative Cascade Visualisations
# ===========================================================================


def _run_representative_viz(
    embeddings: np.ndarray,
    cid_to_data: dict[int, Data],
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    fig_dir: Path,
    n_reps: int = 5,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import networkx as nx
    except ImportError as e:
        print(f"[explain/viz] WARNING: networkx/matplotlib unavailable ({e}). Skipping.")
        return

    arch_dir = fig_dir / "archetypes"

    non_noise_mask = labels != -1
    emb_clean = embeddings[non_noise_mask]
    df_clean = df[non_noise_mask].reset_index(drop=True)
    labs_clean = labels[non_noise_mask]

    for k in cluster_ids:
        mask_k = labs_clean == k
        emb_k = emb_clean[mask_k]
        df_k = df_clean[mask_k].reset_index(drop=True)

        if len(df_k) == 0:
            continue

        # Closest n_reps to centroid
        centroid = emb_k.mean(axis=0)
        dists = np.linalg.norm(emb_k - centroid, axis=1)
        rep_idxs = np.argsort(dists)[:n_reps]

        for rep_i, idx in enumerate(rep_idxs):
            cid = int(df_k.iloc[idx]["cascade_id"])
            data = cid_to_data.get(cid)
            if data is None:
                continue

            _draw_cascade_tree(data, k, rep_i, arch_dir)

    print(f"[explain/viz] Representative cascade trees saved → {arch_dir}")


def _draw_cascade_tree(data: Data, cluster_k: int, rep_i: int, arch_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx

    edge_index = data.edge_index.numpy()
    edge_type = data.edge_type.numpy()
    x = data.x.numpy()  # [N, 6]: [temporal, sent_label, sent_score, depth, in_deg, out_deg]
    N = x.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for e in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, e]), int(edge_index[1, e]), etype=int(edge_type[e]))

    # Hierarchical tree layout centred on root (node 0)
    try:
        pos = _hierarchy_pos(G, root=0)
    except Exception:
        pos = nx.spring_layout(G, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_configs = [
        ("Edge Type",    {0: "#2196F3", 1: "#FF5722", 2: "#9E9E9E"},   [int(et) for et in data.edge_type.numpy()]),
        ("Depth",        None,   x[:, 3].tolist()),  # depth feature
        ("Sentiment",    {0: "#EF5350", 1: "#78909C", 2: "#66BB6A"},   [int(sl) for sl in x[:, 1].tolist()]),  # sent_label
    ]

    for ax, (title, colour_map, node_vals) in zip(axes, panel_configs):
        if colour_map is not None:
            node_colours = [colour_map.get(v, "#BDBDBD") for v in node_vals]
        else:
            # Continuous: map depth to colourmap
            vals = np.array(node_vals, dtype=float)
            norm = plt.Normalize(vmin=vals.min(), vmax=max(vals.max(), vals.min() + 1))
            node_colours = [cm.Blues(norm(v)) for v in vals]

        nx.draw_networkx(
            G, pos=pos, ax=ax, node_color=node_colours,
            with_labels=False, node_size=max(10, 300 // max(N, 1)),
            arrows=True, arrowsize=8, edge_color="#BDBDBD", width=0.5,
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Archetype C{cluster_k} — Representative #{rep_i + 1} "
        f"(n={N} nodes, depth={int(x[:, 3].max())})",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = arch_dir / f"cluster_{cluster_k}_rep_{rep_i}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hierarchy_pos(G: "nx.DiGraph", root: int, width: float = 1.0, vert_gap: float = 0.2,
                   vert_loc: float = 0.0, xcenter: float = 0.5) -> dict:
    """Reingold-Tilford inspired hierarchical tree layout for directed trees."""
    import networkx as nx

    # Use undirected BFS to build depth layers
    if not nx.is_weakly_connected(G):
        # Only draw the root component
        comp = nx.node_connected_component(G.to_undirected(), root)
        G = G.subgraph(comp)

    pos = {}
    bfs_tree = nx.bfs_tree(G.to_undirected(), root)
    levels: dict[int, list[int]] = {}
    for node in nx.bfs_tree(G.to_undirected(), root).nodes():
        depth = nx.shortest_path_length(bfs_tree, root, node)
        levels.setdefault(depth, []).append(node)

    for depth, nodes in levels.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i + 1) / (n + 1), -depth * vert_gap)

    return pos


# ===========================================================================
# Utilities
# ===========================================================================


def _load_model(cfg: dict, output_dir: Path, dev: torch.device) -> BlueCCGL:
    ckpt_path = output_dir / "train" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=dev)
    model = BlueCCGL(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"[explain] Model loaded: epoch={epoch}  loss={loss:.4f}")
    return model
