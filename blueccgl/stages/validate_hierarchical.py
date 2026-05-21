"""
--stage validate_hierarchical

Validates the macro-archetype solution from --stage hierarchical using four
fully implemented methods and one stub:

  1. Cophenetic correlation  — how faithfully the Ward dendrogram represents
                               pairwise centroid distances (single scalar)
  2. Multi-cut metric sweep  — silhouette / CH / DB across k=2..6 macro cuts;
                               determines whether the selected k is optimal
  3. Bootstrap stability     — ARI across N resampled centroid recomputations;
                               tests sensitivity of macro assignments to which
                               cascades happen to be in the dataset
  4. UMAP seed stability     — ARI across M UMAP re-runs with different random
                               seeds; tests sensitivity to UMAP stochasticity
  5. Prediction strength     — [STUB] Tibshirani & Walther (2005) held-out
                               cluster validation; the closest clustering analog
                               to k-fold cross-validation

Reads:
  {output_dir}/cluster/umap_embeddings_cluster.h5    — 10D UMAP embeddings [N, 10]
  {output_dir}/cluster/cluster_assignments.parquet   — HDBSCAN cluster labels
  {output_dir}/cluster/cluster_persistences.json     — best mcs used
  {output_dir}/extract_embeddings/embeddings.h5      — raw 64D embeddings (UMAP re-runs)
  {output_dir}/hierarchical/hierarchical_results.json — n_macro, cut_height

Writes:
  {output_dir}/validate_hierarchical/validation_report.json
  {output_dir}/validate_hierarchical/figures/metric_sweep.png
  {output_dir}/validate_hierarchical/figures/bootstrap_stability.png
  {output_dir}/validate_hierarchical/figures/umap_seed_stability.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ── shared helper ─────────────────────────────────────────────────────────────

def _cascade_macro_labels(
    hdbscan_labels: np.ndarray,
    cluster_ids: list[int],
    macro_per_cluster: np.ndarray,  # 1-indexed output of fcluster, length = len(cluster_ids)
) -> np.ndarray:
    """Return per-cascade macro label array; noise cascades get -1."""
    out = np.full(len(hdbscan_labels), -1, dtype=int)
    for i, cid in enumerate(cluster_ids):
        out[hdbscan_labels == cid] = int(macro_per_cluster[i]) - 1
    return out


# ── validation 1: cophenetic correlation ─────────────────────────────────────

def _cophenetic(Z: np.ndarray, centroids: np.ndarray) -> float:
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    cc, _ = cophenet(Z, pdist(centroids, metric="euclidean"))
    return float(cc)


# ── validation 2: multi-cut internal metric sweep ────────────────────────────

def _metric_sweep(
    umap_emb: np.ndarray,
    hdbscan_labels: np.ndarray,
    cluster_ids: list[int],
    centroids: np.ndarray,
    n_macro_selected: int,
    k_range: range,
) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    Z = linkage(pdist(centroids, metric="euclidean"), method="ward")

    # Work only on non-noise cascades
    non_noise_mask = hdbscan_labels >= 0
    emb_nn = umap_emb[non_noise_mask]
    labels_nn = hdbscan_labels[non_noise_mask]

    k_vals, sils, chs, dbs = [], [], [], []

    for k in k_range:
        if k > len(cluster_ids):
            break
        macro_per_cluster = fcluster(Z, t=k, criterion="maxclust")
        cascade_macro = _cascade_macro_labels(labels_nn, cluster_ids, macro_per_cluster)
        valid_mask = cascade_macro >= 0
        emb_v = emb_nn[valid_mask]
        labels_v = cascade_macro[valid_mask]

        if len(np.unique(labels_v)) < 2:
            continue

        # Silhouette is O(N^2) — subsample for speed
        if len(labels_v) > 8_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(labels_v), 8_000, replace=False)
            sil = silhouette_score(emb_v[idx], labels_v[idx])
        else:
            sil = silhouette_score(emb_v, labels_v)

        ch = calinski_harabasz_score(emb_v, labels_v)
        db = davies_bouldin_score(emb_v, labels_v)

        k_vals.append(k)
        sils.append(float(sil))
        chs.append(float(ch))
        dbs.append(float(db))

    return {
        "k_values":          k_vals,
        "silhouette":        sils,
        "calinski_harabasz": chs,
        "davies_bouldin":    dbs,
        "selected_k":        n_macro_selected,
    }


# ── validation 3: bootstrap stability ────────────────────────────────────────

def _bootstrap_stability(
    umap_emb: np.ndarray,
    hdbscan_labels: np.ndarray,
    cluster_ids: list[int],
    original_cascade_macro: np.ndarray,
    n_macro: int,
    n_iter: int,
    frac: float,
    seed: int,
) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    ari_values: list[float] = []

    # Work on non-noise slice (same mask for both original and bootstrap)
    non_noise_mask = hdbscan_labels >= 0
    orig_nn = original_cascade_macro[non_noise_mask]
    labels_nn = hdbscan_labels[non_noise_mask]
    emb_nn = umap_emb[non_noise_mask]

    for _ in range(n_iter):
        # Resample `frac` of cascades within each HDBSCAN cluster → new centroid
        new_centroids = []
        for cid in cluster_ids:
            idx = np.where(labels_nn == cid)[0]
            n_keep = max(2, int(len(idx) * frac))
            sampled = rng.choice(idx, size=n_keep, replace=False)
            new_centroids.append(emb_nn[sampled].mean(axis=0))
        new_centroids_arr = np.stack(new_centroids)

        Z_b = linkage(pdist(new_centroids_arr, metric="euclidean"), method="ward")
        boot_macro_per_cluster = fcluster(Z_b, t=n_macro, criterion="maxclust")
        boot_cascade_macro = _cascade_macro_labels(labels_nn, cluster_ids, boot_macro_per_cluster)

        # ARI is permutation-invariant — no Hungarian matching needed
        ari = adjusted_rand_score(orig_nn, boot_cascade_macro)
        ari_values.append(float(ari))

    return {
        "n_iterations": n_iter,
        "resample_frac": frac,
        "ari_values":   ari_values,
        "mean_ari":     float(np.mean(ari_values)),
        "std_ari":      float(np.std(ari_values)),
        "min_ari":      float(np.min(ari_values)),
        "max_ari":      float(np.max(ari_values)),
    }


# ── validation 4: UMAP seed stability ────────────────────────────────────────

def _umap_seed_stability(
    raw_emb: np.ndarray,
    original_cascade_macro: np.ndarray,  # per cascade, noise = -1
    n_macro: int,
    umap_params: dict,
    best_mcs: int,
    min_samples_ratio: float,
    seeds: list[int],
) -> dict:
    try:
        import umap as umap_lib
    except ImportError:
        return {"status": "skipped — umap-learn not installed", "ari_values": []}
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        return {"status": "skipped — hdbscan not installed", "ari_values": []}

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score

    ari_values: list[float] = []
    n_valid = 0

    for seed in seeds:
        print(f"  [validate_hierarchical] UMAP seed={seed} ...", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap_lib.UMAP(
                n_components=umap_params["n_components"],
                n_neighbors=umap_params["n_neighbors"],
                min_dist=umap_params["min_dist"],
                random_state=seed,
            )
            new_umap = reducer.fit_transform(raw_emb)

        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=best_mcs,
            min_samples=max(1, round(min_samples_ratio * best_mcs)),
        )
        new_labels = clusterer.fit_predict(new_umap)

        new_cluster_ids = sorted(k for k in np.unique(new_labels) if k >= 0)
        if len(new_cluster_ids) < 2:
            print(f"    degenerate run (0–1 clusters) — skipped")
            continue

        new_centroids = np.stack(
            [new_umap[new_labels == k].mean(axis=0) for k in new_cluster_ids]
        )
        cut_k = min(n_macro, len(new_cluster_ids))
        Z_s = linkage(pdist(new_centroids, metric="euclidean"), method="ward")
        new_macro_per_cluster = fcluster(Z_s, t=cut_k, criterion="maxclust")
        new_cascade_macro = _cascade_macro_labels(new_labels, new_cluster_ids, new_macro_per_cluster)

        # ARI on cascades non-noise in both original and new run
        both_valid = (original_cascade_macro >= 0) & (new_cascade_macro >= 0)
        if both_valid.sum() < 100:
            print(f"    too few overlapping non-noise cascades — skipped")
            continue

        ari = adjusted_rand_score(
            original_cascade_macro[both_valid],
            new_cascade_macro[both_valid],
        )
        ari_values.append(float(ari))
        n_valid += 1
        print(f"    ARI={ari:.3f}  clusters={len(new_cluster_ids)}")

    result: dict = {
        "seeds":       seeds,
        "n_valid_runs": n_valid,
        "ari_values":  ari_values,
    }
    if ari_values:
        result["mean_ari"] = float(np.mean(ari_values))
        result["std_ari"]  = float(np.std(ari_values))
        result["min_ari"]  = float(np.min(ari_values))
        result["max_ari"]  = float(np.max(ari_values))
    return result


# ── validation 5: prediction strength (STUB) ─────────────────────────────────

def _prediction_strength_stub() -> dict:
    """
    STUB — Tibshirani & Walther (2005) prediction strength.

    Algorithm (to implement):
      For rep in range(n_reps):
        1. Split cascades 50/50, stratified by HDBSCAN cluster → train_idx, test_idx
        2. Compute macro centroids from train_idx cascades (in 10D UMAP space)
        3. Assign test cascades to nearest train centroid → A_test (n_macro groups)
        4. Independently re-run Ward on test-set centroids → B_test
        5. For each pair of test cascades in the same macro under A_test,
           check if they are also in the same macro under B_test
        6. PS(k) = min over macros of (within-macro pair agreement fraction)
      Report: mean PS ± std across reps. Threshold: PS > 0.8 → valid solution.

    Reference:
      Tibshirani, R., & Walther, G. (2005). Cluster validation by prediction
      strength. Journal of Computational and Graphical Statistics, 14(3), 511–528.
    """
    return {"status": "stub — not yet implemented"}


# ── figures ───────────────────────────────────────────────────────────────────

def _plot_metric_sweep(sweep: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    k_vals = sweep["k_values"]
    selected_k = sweep["selected_k"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = [
        ("silhouette",        "Silhouette (↑)",        "steelblue"),
        ("calinski_harabasz", "Calinski-Harabasz (↑)", "seagreen"),
        ("davies_bouldin",    "Davies-Bouldin (↓)",    "firebrick"),
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        ax.plot(k_vals, sweep[key], marker="o", color=color, linewidth=2, markersize=7)
        ax.axvline(selected_k, color="black", linestyle="--", linewidth=1,
                   label=f"selected k={selected_k}")
        # Highlight the selected-k point
        if selected_k in k_vals:
            idx = k_vals.index(selected_k)
            ax.scatter([selected_k], [sweep[key][idx]], color="black", zorder=5, s=60)
        ax.set_xlabel("Number of macro-archetypes (k)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(k_vals)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Internal Metric Sweep Across Macro-Archetype Cuts", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[validate_hierarchical] saved metric_sweep.png")


def _plot_bootstrap_stability(boot: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    aris = boot["ari_values"]
    mean, std = boot["mean_ari"], boot["std_ari"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aris, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean, color="black",   linewidth=2,   label=f"mean={mean:.3f}")
    ax.axvline(mean - std, color="grey", linewidth=1.2, linestyle="--",
               label=f"±1 std ({std:.3f})")
    ax.axvline(mean + std, color="grey", linewidth=1.2, linestyle="--")
    ax.axvline(0.75, color="firebrick", linewidth=1.2, linestyle=":",
               label="stability threshold (0.75)")
    ax.set_xlabel("Adjusted Rand Index", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"Bootstrap Macro-Archetype Stability\n"
        f"(n={boot['n_iterations']} resamples at {boot['resample_frac']:.0%})", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[validate_hierarchical] saved bootstrap_stability.png")


def _plot_umap_seed_stability(seed_res: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    aris = seed_res.get("ari_values", [])
    if not aris:
        print("[validate_hierarchical] UMAP seed stability: no valid runs — figure skipped")
        return

    seeds = seed_res.get("seeds", list(range(len(aris))))[:len(aris)]
    mean = seed_res.get("mean_ari", float(np.mean(aris)))
    std  = seed_res.get("std_ari",  float(np.std(aris)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(aris)), aris, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axhline(mean, color="black",   linewidth=2,   label=f"mean={mean:.3f}")
    ax.axhline(mean - std, color="grey", linewidth=1.2, linestyle="--",
               label=f"±1 std ({std:.3f})")
    ax.axhline(mean + std, color="grey", linewidth=1.2, linestyle="--")
    ax.axhline(0.75, color="firebrick", linewidth=1.2, linestyle=":",
               label="stability threshold (0.75)")
    ax.set_xticks(range(len(aris)))
    ax.set_xticklabels([f"seed={s}" for s in seeds], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Adjusted Rand Index vs. original", fontsize=11)
    ax.set_title(
        f"UMAP Seed Stability ({seed_res.get('n_valid_runs', len(aris))} valid runs)", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "umap_seed_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[validate_hierarchical] saved umap_seed_stability.png")


# ── main entry point ──────────────────────────────────────────────────────────

def run(cfg: dict, output_dir: Path) -> None:
    import blueccgl.utils.mpl_compat  # noqa: F401 — Python 3.14 deepcopy patch
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    out_dir = output_dir / "validate_hierarchical"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    vcfg: dict = cfg.get("validate_hierarchical", {})
    n_bootstrap:       int   = vcfg.get("n_bootstrap",         100)
    bootstrap_frac:    float = vcfg.get("bootstrap_frac",      0.80)
    umap_seeds:        list  = vcfg.get("umap_seeds",          list(range(10)))
    k_min:             int   = vcfg.get("k_sweep_min",         2)
    k_max:             int   = vcfg.get("k_sweep_max",         6)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("[validate_hierarchical] loading inputs ...")

    with h5py.File(output_dir / "cluster" / "umap_embeddings_cluster.h5", "r") as f:
        umap_emb = f["embeddings"][:]                     # [N, 10]

    with h5py.File(output_dir / "extract_embeddings" / "embeddings.h5", "r") as f:
        raw_emb = f["embeddings"][:]                      # [N, 64]

    assignments = pd.read_parquet(
        output_dir / "cluster" / "cluster_assignments.parquet"
    )
    hdbscan_labels = assignments["cluster_label"].values  # [N]

    with open(output_dir / "cluster" / "cluster_persistences.json") as f:
        persistences = json.load(f)
    best_mcs: int = int(persistences.get("min_cluster_size_used", 300))

    with open(output_dir / "hierarchical" / "hierarchical_results.json") as f:
        hier = json.load(f)
    n_macro: int = int(hier["n_macro_archetypes"])

    cluster_ids = sorted(k for k in np.unique(hdbscan_labels) if k >= 0)
    n_clusters  = len(cluster_ids)
    n_noise     = int((hdbscan_labels == -1).sum())
    print(f"[validate_hierarchical] {n_clusters} HDBSCAN clusters, "
          f"{n_noise} noise, n_macro={n_macro}, N={len(umap_emb):,}")

    # ── Recompute Ward linkage (deterministic from centroids) ─────────────────
    centroids = np.stack([umap_emb[hdbscan_labels == k].mean(axis=0)
                          for k in cluster_ids])           # [K, 10]
    Z = linkage(pdist(centroids, metric="euclidean"), method="ward")

    # Original cascade-level macro labels (used as reference for ARI)
    orig_macro_per_cluster = fcluster(Z, t=n_macro, criterion="maxclust")
    original_cascade_macro = _cascade_macro_labels(
        hdbscan_labels, cluster_ids, orig_macro_per_cluster
    )

    # ── 1. Cophenetic correlation ─────────────────────────────────────────────
    print("[validate_hierarchical] (1/4) cophenetic correlation ...")
    cc = _cophenetic(Z, centroids)
    print(f"  cophenetic r = {cc:.4f}  ({'good' if cc > 0.75 else 'WARN: below 0.75'})")

    # ── 2. Multi-cut metric sweep ─────────────────────────────────────────────
    print(f"[validate_hierarchical] (2/4) metric sweep k={k_min}..{k_max} ...")
    sweep = _metric_sweep(
        umap_emb, hdbscan_labels, cluster_ids, centroids,
        n_macro, range(k_min, k_max + 1),
    )
    best_sil_k = sweep["k_values"][int(np.argmax(sweep["silhouette"]))]
    best_ch_k  = sweep["k_values"][int(np.argmax(sweep["calinski_harabasz"]))]
    best_db_k  = sweep["k_values"][int(np.argmin(sweep["davies_bouldin"]))]
    print(f"  best k by silhouette={best_sil_k}, CH={best_ch_k}, DB={best_db_k}  "
          f"(selected={n_macro})")

    # ── 3. Bootstrap stability ────────────────────────────────────────────────
    print(f"[validate_hierarchical] (3/4) bootstrap stability "
          f"(n={n_bootstrap}, frac={bootstrap_frac}) ...")
    boot = _bootstrap_stability(
        umap_emb, hdbscan_labels, cluster_ids, original_cascade_macro,
        n_macro, n_bootstrap, bootstrap_frac, seed=42,
    )
    status = "STABLE" if boot["mean_ari"] >= 0.75 else "WARN: below threshold"
    print(f"  mean ARI={boot['mean_ari']:.3f} ± {boot['std_ari']:.3f}  [{status}]")

    # ── 4. UMAP seed stability ────────────────────────────────────────────────
    ccfg: dict = cfg.get("clustering", {})
    umap_params = {
        "n_components": int(ccfg.get("umap_cluster_n_components", 10)),
        "n_neighbors":  int(ccfg.get("umap_n_neighbors",          15)),
        "min_dist":     float(ccfg.get("umap_min_dist",           0.1)),
    }
    min_samples_ratio: float = float(ccfg.get("min_samples_ratio", 0.4))

    print(f"[validate_hierarchical] (4/4) UMAP seed stability "
          f"({len(umap_seeds)} seeds) ...")
    seed_res = _umap_seed_stability(
        raw_emb, original_cascade_macro, n_macro,
        umap_params, best_mcs, min_samples_ratio, umap_seeds,
    )
    if seed_res.get("ari_values"):
        status_s = "STABLE" if seed_res["mean_ari"] >= 0.75 else "WARN: below threshold"
        print(f"  mean ARI={seed_res['mean_ari']:.3f} ± {seed_res['std_ari']:.3f}  [{status_s}]")

    # ── 5. Prediction strength (stub) ─────────────────────────────────────────
    ps = _prediction_strength_stub()
    print("[validate_hierarchical] (5/5) prediction strength: STUB — not yet implemented")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_metric_sweep(sweep, fig_dir)
    _plot_bootstrap_stability(boot, fig_dir)
    _plot_umap_seed_stability(seed_res, fig_dir)

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n[validate_hierarchical] ── VALIDATION SUMMARY ──")
    print(f"  Cophenetic r               : {cc:.4f}")
    print(f"  Metric sweep best k        : sil={best_sil_k} / CH={best_ch_k} / DB={best_db_k}  (selected={n_macro})")
    print(f"  Bootstrap ARI              : {boot['mean_ari']:.3f} ± {boot['std_ari']:.3f}  (n={n_bootstrap})")
    if seed_res.get("ari_values"):
        print(f"  UMAP seed ARI              : {seed_res['mean_ari']:.3f} ± {seed_res['std_ari']:.3f}  (n={seed_res['n_valid_runs']} runs)")
    else:
        print(f"  UMAP seed ARI              : {seed_res.get('status', 'n/a')}")
    print(f"  Prediction strength        : {ps['status']}")
    print()

    # ── Write checkpoint ──────────────────────────────────────────────────────
    report = {
        "n_macro_archetypes":    n_macro,
        "n_hdbscan_clusters":    n_clusters,
        "cophenetic_correlation": cc,
        "metric_sweep":          sweep,
        "bootstrap_stability":   {k: v for k, v in boot.items() if k != "ari_values"},
        "bootstrap_ari_values":  boot["ari_values"],
        "umap_seed_stability":   {k: v for k, v in seed_res.items() if k != "ari_values"},
        "umap_seed_ari_values":  seed_res.get("ari_values", []),
        "prediction_strength":   ps,
    }
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[validate_hierarchical] done — report saved → {report_path}")
