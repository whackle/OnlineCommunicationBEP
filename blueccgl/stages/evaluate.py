"""Stage: evaluate — Clustering quality metrics with bootstrap CIs + UMAP visualisation.

Reads:
  {output_dir}/extract_embeddings/embeddings.h5
  {output_dir}/extract_embeddings/cascade_index.parquet
  {output_dir}/cluster/cluster_assignments.parquet

Writes:
  {output_dir}/evaluate/metrics.json
  {output_dir}/evaluate/umap_figure.png
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
except ImportError:
    raise ImportError("scikit-learn not installed: pip install scikit-learn")

# Silhouette + other sklearn metrics are expensive on large N; cap the sample
_BOOTSTRAP_SAMPLE = 5_000
# DBCV is slow — cap bootstrap iterations separately
_DBCV_MAX_BOOT = 100


def run(cfg: dict, output_dir: Path) -> None:
    out_dir = output_dir / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)

    ecfg = cfg.get("evaluation", {})
    bootstrap_n: int = ecfg.get("bootstrap_n", 1000)
    dpi: int = cfg.get("explain", {}).get("figures_dpi", 300)

    # ── 1. Load data ──────────────────────────────────────────────────────
    emb_path = output_dir / "extract_embeddings" / "embeddings.h5"
    print(f"[evaluate] Loading embeddings from {emb_path}")
    with h5py.File(emb_path, "r") as f:
        embeddings: np.ndarray = f["embeddings"][:]

    meta_df = pd.read_parquet(
        output_dir / "extract_embeddings" / "cascade_index.parquet"
    )
    assign_df = pd.read_parquet(
        output_dir / "cluster" / "cluster_assignments.parquet"
    )
    df = meta_df.merge(assign_df, on="cascade_id")
    labels = df["cluster_label"].values

    # ── 2. Separate noise ─────────────────────────────────────────────────
    non_noise_mask = labels != -1
    emb_clean = embeddings[non_noise_mask]
    lab_clean = labels[non_noise_mask]
    n_clean = int(non_noise_mask.sum())
    n_noise = int((~non_noise_mask).sum())
    n_clusters = int(len(set(lab_clean)))

    print(
        f"[evaluate] {n_clean:,} non-noise | {n_clusters} clusters | "
        f"{n_noise:,} noise ({n_noise / len(labels):.2%})"
    )

    # Load cluster persistence (stability) saved by cluster stage
    persistence_path = output_dir / "cluster" / "cluster_persistences.json"
    mean_persistence: float | None = None
    if persistence_path.exists():
        pdata = json.loads(persistence_path.read_text())
        mean_persistence = pdata.get("mean_persistence")

    # ── 3. Bootstrap metric helper ────────────────────────────────────────
    sample_n = min(_BOOTSTRAP_SAMPLE, n_clean)
    rng = np.random.default_rng(42)

    def _bootstrap(metric_fn: callable, n_boot: int) -> dict:
        scores: list[float] = []
        for _ in range(n_boot):
            idx = rng.choice(n_clean, size=sample_n, replace=False)
            try:
                s = float(metric_fn(emb_clean[idx], lab_clean[idx]))
                scores.append(s)
            except Exception:
                pass
        if not scores:
            return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None, "n_boot": 0}
        arr = np.array(scores)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "n_boot": len(scores),
        }

    # ── 4. Compute metrics ────────────────────────────────────────────────
    print(f"[evaluate] Bootstrapping metrics (n={bootstrap_n}, sample={sample_n}) ...")
    metrics: dict = {}

    metrics["silhouette"] = _bootstrap(silhouette_score, bootstrap_n)
    metrics["davies_bouldin"] = _bootstrap(davies_bouldin_score, bootstrap_n)
    metrics["calinski_harabasz"] = _bootstrap(calinski_harabasz_score, bootstrap_n)

    # DBCV via hdbscan.validity_index (expensive — cap iterations)
    try:
        from hdbscan import validity_index  # type: ignore

        def _dbcv(emb, labs):
            return validity_index(emb.astype(np.float64), labs)

        metrics["dbcv"] = _bootstrap(_dbcv, min(bootstrap_n, _DBCV_MAX_BOOT))
        print(f"[evaluate] DBCV computed (capped at {min(bootstrap_n, _DBCV_MAX_BOOT)} bootstraps)")
    except (ImportError, Exception) as e:
        print(f"[evaluate] WARNING: DBCV unavailable ({e}). Setting to null.")
        metrics["dbcv"] = None

    metrics["meta"] = {
        "n_cascades_total": int(len(labels)),
        "n_cascades_clustered": n_clean,
        "n_clusters": n_clusters,
        "noise_count": n_noise,
        "noise_fraction": float(n_noise / len(labels)),
        "mean_cluster_persistence": mean_persistence,
        "min_cluster_size_used": (
            int(df["min_cluster_size_used"].iloc[0])
            if "min_cluster_size_used" in df.columns
            else None
        ),
        "bootstrap_sample_size": sample_n,
    }

    # ── Central metrics block ─────────────────────────────────────────────
    _SEP = "─" * 55
    print(f"\n{_SEP}")
    print("  CENTRAL METRICS")
    print(_SEP)
    print(f"  Number of clusters    : {n_clusters}")
    print(f"  Noise fraction        : {n_noise / len(labels):.2%}  ({n_noise:,} / {len(labels):,})")
    if mean_persistence is not None:
        print(f"  Cluster stability     : {mean_persistence:.4f}  (mean HDBSCAN persistence)")
    else:
        print("  Cluster stability     : n/a  (cluster_persistences.json not found)")
    dbcv_val = metrics.get("dbcv") or {}
    if isinstance(dbcv_val, dict) and dbcv_val.get("mean") is not None:
        print(f"  DBCV                  : {dbcv_val['mean']:.4f}  (density-based cluster validity)")
    else:
        print("  DBCV                  : n/a")
    print(f"{_SEP}\n")

    # ── Full bootstrapped metrics ─────────────────────────────────────────
    for name, v in metrics.items():
        if isinstance(v, dict) and v.get("mean") is not None:
            print(
                f"  {name:22s}: {v['mean']:.4f} ± {v['std']:.4f}  "
                f"95% CI [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]"
            )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] Metrics saved → {metrics_path}")

    # ── 5. UMAP visualisation ─────────────────────────────────────────────
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
        import umap  # type: ignore

        matplotlib.use("Agg")

        ccfg = cfg.get("clustering", {})
        n_neighbors = ccfg.get("umap_n_neighbors", 15)
        min_dist = ccfg.get("umap_min_dist", 0.1)

        print(f"[evaluate] UMAP projection (n_neighbors={n_neighbors}) ...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False,
        )
        z2d = reducer.fit_transform(embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel A: coloured by cluster label
        sc1 = axes[0].scatter(
            z2d[:, 0], z2d[:, 1],
            c=labels, cmap="tab20", s=4, alpha=0.6, linewidths=0,
        )
        axes[0].set_title(f"Cluster Assignments ({n_clusters} clusters)")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")
        plt.colorbar(sc1, ax=axes[0], label="Cluster")

        # Panel B: coloured by event_window
        ew = df["event_window"].values
        sc2 = axes[1].scatter(
            z2d[:, 0], z2d[:, 1],
            c=ew, cmap="tab10", s=4, alpha=0.6, linewidths=0,
        )
        axes[1].set_title("Event Window")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")
        plt.colorbar(sc2, ax=axes[1], label="Event Window")

        fig.suptitle(
            f"BlueCCGL Embeddings — {n_clusters} clusters, {n_clean:,} cascades",
            fontsize=13,
        )
        plt.tight_layout()
        fig_path = out_dir / "umap_figure.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] UMAP figure saved → {fig_path}")

        # Save UMAP coordinates for reuse in explain stage
        umap_df = pd.DataFrame(
            {"cascade_id": df["cascade_id"].values, "umap_x": z2d[:, 0], "umap_y": z2d[:, 1]}
        )
        umap_df.to_parquet(out_dir / "umap_coords.parquet", index=False)

    except ImportError as e:
        print(f"[evaluate] WARNING: UMAP/matplotlib unavailable ({e}). Skipping figure.")
