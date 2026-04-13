"""Stage: cluster — HDBSCAN over extracted embeddings with grid search.

Reads:
  {output_dir}/extract_embeddings/embeddings.h5
  {output_dir}/extract_embeddings/cascade_index.parquet

Writes:
  {output_dir}/cluster/cluster_assignments.parquet
  {output_dir}/cluster/grid_search_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    import hdbscan as hdbscan_lib
except ImportError:
    raise ImportError("hdbscan not installed: pip install hdbscan")

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    raise ImportError("scikit-learn not installed: pip install scikit-learn")

# Maximum sample used for silhouette during grid search (keeps grid search fast)
_SILHOUETTE_MAX_SAMPLE = 5_000


def run(cfg: dict, output_dir: Path) -> None:
    out_dir = output_dir / "cluster"
    out_dir.mkdir(parents=True, exist_ok=True)

    ccfg = cfg.get("clustering", {})
    grid: list[int] = ccfg.get("min_cluster_size_grid", [15, 25, 40])
    min_samples_ratio: float = ccfg.get("min_samples_ratio", 0.4)  # min_samples = round(ratio * mcs)
    noise_threshold: float = ccfg.get("noise_threshold", 0.30)

    # ── 1. Load embeddings ────────────────────────────────────────────────
    emb_path = output_dir / "extract_embeddings" / "embeddings.h5"
    print(f"[cluster] Loading embeddings from {emb_path}")
    with h5py.File(emb_path, "r") as f:
        embeddings: np.ndarray = f["embeddings"][:]
    print(f"[cluster] Embeddings: {embeddings.shape}")

    idx_path = output_dir / "extract_embeddings" / "cascade_index.parquet"
    meta_df = pd.read_parquet(idx_path)
    assert len(meta_df) == embeddings.shape[0], "Index/embedding count mismatch"

    # ── 2. Grid search ────────────────────────────────────────────────────
    grid_results: list[dict] = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_clusterer: hdbscan_lib.HDBSCAN | None = None
    best_mcs: int = grid[0]

    for mcs in grid:
        min_samples_computed = max(1, round(min_samples_ratio * mcs))
        print(
            f"[cluster]   min_cluster_size={mcs}  min_samples={min_samples_computed} ...",
            end=" ", flush=True,
        )
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_samples_computed,
            core_dist_n_jobs=-1,
            prediction_data=True,
        )
        labels: np.ndarray = clusterer.fit_predict(embeddings)

        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_count = int((labels == -1).sum())
        noise_frac = noise_count / len(labels)

        # Silhouette only on non-noise, and only if clustering is non-degenerate
        if n_clusters >= 2 and noise_frac <= noise_threshold:
            non_noise = labels != -1
            n_sample = min(_SILHOUETTE_MAX_SAMPLE, non_noise.sum())
            score = float(
                silhouette_score(
                    embeddings[non_noise],
                    labels[non_noise],
                    sample_size=n_sample,
                    random_state=42,
                )
            )
        else:
            score = -1.0  # degenerate: skip

        result = {
            "min_cluster_size": mcs,
            "min_samples": min_samples_computed,
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_fraction": float(noise_frac),
            "silhouette": score,
            "selected": False,
        }
        grid_results.append(result)
        print(
            f"n_clusters={n_clusters}  noise={noise_frac:.2%}  silhouette={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_labels = labels
            best_clusterer = clusterer
            best_mcs = mcs

    # Fall back to smallest mcs if all configs were degenerate
    if best_labels is None or best_score < 0:
        print(
            f"[cluster] WARNING: all grid configs produced >{noise_threshold:.0%} noise or <2 clusters. "
            f"Falling back to mcs={grid[0]} without noise filtering."
        )
        best_mcs = grid[0]
        best_clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=best_mcs,
            min_samples=max(1, round(min_samples_ratio * best_mcs)),
        )
        best_labels = best_clusterer.fit_predict(embeddings)
        for r in grid_results:
            if r["min_cluster_size"] == best_mcs:
                r["selected"] = True
    else:
        for r in grid_results:
            if r["min_cluster_size"] == best_mcs:
                r["selected"] = True

    n_final = int(len(set(best_labels)) - (1 if -1 in best_labels else 0))
    noise_final = float((best_labels == -1).sum() / len(best_labels))
    print(
        f"[cluster] Selected mcs={best_mcs}: "
        f"{n_final} clusters, {noise_final:.2%} noise"
    )

    # ── 3. Save cluster persistence (stability) ────────────────────────────
    assert best_clusterer is not None
    persistence_data = {
        "mean_persistence": float(np.mean(best_clusterer.cluster_persistence_)),
        "per_cluster": best_clusterer.cluster_persistence_.tolist(),
        "n_clusters": n_final,
        "min_cluster_size_used": best_mcs,
    }
    persistence_path = out_dir / "cluster_persistences.json"
    persistence_path.write_text(json.dumps(persistence_data, indent=2))
    print(
        f"[cluster] Cluster persistence (stability) — mean={persistence_data['mean_persistence']:.4f} → {persistence_path}"
    )

    # ── 4. Save outputs ───────────────────────────────────────────────────
    grid_path = out_dir / "grid_search_results.json"
    grid_path.write_text(json.dumps(grid_results, indent=2))
    print(f"[cluster] Grid search results → {grid_path}")

    assign_df = meta_df[["cascade_id"]].copy()
    assign_df["cluster_label"] = best_labels
    assign_df["min_cluster_size_used"] = best_mcs

    assign_path = out_dir / "cluster_assignments.parquet"
    assign_df.to_parquet(assign_path, index=False)
    print(f"[cluster] Cluster assignments → {assign_path}")
