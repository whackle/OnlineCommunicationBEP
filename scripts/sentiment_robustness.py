"""
Robustness check: how much of the BlueCCGL cluster structure is driven by sentiment?

Two analyses on existing v8 outputs — no retraining required:

  1. Sentiment-only baseline
     HDBSCAN on the raw 3D sentiment_composition vectors.
     If silhouette ≈ v8 (0.413), sentiment alone explains the clustering.

  2. Sentiment-projected-out recluster
     Fit a linear subspace capturing sentiment from the 64D embeddings,
     subtract it, recluster. Measures the strictly structural component.

Usage (from repo root):
    python scripts/sentiment_robustness.py --output-dir Outputs_17_04

Writes:
    Outputs_17_04/robustness/sentiment_robustness.json
    Outputs_17_04/robustness/sentiment_robustness.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Outputs_17_04")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rob_dir = out_dir / "robustness"
    rob_dir.mkdir(exist_ok=True)

    steps = tqdm(
        ["load", "analysis1", "project", "umap", "cluster2", "baseline", "save"],
        desc="robustness", unit="step", ncols=80,
    )

    # ── Load data ──────────────────────────────────────────────────────────
    steps.set_description("loading data")
    t0 = time.time()
    with h5py.File(out_dir / "extract_embeddings" / "embeddings.h5", "r") as f:
        embeddings = f["embeddings"][:]                  # [N, 64]

    cascade_index = pd.read_parquet(
        out_dir / "extract_embeddings" / "cascade_index.parquet"
    )
    cluster_assignments = pd.read_parquet(
        out_dir / "cluster" / "cluster_assignments.parquet"
    )
    df = cascade_index.merge(cluster_assignments[["cascade_id", "cluster_label"]], on="cascade_id")
    assert len(df) == len(embeddings), "Row count mismatch between embeddings and index"

    labels = df["cluster_label"].values
    non_noise = labels != -1

    with h5py.File(out_dir / "cluster" / "umap_embeddings_cluster.h5", "r") as f:
        umap10 = f["embeddings"][:]                      # [N, 10]

    sent_cols = ["sentiment_neg", "sentiment_neu", "sentiment_pos"]
    if all(c in df.columns for c in sent_cols):
        sent_mat = df[sent_cols].values.astype(np.float32)
    elif "sentiment_composition" in df.columns:
        sent_mat = np.stack(df["sentiment_composition"].values).astype(np.float32)
    else:
        raise RuntimeError("sentiment columns not found in cascade_index")

    tqdm.write(f"  loaded {len(embeddings):,} cascades ({non_noise.sum():,} non-noise)  [{time.time()-t0:.1f}s]")
    steps.update(1)

    results = {}

    # ── Analysis 1: Sentiment-only HDBSCAN ────────────────────────────────
    steps.set_description("analysis 1: sentiment-only HDBSCAN")
    t0 = time.time()
    sent_labels, sent_metrics = _cluster_and_score(
        sent_mat, min_cluster_size=300, seed=args.seed, label="sentiment-only"
    )
    results["sentiment_only"] = sent_metrics
    tqdm.write(f"  sentiment-only → k={sent_metrics.get('n_clusters')}  "
               f"noise={sent_metrics.get('noise_pct')}%  "
               f"sil={sent_metrics.get('silhouette')}  [{time.time()-t0:.1f}s]")
    steps.update(1)

    # ── Analysis 2a: Project out sentiment ────────────────────────────────
    steps.set_description("analysis 2: projecting out sentiment")
    t0 = time.time()
    emb_clean, sent_var_explained = _project_out_sentiment(embeddings, sent_mat)
    tqdm.write(f"  sentiment R² in 64D embeddings: {sent_var_explained:.3f}  [{time.time()-t0:.1f}s]")
    steps.update(1)

    # ── Analysis 2b: UMAP on cleaned embeddings ───────────────────────────
    # verbose=True activates UMAP's own internal tqdm epoch bar
    steps.set_description("analysis 2: UMAP 64D→10D  (slow — see bar below)")
    tqdm.write("  UMAP 64D→10D on sentiment-free embeddings (this takes ~10-15 min on CPU)...")
    import umap as umap_lib
    t0 = time.time()
    reducer = umap_lib.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.1,
        random_state=args.seed, verbose=True,
    )
    umap_clean = reducer.fit_transform(emb_clean).astype(np.float32)
    tqdm.write(f"  UMAP done  [{time.time()-t0:.1f}s]")
    steps.update(1)

    # ── Analysis 2c: HDBSCAN + score on cleaned UMAP ─────────────────────
    steps.set_description("analysis 2: HDBSCAN + score")
    t0 = time.time()
    clean_labels, clean_metrics = _cluster_and_score(
        umap_clean, min_cluster_size=300, seed=args.seed, label="sentiment-projected-out"
    )
    results["sentiment_projected_out"] = clean_metrics
    results["sentiment_subspace_r2"] = round(float(sent_var_explained), 4)
    tqdm.write(f"  projected-out → k={clean_metrics.get('n_clusters')}  "
               f"noise={clean_metrics.get('noise_pct')}%  "
               f"sil={clean_metrics.get('silhouette')}  [{time.time()-t0:.1f}s]")
    steps.update(1)

    # ── Analysis 3: v8 baseline metrics for comparison ────────────────────
    steps.set_description("loading v8 baseline metrics")
    metrics_path = out_dir / "evaluate" / "metrics.json"
    if metrics_path.exists():
        v8 = json.loads(metrics_path.read_text())
        results["v8_baseline"] = {
            "silhouette": v8.get("silhouette", {}).get("mean"),
            "davies_bouldin": v8.get("davies_bouldin", {}).get("mean"),
            "dbcv": v8.get("dbcv", {}).get("mean"),
            "n_clusters": v8.get("meta", {}).get("n_clusters"),
            "noise_pct": v8.get("meta", {}).get("noise_pct"),
        }
        r = results["v8_baseline"]
        tqdm.write(f"  v8 baseline: k={r.get('n_clusters','?')}  noise={r.get('noise_pct','?')}%  "
                   f"sil={r.get('silhouette','?')}  DB={r.get('davies_bouldin','?')}  "
                   f"DBCV={r.get('dbcv','?')}")
    else:
        tqdm.write("  v8 baseline metrics.json not found — skipping comparison")
    steps.update(1)

    # ── Summary table ──────────────────────────────────────────────────────
    steps.set_description("saving results")
    tqdm.write("\n[robustness] ── Summary ──────────────────────────────────────────")
    tqdm.write(f"{'Condition':<35} {'Sil':>6} {'DB':>6} {'DBCV':>7} {'K':>4} {'Noise%':>7}")
    tqdm.write("-" * 65)
    for key, label in [
        ("v8_baseline",              "v8 full model (10D UMAP)   "),
        ("sentiment_only",           "Sentiment-only (3D)        "),
        ("sentiment_projected_out",  "Sentiment-projected-out    "),
    ]:
        r = results.get(key, {})
        sil  = f"{r.get('silhouette'):.3f}"  if r.get("silhouette")  is not None else "—"
        db   = f"{r.get('davies_bouldin'):.3f}" if r.get("davies_bouldin") is not None else "—"
        dbcv = f"{r.get('dbcv'):.3f}"        if r.get("dbcv")        is not None else "—"
        k    = str(r.get("n_clusters", "—"))
        noise = f"{r.get('noise_pct'):.1f}"  if r.get("noise_pct")   is not None else "—"
        tqdm.write(f"{label:<35} {sil:>6} {db:>6} {dbcv:>7} {k:>4} {noise:>7}")

    sent_sil = results.get("sentiment_only", {}).get("silhouette")
    v8_sil   = results.get("v8_baseline",    {}).get("silhouette")
    if sent_sil is not None and v8_sil is not None:
        delta = v8_sil - sent_sil
        pct   = 100 * delta / max(abs(sent_sil), 1e-6)
        tqdm.write(f"\n  Model vs sentiment-only Δsil = {delta:+.3f} ({pct:+.1f}%)")
        if abs(delta) < 0.05:
            tqdm.write("  WARNING: model barely exceeds sentiment-only baseline — "
                       "cluster structure may be sentiment-driven.")
        elif delta > 0.1:
            tqdm.write("  Model substantially outperforms sentiment baseline — "
                       "structural signal is present beyond sentiment.")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = rob_dir / "sentiment_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))
    tqdm.write(f"\n[robustness] Results → {out_path}")
    _plot(results, rob_dir / "sentiment_robustness.png")
    steps.update(1)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cluster_and_score(X: np.ndarray, min_cluster_size: int, seed: int, label: str) -> tuple:
    import hdbscan as hdbscan_lib
    from sklearn.metrics import (
        davies_bouldin_score,
        calinski_harabasz_score,
        silhouette_score,
    )

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, round(0.4 * min_cluster_size)),
        core_dist_n_jobs=1,
        prediction_data=False,
    )
    lbls = clusterer.fit_predict(X)
    n_clusters = int((lbls >= 0).sum() > 0 and len(set(lbls[lbls >= 0])))
    n_clusters = len(set(lbls[lbls >= 0]))
    noise_pct  = round(100 * (lbls == -1).mean(), 2)
    non_noise  = lbls != -1

    metrics: dict = {"n_clusters": n_clusters, "noise_pct": noise_pct}

    if n_clusters >= 2 and non_noise.sum() >= 10:
        X_nn  = X[non_noise]
        l_nn  = lbls[non_noise]
        cap   = 5000
        if len(X_nn) > cap:
            idx = np.random.default_rng(seed).choice(len(X_nn), cap, replace=False)
            X_s, l_s = X_nn[idx], l_nn[idx]
        else:
            X_s, l_s = X_nn, l_nn

        metrics["silhouette"]      = round(float(silhouette_score(X_s, l_s)), 4)
        metrics["davies_bouldin"]  = round(float(davies_bouldin_score(X_nn, l_nn)), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_nn, l_nn)), 2)

        try:
            from hdbscan import validity_index
            metrics["dbcv"] = round(float(validity_index(X_nn.astype(np.float64), l_nn)), 4)
        except Exception:
            metrics["dbcv"] = None
    else:
        print(f"  [{label}] Too few clusters ({n_clusters}) or non-noise points to score.")

    return lbls, metrics


def _project_out_sentiment(embeddings: np.ndarray, sent_mat: np.ndarray) -> tuple:
    """
    Regress sentiment_composition onto embeddings, subtract the fitted component.
    Returns (cleaned_embeddings, R²_of_sentiment_in_embedding_space).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    # Fit: predict embeddings from sentiment — this finds the sentiment directions
    reg = Ridge(alpha=1.0)
    reg.fit(sent_mat, emb_scaled)
    sent_component = reg.predict(sent_mat)   # [N, 64] — the sentiment-explainable part

    # R²: how much variance in embeddings does sentiment explain?
    ss_res = np.sum((emb_scaled - sent_component) ** 2)
    ss_tot = np.sum((emb_scaled - emb_scaled.mean(axis=0)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    # Subtract sentiment component
    emb_clean_scaled = emb_scaled - sent_component
    emb_clean = scaler.inverse_transform(emb_clean_scaled)

    return emb_clean.astype(np.float32), r2


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}: k={m.get('n_clusters','?')}  noise={m.get('noise_pct','?')}%  "
          f"sil={m.get('silhouette','?')}  DB={m.get('davies_bouldin','?')}  "
          f"DBCV={m.get('dbcv','?')}")


def _plot(results: dict, path: Path) -> None:
    try:
        import blueccgl.utils.mpl_compat  # noqa: F401  — Python 3.14 patch
        import matplotlib.pyplot as plt

        conditions = []
        sils       = []
        colors     = []
        for key, label, color in [
            ("v8_baseline",             "v8 full model",             "#2196F3"),
            ("sentiment_only",          "Sentiment-only (3D)",       "#FF5722"),
            ("sentiment_projected_out", "Sentiment-projected-out",   "#4CAF50"),
        ]:
            r = results.get(key, {})
            if r.get("silhouette") is not None:
                conditions.append(label)
                sils.append(r["silhouette"])
                colors.append(color)

        if not conditions:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(conditions, sils, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, sils):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.axhline(sils[0], color="#2196F3", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Silhouette Score (10D UMAP space)", fontsize=11)
        ax.set_title("Cluster quality: full model vs sentiment-only baseline", fontsize=12)
        ax.set_ylim(0, max(sils) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[robustness] Plot → {path}")
        plt.close(fig)
    except Exception as e:
        print(f"[robustness] Plot skipped: {e}")


if __name__ == "__main__":
    main()
