"""Stage: ablation — 2³ factorial grid (encoder × augmentation × sentiment).

Reads:
  {output_dir}/build_pyg_dataset/cascade_dataset.pt   (shared across all variants)

Writes:
  {output_dir}/ablation/{variant}/train/checkpoints/best.pt
  {output_dir}/ablation/{variant}/extract_embeddings/embeddings.h5
  {output_dir}/ablation/{variant}/cluster/cluster_assignments.parquet
  {output_dir}/ablation/{variant}/evaluate/metrics.json
  {output_dir}/ablation/ablation_results.json          (checkpoint — written last)
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

from blueccgl.config import apply_overrides

# Variant name → config overrides.  Baseline (rgcn_feed_sent) already done in Outputs_17_04.
VARIANTS: dict[str, dict] = {
    "gcn_feed_sent":    {"model.encoder": "gcn",  "augmentation.strategy": "feed_aware", "model.use_sentiment": True},
    "rgcn_rand_sent":   {"model.encoder": "rgcn", "augmentation.strategy": "random",     "model.use_sentiment": True},
    "rgcn_feed_nosent": {"model.encoder": "rgcn", "augmentation.strategy": "feed_aware", "model.use_sentiment": False},
    "gcn_rand_sent":    {"model.encoder": "gcn",  "augmentation.strategy": "random",     "model.use_sentiment": True},
    "gcn_feed_nosent":  {"model.encoder": "gcn",  "augmentation.strategy": "feed_aware", "model.use_sentiment": False},
    "rgcn_rand_nosent": {"model.encoder": "rgcn", "augmentation.strategy": "random",     "model.use_sentiment": False},
    "gcn_rand_nosent":  {"model.encoder": "gcn",  "augmentation.strategy": "random",     "model.use_sentiment": False},
}


def run(cfg: dict, output_dir: Path, device: str, seed: int) -> None:
    main_pyg = output_dir / "build_pyg_dataset" / "cascade_dataset.pt"
    if not main_pyg.exists():
        raise FileNotFoundError(
            f"Dataset not found at {main_pyg}. Run --stage build_pyg_dataset first."
        )

    ablation_root = output_dir / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t_total = time.time()

    for variant_name, overrides in VARIANTS.items():
        variant_dir = ablation_root / variant_name
        variant_dir.mkdir(exist_ok=True)

        eval_path = variant_dir / "evaluate" / "metrics.json"
        if eval_path.exists():
            print(f"[ablation] {variant_name}: checkpoint found, skipping.")
            results.append(_read_result(variant_name, variant_dir, overrides))
            continue

        print(f"\n[ablation] ── Variant: {variant_name} ──")
        _link_dataset(main_pyg, variant_dir)

        variant_cfg = apply_overrides(copy.deepcopy(cfg), overrides)

        from blueccgl.seed import seed_everything
        seed_everything(seed)

        from blueccgl.stages import train as train_stage
        from blueccgl.stages import extract_embeddings as embed_stage
        from blueccgl.stages import cluster as cluster_stage
        from blueccgl.stages import evaluate as eval_stage

        train_stage.run(variant_cfg, variant_dir, device, seed, ablation=None)
        embed_stage.run(variant_cfg, variant_dir, device)
        cluster_stage.run(variant_cfg, variant_dir)
        eval_stage.run(variant_cfg, variant_dir)

        results.append(_read_result(variant_name, variant_dir, overrides))
        print(f"[ablation] {variant_name}: done.")

    elapsed = (time.time() - t_total) / 60
    print(f"\n[ablation] All variants complete in {elapsed:.1f} min.")

    # Write checkpoint (stage completion marker)
    checkpoint = ablation_root / "ablation_results.json"
    checkpoint.write_text(json.dumps(results, indent=2))
    print(f"[ablation] Results → {checkpoint}")

    _print_table(results)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _link_dataset(main_pyg: Path, variant_dir: Path) -> None:
    """Symlink the shared cascade_dataset.pt into the variant output dir."""
    variant_pyg_dir = variant_dir / "build_pyg_dataset"
    variant_pyg_dir.mkdir(exist_ok=True)

    target = variant_pyg_dir / "cascade_dataset.pt"
    if target.exists() or target.is_symlink():
        return

    try:
        os.symlink(main_pyg.resolve(), target)
    except OSError:
        # Fallback on filesystems that don't support symlinks
        import shutil
        shutil.copy2(main_pyg, target)

    # Also link dataset_stats.json if available
    stats_src = main_pyg.parent / "dataset_stats.json"
    stats_dst = variant_pyg_dir / "dataset_stats.json"
    if stats_src.exists() and not stats_dst.exists():
        try:
            os.symlink(stats_src.resolve(), stats_dst)
        except OSError:
            pass


def _read_result(variant_name: str, variant_dir: Path, overrides: dict) -> dict:
    """Read clustering metrics + training metadata for a completed variant."""
    result: dict = {"variant": variant_name}

    # Encoder / aug / sentiment labels
    result["encoder"] = "GCN" if overrides.get("model.encoder") == "gcn" else "R-GCN"
    result["aug"] = "random" if overrides.get("augmentation.strategy") == "random" else "feed-aware"
    result["sentiment"] = "yes" if overrides.get("model.use_sentiment", True) else "no"

    # Clustering metrics
    metrics_path = variant_dir / "evaluate" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        result["sil_mean"]  = m.get("silhouette", {}).get("mean")
        result["sil_ci_lo"] = m.get("silhouette", {}).get("ci_lo")
        result["sil_ci_hi"] = m.get("silhouette", {}).get("ci_hi")
        result["db_mean"]   = m.get("davies_bouldin", {}).get("mean")
        result["db_ci_lo"]  = m.get("davies_bouldin", {}).get("ci_lo")
        result["db_ci_hi"]  = m.get("davies_bouldin", {}).get("ci_hi")
        result["ch_mean"]   = m.get("calinski_harabasz", {}).get("mean")
        result["ch_ci_lo"]  = m.get("calinski_harabasz", {}).get("ci_lo")
        result["ch_ci_hi"]  = m.get("calinski_harabasz", {}).get("ci_hi")
        result["dbcv_mean"] = m.get("dbcv", {}).get("mean")
        result["n_clusters"] = m.get("meta", {}).get("n_clusters")
        result["noise_pct"]  = m.get("meta", {}).get("noise_pct")

    # Training metadata
    log_path = variant_dir / "train" / "training_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        losses = [e["loss"] for e in log]
        best_idx = int(min(range(len(losses)), key=lambda i: losses[i]))
        result["best_loss"]  = round(losses[best_idx], 5)
        result["best_epoch"] = log[best_idx]["epoch"]
        result["final_loss"] = round(log[-1]["loss"], 5)
        result["total_epochs"] = len(log)
        fallbacks = [e.get("aug_fallback_rate", float("nan")) for e in log]
        result["aug_fallback_rate"] = round(
            sum(f for f in fallbacks if f == f) / max(sum(f == f for f in fallbacks), 1), 4
        )

    result["status"] = "done" if metrics_path.exists() else "missing"
    return result


def _print_table(results: list[dict]) -> None:
    header = f"{'Variant':<22} {'Enc':<6} {'Aug':<11} {'Sent':<5} {'Sil':>6} {'DB':>6} {'CH':>8} {'DBCV':>7} {'K':>3} {'Noise%':>7} {'FallBk':>7}"
    print(f"\n[ablation] ── Results ──────────────────────────────────────────────")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['variant']:<22} {r.get('encoder','?'):<6} {r.get('aug','?'):<11} "
            f"{r.get('sentiment','?'):<5} "
            f"{r.get('sil_mean') or '—':>6} {r.get('db_mean') or '—':>6} "
            f"{r.get('ch_mean') or '—':>8} {r.get('dbcv_mean') or '—':>7} "
            f"{r.get('n_clusters') or '—':>3} {r.get('noise_pct') or '—':>7} "
            f"{r.get('aug_fallback_rate') or '—':>7}"
        )
