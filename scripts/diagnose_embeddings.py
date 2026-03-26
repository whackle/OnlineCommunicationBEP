#!/usr/bin/env python
"""
diagnose_embeddings.py -- Post-training embedding health check for BlueCCGL.

Loads best.pt, forward-passes all cascades, and computes:
  1. Collapse diagnostics (cosine sim, norms, per-dim variance, SVD spectrum)
  2. Discriminative power (uniformity, alignment, intra/inter-event similarity)
  3. Pool-level health (per-level effective dimensionality)

Outputs:
  outputs/diagnostics/embedding_diagnostics.json
  outputs/diagnostics/*.png  (6 plots)
  Printed summary table + HEALTHY / MARGINAL / COLLAPSED verdict

Usage (from repo root):
    .venv/bin/python scripts/diagnose_embeddings.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Plotting disabled: matplotlib 3.10 + Python 3.14 has a deepcopy recursion bug
# that makes savefig crash. All numerical diagnostics work fine without plots.
# Re-enable when matplotlib ships a fix (tracked upstream).
SKIP_PLOTS = True

if not SKIP_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── Add repo root to path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blueccgl.config import load_config
from blueccgl.models.blueccgl import BlueCCGL
from blueccgl.seed import seed_everything

# ── Constants ─────────────────────────────────────────────────────────────────
COSINE_SAMPLE = 5000
UNIFORMITY_PAIRS = 10_000
ALIGNMENT_SAMPLE = 1000
POOL_SAMPLE = 2000
INFERENCE_BATCH = 128  # conservative for DiffPool dense ops on CPU

EVENT_NAMES = {0: "baseline", 1: "E1", 2: "E2", 3: "E3", 4: "E4", 5: "E5", 6: "E6"}


def main():
    t0 = time.time()
    seed_everything(42)

    cfg = load_config(ROOT / "config.yaml")
    out_dir = ROOT / "outputs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model + dataset ───────────────────────────────────────────
    print("[diag] Loading dataset...")
    data_list = torch.load(
        ROOT / "outputs" / "build_pyg_dataset" / "cascade_dataset.pt",
        weights_only=False,
    )
    N = len(data_list)
    print(f"[diag] {N:,} cascades loaded")

    print("[diag] Loading model from best.pt...")
    model = BlueCCGL(cfg)
    ckpt = torch.load(
        ROOT / "outputs" / "train" / "checkpoints" / "best.pt",
        weights_only=False,
        map_location="cpu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[diag] Model loaded (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.5f})")

    # ── 1.5. Training log analysis ────────────────────────────────────────
    train_log = _load_training_log(ROOT / "outputs" / "train" / "training_log.json")

    # ── 2. Extract embeddings + pool-level representations ────────────────
    print("[diag] Extracting embeddings...")
    embeddings, event_windows, pools = _extract_all(model, data_list)
    emb_dim = embeddings.shape[1]
    print(f"[diag] Embeddings: {embeddings.shape}  pools: {[p.shape for p in pools.values()]}")

    # ── 3. Collapse diagnostics ───────────────────────────────────────────
    print("[diag] Computing collapse diagnostics...")
    collapse = _collapse_diagnostics(embeddings, out_dir)

    # ── 4. Discriminative power ───────────────────────────────────────────
    print("[diag] Computing discriminative power metrics...")
    discriminative = _discriminative_metrics(embeddings, event_windows, model, data_list, cfg, out_dir)

    # ── 5. Pool-level diagnostics ─────────────────────────────────────────
    print("[diag] Computing pool-level diagnostics...")
    pool_diag = _pool_diagnostics(pools, out_dir)

    # ── 6. Verdict ────────────────────────────────────────────────────────
    verdict, reasoning, recommendations = _verdict(collapse, discriminative, pool_diag)

    # ── 7. Report ─────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": "outputs/train/checkpoints/best.pt",
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_loss": ckpt.get("loss"),
        "n_cascades": N,
        "embedding_dim": emb_dim,
        "training_log": train_log,
        "collapse": collapse,
        "discriminative": discriminative,
        "pool_levels": pool_diag,
        "verdict": verdict,
        "reasoning": reasoning,
        "recommendations": recommendations,
    }
    (out_dir / "embedding_diagnostics.json").write_text(json.dumps(report, indent=2, default=float))

    _print_summary(collapse, discriminative, pool_diag, verdict, reasoning, recommendations, emb_dim, train_log)

    elapsed = time.time() - t0
    print(f"\n[diag] Done in {elapsed:.1f}s. Outputs in {out_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═════════════════════════════════════════════════════════════════════════════

def _extract_all(
    model: BlueCCGL,
    data_list: list,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Single-pass extraction of embeddings + pool-level representations."""
    loader = DataLoader(data_list, batch_size=INFERENCE_BATCH, shuffle=False)

    all_emb, all_events = [], []
    all_pool0, all_pool1, all_pool2 = [], [], []
    all_root = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="extracting", unit="batch"):
            # -- Embeddings via encode() --
            emb = model.encode(batch)
            all_emb.append(emb.cpu())
            all_events.append(batch.event_window.cpu())

            # -- Encoder node embeddings (needed for pool-level and root capture) --
            x = batch.x
            if not model.use_sentiment:
                x = x.clone()
                x[:, 1] = 0.0
                x[:, 2] = 0.0
            h = model.encoder(x, batch.edge_index, batch.edge_type)

            # -- Pool-level capture (DiffPool mode) --
            if model.use_diffpool:
                p0, p1, p2, _ = model.pooler(h, batch.edge_index, batch.batch)
                all_pool0.append(p0.cpu())
                all_pool1.append(p1.cpu())
                all_pool2.append(p2.cpu())

            # -- Root node capture (node 0 of each graph in batch) --
            if model.use_root_readout:
                root_h = h[batch.ptr[:-1]]    # [B, hidden_dim]
                all_root.append(root_h.cpu())

    embeddings = torch.cat(all_emb, dim=0)
    event_windows = torch.cat(all_events, dim=0)
    pools = {}
    if all_pool0:
        pools["pool_0"] = torch.cat(all_pool0, dim=0)
        pools["pool_1"] = torch.cat(all_pool1, dim=0)
        pools["pool_2"] = torch.cat(all_pool2, dim=0)
    if all_root:
        pools["root_node"] = torch.cat(all_root, dim=0)

    return embeddings, event_windows, pools


# ═════════════════════════════════════════════════════════════════════════════
# 1.5. Training log analysis
# ═════════════════════════════════════════════════════════════════════════════

def _load_training_log(log_path: Path) -> dict:
    """Load and summarise training_log.json. Returns empty dict if not found."""
    if not log_path.exists():
        return {"available": False}

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        return {"available": False}

    first = log[0]
    last = log[-1]
    best = min(log, key=lambda e: e["loss"])
    has_covariance = "covariance" in first

    summary: dict = {
        "available": True,
        "n_epochs": len(log),
        "initial_loss": first["loss"],
        "final_loss": last["loss"],
        "best_loss": best["loss"],
        "best_epoch": best["epoch"],
        "final_contrastive": last.get("contrastive"),
        "final_aux": last.get("aux"),
        "final_variance": last.get("variance"),
        "final_grad_norm": last.get("grad_norm_mean"),
        "has_covariance_term": has_covariance,
    }

    if has_covariance:
        cov_vals = [e["covariance"] for e in log if "covariance" in e]
        summary["initial_covariance"] = cov_vals[0] if cov_vals else None
        summary["final_covariance"] = cov_vals[-1] if cov_vals else None
        # Check convergence: is final cov < 10% of initial?
        if cov_vals and cov_vals[0] > 0:
            summary["covariance_reduction_pct"] = round(
                100 * (1 - cov_vals[-1] / cov_vals[0]), 1
            )
        else:
            summary["covariance_reduction_pct"] = None

    # Gradient norm trend (is clipping still happening constantly?)
    grad_norms = [e.get("grad_norm_mean", 0) for e in log]
    grad_clip = 2.0  # v4 default; approximate
    epochs_clipping = sum(1 for g in grad_norms if g > grad_clip)
    summary["grad_clip_fraction"] = round(epochs_clipping / len(grad_norms), 2)
    summary["final_grad_norm"] = grad_norms[-1]

    # Augmentation fallback rate (v6+)
    has_fallback = "aug_fallback_rate" in first
    summary["has_fallback_rate"] = has_fallback
    if has_fallback:
        summary["final_fallback_rate"] = last.get("aug_fallback_rate", 0)

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 3. Collapse diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def _collapse_diagnostics(embeddings: torch.Tensor, out_dir: Path) -> dict:
    N, D = embeddings.shape

    # --- 3a. Cosine similarity histogram (sampled) ---
    n_sample = min(COSINE_SAMPLE, N)
    idx = torch.randperm(N)[:n_sample]
    Z_sample = F.normalize(embeddings[idx], dim=1)
    cos_sim = (Z_sample @ Z_sample.T).numpy()
    mask = ~np.eye(n_sample, dtype=bool)
    off_diag = cos_sim[mask]

    median_cos = float(np.median(off_diag))
    mean_cos = float(np.mean(off_diag))
    std_cos = float(np.std(off_diag))
    p95_cos = float(np.percentile(off_diag, 95))
    p99_cos = float(np.percentile(off_diag, 99))

    if not SKIP_PLOTS:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(off_diag, bins=100, density=True, alpha=0.7, color="steelblue")
        ax.axvline(median_cos, color="red", ls="--")
        ax.axvline(0.5, color="orange", ls=":", alpha=0.7)
        ax.axvline(0.8, color="darkred", ls=":", alpha=0.7)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"Pairwise Cosine Similarity (n={n_sample}, median={median_cos:.3f})")
        fig.savefig(out_dir / "cosine_sim_histogram.png", dpi=150)
        plt.close(fig)

    # --- 3b. Embedding norm distribution ---
    norms = torch.norm(embeddings, dim=1).numpy()
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))
    norm_cv = norm_std / norm_mean if norm_mean > 0 else 0.0

    if not SKIP_PLOTS:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(norms, bins=80, density=True, alpha=0.7, color="darkorange")
        ax.axvline(norm_mean, color="red", ls="--")
        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Density")
        ax.set_title(f"Embedding Norm Distribution (mean={norm_mean:.3f}, std={norm_std:.3f})")
        fig.savefig(out_dir / "norm_distribution.png", dpi=150)
        plt.close(fig)

    # --- 3c. Per-dimension variance ---
    dim_var = embeddings.var(dim=0).numpy()
    collapsed_dims = int((dim_var < 0.01).sum())
    participation_ratio = float(dim_var.sum() ** 2 / (dim_var ** 2).sum()) if (dim_var ** 2).sum() > 0 else 0.0

    if not SKIP_PLOTS:
        sorted_var = np.sort(dim_var)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(D), sorted_var, color="teal", alpha=0.7)
        ax.axhline(0.01, color="red", ls="--", alpha=0.7)
        ax.set_xlabel("Dimension (sorted)")
        ax.set_ylabel("Variance")
        ax.set_title(f"Per-Dimension Variance (collapsed={collapsed_dims}/{D}, red=0.01 threshold)")
        fig.savefig(out_dir / "per_dim_variance.png", dpi=150)
        plt.close(fig)

    # --- 3d. SVD spectrum + effective rank ---
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    S_np = S.numpy()
    # Effective rank via Shannon entropy
    p = S_np / S_np.sum()
    p = p[p > 1e-12]  # avoid log(0)
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    cumvar = np.cumsum(S_np ** 2) / (S_np ** 2).sum()
    top1_var = float(cumvar[0])
    top5_var = float(cumvar[min(4, len(cumvar) - 1)])
    top10_var = float(cumvar[min(9, len(cumvar) - 1)])

    if not SKIP_PLOTS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(range(len(S_np)), S_np / S_np[0], color="navy", alpha=0.7)
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Normalized Singular Value")
        ax1.set_title(f"SVD Spectrum (eff. rank={eff_rank:.1f})")
        ax1.set_yscale("log")
        ax2.bar(range(len(cumvar)), cumvar, color="darkgreen", alpha=0.7)
        ax2.axhline(0.9, color="red", ls="--", alpha=0.5)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Cumulative Variance Explained")
        ax2.set_title("Cumulative Variance (red=90%)")
        fig.savefig(out_dir / "svd_spectrum.png", dpi=150)
        plt.close(fig)

    return {
        "median_cosine_sim": median_cos,
        "mean_cosine_sim": mean_cos,
        "cosine_sim_std": std_cos,
        "cosine_sim_p95": p95_cos,
        "cosine_sim_p99": p99_cos,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_cv,
        "collapsed_dims": collapsed_dims,
        "participation_ratio": participation_ratio,
        "effective_rank": eff_rank,
        "top1_variance_ratio": top1_var,
        "top5_variance_ratio": top5_var,
        "top10_variance_ratio": top10_var,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. Discriminative power
# ═════════════════════════════════════════════════════════════════════════════

def _discriminative_metrics(
    embeddings: torch.Tensor,
    event_windows: torch.Tensor,
    model: BlueCCGL,
    data_list: list,
    cfg: dict,
    out_dir: Path,
) -> dict:
    Z = F.normalize(embeddings, dim=1)
    N = Z.shape[0]

    # --- 4a. Intra vs inter-event similarity ---
    intra_sims = {}
    event_ids = event_windows.numpy()
    unique_events = sorted(set(event_ids.tolist()))

    for ev in unique_events:
        ev_mask = event_ids == ev
        ev_idx = np.where(ev_mask)[0]
        if len(ev_idx) < 10:
            continue
        sample_n = min(500, len(ev_idx))
        sampled = np.random.choice(ev_idx, size=sample_n, replace=False)
        Z_ev = Z[sampled]
        cos_ev = (Z_ev @ Z_ev.T).numpy()
        mask = ~np.eye(sample_n, dtype=bool)
        intra_sims[EVENT_NAMES.get(ev, str(ev))] = {
            "mean": float(np.mean(cos_ev[mask])),
            "n_cascades": int(ev_mask.sum()),
        }

    # Inter-event: random pairs from different events
    inter_pairs = 5000
    idx1 = np.random.randint(0, N, size=inter_pairs * 2)
    idx2 = np.random.randint(0, N, size=inter_pairs * 2)
    diff_event = event_ids[idx1] != event_ids[idx2]
    idx1 = idx1[diff_event][:inter_pairs]
    idx2 = idx2[diff_event][:inter_pairs]
    inter_cos = (Z[idx1] * Z[idx2]).sum(dim=1).numpy()
    inter_event_sim = float(np.mean(inter_cos))

    if not SKIP_PLOTS:
        fig, ax = plt.subplots(figsize=(10, 5))
        events = sorted(intra_sims.keys())
        intra_vals = [intra_sims[e]["mean"] for e in events]
        x_pos = range(len(events))
        ax.bar(x_pos, intra_vals, color="steelblue", alpha=0.7)
        ax.axhline(inter_event_sim, color="red", ls="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(events, rotation=45)
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title(f"Intra-Event vs Inter-Event Similarity (red=inter={inter_event_sim:.3f})")
        fig.savefig(out_dir / "event_similarity.png", dpi=150)
        plt.close(fig)

    # --- 4b. Uniformity (Wang & Isola 2020) ---
    n_pairs = min(UNIFORMITY_PAIRS, N)
    p1 = torch.randperm(N)[:n_pairs]
    p2 = torch.randperm(N)[:n_pairs]
    sq_dist = ((Z[p1] - Z[p2]) ** 2).sum(dim=1)
    uniformity = float(torch.log(torch.exp(-2 * sq_dist).mean()).item())

    # --- 4c. Alignment (augmented pair similarity) ---
    alignment = _compute_alignment(model, data_list, cfg)

    return {
        "intra_event_sim": intra_sims,
        "inter_event_sim": inter_event_sim,
        "uniformity": uniformity,
        "alignment": alignment,
    }


def _compute_alignment(model: BlueCCGL, data_list: list, cfg: dict) -> float:
    """Alignment: mean cosine sim between two augmented views of same cascade."""
    from blueccgl.stages.train import _build_augmentor

    augment_pair = _build_augmentor(cfg["augmentation"])

    sample_n = min(ALIGNMENT_SAMPLE, len(data_list))
    indices = random.sample(range(len(data_list)), sample_n)

    view1_list, view2_list = [], []
    for i in indices:
        try:
            v1, v2 = augment_pair(data_list[i])
            if v1 is not None and v2 is not None and v1.num_nodes >= 3 and v2.num_nodes >= 3:
                view1_list.append(v1)
                view2_list.append(v2)
        except Exception:
            continue

    if len(view1_list) < 10:
        return float("nan")

    from torch_geometric.data import Batch

    sims = []
    bs = 128
    with torch.no_grad():
        for start in range(0, len(view1_list), bs):
            end = min(start + bs, len(view1_list))
            b1 = Batch.from_data_list(view1_list[start:end])
            b2 = Batch.from_data_list(view2_list[start:end])
            z1 = F.normalize(model.encode(b1), dim=1)
            z2 = F.normalize(model.encode(b2), dim=1)
            sims.append((z1 * z2).sum(dim=1))

    all_sims = torch.cat(sims).numpy()
    return float(np.mean(all_sims))


# ═════════════════════════════════════════════════════════════════════════════
# 5. Pool-level diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def _pool_diagnostics(pools: dict[str, torch.Tensor], out_dir: Path) -> dict:
    if not pools:
        return {"note": "DiffPool disabled and root readout not enabled — no pool-level diagnostics"}

    results = {}

    for idx, (name, P) in enumerate(sorted(pools.items())):
        N, D = P.shape

        # Per-dim variance
        dim_var = P.var(dim=0).numpy()
        collapsed = int((dim_var < 0.01).sum())
        pr = float(dim_var.sum() ** 2 / (dim_var ** 2).sum()) if (dim_var ** 2).sum() > 0 else 0.0

        # Cosine sim (sampled)
        n_s = min(POOL_SAMPLE, N)
        si = torch.randperm(N)[:n_s]
        Z_s = F.normalize(P[si], dim=1)
        cos_s = (Z_s @ Z_s.T).numpy()
        mask = ~np.eye(n_s, dtype=bool)
        med_cos = float(np.median(cos_s[mask]))

        results[name] = {
            "effective_dim": pr,
            "collapsed_dims": collapsed,
            "median_cosine_sim": med_cos,
        }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 6. Verdict
# ═════════════════════════════════════════════════════════════════════════════

def _verdict(
    collapse: dict,
    discriminative: dict,
    pool_diag: dict,
) -> tuple[str, str, list[str]]:
    median_cos = collapse["median_cosine_sim"]
    eff_rank = collapse["effective_rank"]
    uniformity = discriminative["uniformity"]
    alignment = discriminative["alignment"]

    reasons = []
    recommendations = []

    # -- Collapse checks --
    if median_cos > 0.8:
        reasons.append(f"FAIL: median cosine sim = {median_cos:.3f} (>0.8 = severe collapse)")
        recommendations.append("Lower temperature from 0.5 to 0.1-0.3 to sharpen contrastive objective")
    elif median_cos > 0.5:
        reasons.append(f"WARN: median cosine sim = {median_cos:.3f} (0.5-0.8 = partial collapse)")
        recommendations.append("Consider lowering temperature to 0.3")
    else:
        reasons.append(f"OK: median cosine sim = {median_cos:.3f} (<0.5)")

    if eff_rank < 10:
        reasons.append(f"FAIL: effective rank = {eff_rank:.1f} (<10 = severe dimensional collapse)")
        recommendations.append(
            "Increase covariance_weight (try 0.1) or switch to pool:mean — "
            "if covariance_weight is already set, DiffPool may be incompatible with this contrastive setup"
        )
    elif eff_rank < 20:
        reasons.append(f"WARN: effective rank = {eff_rank:.1f} (10-20 = partial dimensional collapse)")
        recommendations.append("Increase covariance_weight or train for more epochs to allow decorrelation to converge")
    else:
        reasons.append(f"OK: effective rank = {eff_rank:.1f} (>20)")

    if uniformity > -1.0:
        reasons.append(f"FAIL: uniformity = {uniformity:.3f} (>-1.0 = poorly distributed)")
        recommendations.append("Increase augmentation diversity/strength to spread embeddings on hypersphere")
    elif uniformity > -2.0:
        reasons.append(f"WARN: uniformity = {uniformity:.3f} (-2.0 to -1.0 = moderate)")
    else:
        reasons.append(f"OK: uniformity = {uniformity:.3f} (<-2.0)")

    # -- Alignment --
    if not np.isnan(alignment):
        if alignment < 0.3:
            reasons.append(f"WARN: alignment = {alignment:.3f} (<0.3 = augmentations may be too destructive)")
            recommendations.append("Reduce augmentation strength (increase retain/prefix ratios)")
        elif alignment > 0.95:
            reasons.append(f"WARN: alignment = {alignment:.3f} (>0.95 = augmentations too weak / model ignoring them)")
        else:
            reasons.append(f"OK: alignment = {alignment:.3f}")

    # -- Pool-level collapse --
    for pname, pinfo in pool_diag.items():
        if isinstance(pinfo, dict) and "median_cosine_sim" in pinfo:
            if pinfo["median_cosine_sim"] > 0.9:
                reasons.append(f"WARN: {pname} median cosine = {pinfo['median_cosine_sim']:.3f} (>0.9 = pool collapsed)")
                if pname == "pool_2":
                    recommendations.append("Increase diffpool_aux_weight (try 0.2) — pool_2 is collapsing, encoder needs more auxiliary signal")
                else:
                    recommendations.append(f"Reduce diffpool_aux_weight — {pname} is over-regularised")
        # Extra check: pool_0 eff_dim should be high (it's the encoder output before DiffPool)
        if pname == "pool_0" and isinstance(pinfo, dict):
            ed = pinfo.get("effective_dim", 0)
            if ed < 10:
                reasons.append(f"FAIL: pool_0 eff_dim = {ed:.1f} (<10 = encoder representations collapsed)")
                recommendations.append(
                    "pool_0 (raw encoder output) is collapsed — increase diffpool_aux_weight to restore "
                    "node-level diversity signal to the encoder"
                )
            elif ed < 20:
                reasons.append(f"WARN: pool_0 eff_dim = {ed:.1f} (10-20 = encoder partially collapsed)")
        # Root node readout: high eff_dim means root is adding topology-sensitive signal
        if pname == "root_node" and isinstance(pinfo, dict):
            ed = pinfo.get("effective_dim", 0)
            if ed < 10:
                reasons.append(f"WARN: root_node eff_dim = {ed:.1f} (<10 = root representations collapsed; residuals may not be sufficient)")
            elif ed >= 20:
                reasons.append(f"OK: root_node eff_dim = {ed:.1f} (>=20 = root readout adding structural diversity)")

    # -- Overall verdict --
    has_fail = any(r.startswith("FAIL") for r in reasons)
    has_warn = any(r.startswith("WARN") for r in reasons)

    if has_fail:
        verdict = "COLLAPSED"
    elif has_warn:
        verdict = "MARGINAL"
    else:
        verdict = "HEALTHY"

    if not recommendations and verdict != "HEALTHY":
        recommendations.append("Try cosine annealing with warm restarts or extend training with lower LR floor")

    return verdict, "\n".join(reasons), recommendations


# ═════════════════════════════════════════════════════════════════════════════
# 7. Summary printer
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary(collapse, discriminative, pool_diag, verdict, reasoning, recommendations,
                   emb_dim: int = 64, train_log: dict | None = None):
    def _status(val, ok_thresh, warn_thresh, lower_is_better=True):
        if lower_is_better:
            return "OK" if val < ok_thresh else ("WARN" if val < warn_thresh else "FAIL")
        else:
            return "OK" if val > ok_thresh else ("WARN" if val > warn_thresh else "FAIL")

    # ── Training log summary (if available) ──────────────────────────────
    if train_log and train_log.get("available"):
        print("\n" + "=" * 60)
        print("  TRAINING LOG")
        print("=" * 60)
        print(f"  Epochs trained  : {train_log['n_epochs']}")
        print(f"  Loss            : {train_log['initial_loss']:.4f} → {train_log['final_loss']:.4f}  (best={train_log['best_loss']:.4f} @ epoch {train_log['best_epoch']})")
        print(f"  Contrastive     : {train_log.get('final_contrastive', 'n/a')}")
        print(f"  Aux (DiffPool)  : {train_log.get('final_aux', 'n/a')}")
        print(f"  Variance        : {train_log.get('final_variance', 'n/a')}")
        if train_log.get("has_covariance_term"):
            cov_init = train_log.get("initial_covariance", "n/a")
            cov_final = train_log.get("final_covariance", "n/a")
            cov_pct = train_log.get("covariance_reduction_pct")
            cov_status = "OK" if (cov_pct is not None and cov_pct > 30) else "WARN"
            print(f"  Covariance      : {cov_init} → {cov_final}  ({cov_pct}% reduction)  [{cov_status}]")
        else:
            print("  Covariance      : NOT PRESENT (pre-v4 checkpoint)")
        gn = train_log.get("final_grad_norm", 0)
        gclip_frac = train_log.get("grad_clip_fraction", 0)
        gn_status = "OK" if gclip_frac < 0.3 else "WARN"
        print(f"  Grad norm (final): {gn:.3f}  (clipped {gclip_frac*100:.0f}% of epochs)  [{gn_status}]")
        if train_log.get("has_fallback_rate"):
            fb = train_log.get("final_fallback_rate", 0)
            fb_status = "OK" if fb < 0.05 else ("WARN" if fb < 0.20 else "FAIL")
            print(f"  Aug fallback rate: {fb:.3f}  [{fb_status}]")

    print("\n" + "=" * 60)
    print("  EMBEDDING DIAGNOSTICS")
    print("=" * 60)
    print(f"{'Metric':<32} {'Value':>10}   {'Status'}")
    print("-" * 60)

    mc = collapse["median_cosine_sim"]
    print(f"{'Median cosine sim':<32} {mc:>10.4f}   {_status(mc, 0.5, 0.8)}")

    ns = collapse["norm_std"]
    print(f"{'Norm std':<32} {ns:>10.4f}   {'OK' if ns > 0.01 else 'WARN'}")

    er = collapse["effective_rank"]
    print(f"{'Effective rank (SVD)':<32} {er:>10.1f}   {_status(er, 20, 10, lower_is_better=False)}")

    cd = collapse["collapsed_dims"]
    print(f"{'Collapsed dims (<0.01 var)':<32} {cd:>4d}/{emb_dim}   {'OK' if cd < 10 else 'WARN'}")

    t1 = collapse["top1_variance_ratio"]
    print(f"{'Top-1 variance ratio':<32} {t1:>10.4f}   {'WARN' if t1 > 0.5 else 'OK'}")

    t5 = collapse["top5_variance_ratio"]
    print(f"{'Top-5 variance ratio':<32} {t5:>10.4f}")

    u = discriminative["uniformity"]
    print(f"{'Uniformity':<32} {u:>10.4f}   {_status(u, -2.0, -1.0)}")

    a = discriminative["alignment"]
    print(f"{'Alignment':<32} {a:>10.4f}   {'OK' if 0.3 <= a <= 0.95 else 'WARN'}")

    ie = discriminative["inter_event_sim"]
    print(f"{'Inter-event sim':<32} {ie:>10.4f}")

    for ev, info in sorted(discriminative["intra_event_sim"].items()):
        print(f"{'  intra ' + ev:<32} {info['mean']:>10.4f}   (n={info['n_cascades']})")

    has_pool_info = isinstance(pool_diag, dict) and any(k in pool_diag for k in ("pool_0", "root_node"))
    if has_pool_info:
        print("-" * 60)
        for pname in ["pool_0", "pool_1", "pool_2"]:
            if pname in pool_diag:
                pi = pool_diag[pname]
                ed = pi["effective_dim"]
                mc = pi["median_cosine_sim"]
                # pool_0 is the raw encoder output — should be high diversity
                ed_thresh = 20 if pname == "pool_0" else 5
                print(f"{pname + ' eff_dim':<32} {ed:>10.1f}   {'FAIL' if ed < ed_thresh // 2 else ('WARN' if ed < ed_thresh else 'OK')}")
                print(f"{pname + ' median_cos':<32} {mc:>10.4f}   {'WARN' if mc > 0.9 else 'OK'}")
        if "root_node" in pool_diag:
            pi = pool_diag["root_node"]
            ed = pi["effective_dim"]
            mc = pi["median_cosine_sim"]
            print(f"{'root_node eff_dim':<32} {ed:>10.1f}   {'FAIL' if ed < 10 else ('WARN' if ed < 20 else 'OK')}")
            print(f"{'root_node median_cos':<32} {mc:>10.4f}   {'WARN' if mc > 0.9 else 'OK'}")

    print("=" * 60)
    vcolor = {"HEALTHY": "\033[92m", "MARGINAL": "\033[93m", "COLLAPSED": "\033[91m"}
    reset = "\033[0m"
    print(f"\n  VERDICT: {vcolor.get(verdict, '')}{verdict}{reset}\n")

    print("  Reasoning:")
    for line in reasoning.split("\n"):
        print(f"    {line}")

    if recommendations:
        print("\n  Recommendations:")
        for r in recommendations:
            print(f"    - {r}")
    print()


if __name__ == "__main__":
    main()
