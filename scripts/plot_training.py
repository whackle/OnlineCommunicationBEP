#!/usr/bin/env python
"""
plot_training.py — Plot training loss curves from training_log.json.

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py --log outputs/train/training_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(ROOT / "outputs" / "train" / "training_log.json"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "diagnostics" / "training_curves.png"))
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[plot] ERROR: {log_path} not found")
        sys.exit(1)

    with open(log_path) as f:
        log = json.load(f)

    epochs = [e["epoch"] for e in log]
    total   = [e["loss"] for e in log]
    contrast = [e.get("contrastive", 0) for e in log]
    aux      = [e.get("aux", 0) for e in log]
    variance = [e.get("variance", 0) for e in log]
    has_cov  = "covariance" in log[0]
    covariance = [e.get("covariance", 0) for e in log] if has_cov else []
    grad_norms = [e.get("grad_norm_mean", 0) for e in log]

    best_epoch = min(log, key=lambda e: e["loss"])["epoch"]
    best_loss  = min(e["loss"] for e in log)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BlueCCGL v4 — Training Curves", fontsize=13, fontweight="bold")

    # ── Panel 1: Total loss ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, total, color="steelblue", lw=2, label="Total loss")
    ax.axvline(best_epoch, color="red", ls="--", alpha=0.6, label=f"Best epoch {best_epoch}")
    ax.annotate(f"best={best_loss:.4f}", xy=(best_epoch, best_loss),
                xytext=(best_epoch + 2, best_loss + 0.05),
                fontsize=8, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(alpha=0.3)

    # ── Panel 2: Loss components ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, contrast,  lw=1.8, label="Contrastive (InfoNCE)", color="steelblue")
    ax.plot(epochs, variance,  lw=1.5, label="Variance (VICReg)", color="darkorange", ls="--")
    if has_cov:
        ax.plot(epochs, covariance, lw=1.5, label="Covariance (VICReg)", color="green", ls="-.")
    ax.plot(epochs, aux,       lw=1.2, label="Aux (DiffPool)", color="grey", ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(alpha=0.3)

    # ── Panel 3: Gradient norms ──────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, grad_norms, color="purple", lw=1.5, label="Grad norm (mean)")
    clip_val = 2.0  # v4 clip norm
    ax.axhline(clip_val, color="red", ls="--", alpha=0.6, label=f"Clip threshold ({clip_val})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Gradient Norm")
    ax.set_title("Gradient Norms")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")

    # Print summary
    print(f"\n  Epochs        : {len(log)}")
    print(f"  Initial loss  : {total[0]:.4f}")
    print(f"  Final loss    : {total[-1]:.4f}")
    print(f"  Best loss     : {best_loss:.4f} @ epoch {best_epoch}")
    print(f"  Contrastive   : {contrast[0]:.4f} → {contrast[-1]:.4f}")
    if has_cov:
        print(f"  Covariance    : {covariance[0]:.5f} → {covariance[-1]:.5f}")
    print(f"  Grad norm     : {grad_norms[0]:.3f} → {grad_norms[-1]:.3f}")


if __name__ == "__main__":
    main()
