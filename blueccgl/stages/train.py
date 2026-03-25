"""
Stage: train — Contrastive pre-training of BlueCCGL.

Reads:
  {output_dir}/build_pyg_dataset/cascade_dataset.pt

Writes:
  {output_dir}/train/checkpoints/epoch_{N}.pt
  {output_dir}/train/checkpoints/best.pt
  {output_dir}/train/config_snapshot.yaml
  {output_dir}/train/training_log.json
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from blueccgl.models.blueccgl import BlueCCGL
from blueccgl.models.losses import info_nce_loss, vicreg_covariance_loss, vicreg_variance_loss


def run(
    cfg: dict,
    output_dir: Path,
    device: str,
    seed: int,
    ablation: str | None,
) -> None:
    out_dir = output_dir / "train"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
    acfg = cfg["augmentation"]

    # ── 1. Load dataset ──────────────────────────────────────────────────
    dataset_path = output_dir / "build_pyg_dataset" / "cascade_dataset.pt"
    print(f"[train] Loading dataset from {dataset_path}")
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"[train] {len(data_list):,} cascades loaded")

    # ── 2. Build model ───────────────────────────────────────────────────
    dev = torch.device(device)
    model = BlueCCGL(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model: {n_params:,} parameters  device={dev}")

    if ablation == "A5_random_init":
        print("[train] Ablation A5: skipping training (random init)")
        _save_checkpoint(model, None, 0, 0.0, ckpt_dir / "best.pt")
        _save_config(cfg, out_dir)
        return

    # ── 3. Augmentation setup ────────────────────────────────────────────
    augment_pair = _build_augmentor(acfg)

    # ── 4. DataLoader ────────────────────────────────────────────────────
    batch_size = tcfg.get("batch_size", 64)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # ── 5. Optimizer + scheduler ─────────────────────────────────────────
    lr = tcfg.get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = tcfg.get("epochs", 100)
    warmup = tcfg.get("warmup_epochs", 5)
    lr_min = tcfg.get("lr_min", 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup, eta_min=lr_min,
    )

    temperature = tcfg.get("temperature", 0.5)
    grad_clip = tcfg.get("grad_clip_norm", 1.0)
    aux_weight = cfg["model"].get("diffpool_aux_weight", 0.1)
    variance_weight = tcfg.get("variance_weight", 0.0)
    covariance_weight = tcfg.get("covariance_weight", 0.0)
    ckpt_every = tcfg.get("checkpoint_every", 10)
    log_grad_every = tcfg.get("log_grad_norm_every", 50)

    # ── 6. Early stopping + resume setup ────────────────────────────────
    patience = tcfg.get("early_stopping_patience", 15)
    min_delta = tcfg.get("early_stopping_min_delta", 1e-4)

    # Resume from last.pt if it exists (e.g. after a crash / VM preemption)
    start_epoch = 1
    best_loss = float("inf")
    epochs_without_improvement = 0
    log: list[dict] = []

    last_ckpt = ckpt_dir / "last.pt"
    if last_ckpt.exists():
        print(f"[train] Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, weights_only=False, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        log = ckpt.get("log", [])
        print(f"[train] Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

    # ── 7. Training loop ─────────────────────────────────────────────────
    print(f"[train] epochs={epochs}  batch_size={batch_size}  lr={lr}  "
          f"τ={temperature}  patience={patience}")
    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_contrast = 0.0
        epoch_aux = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        n_batches = 0
        grad_norms: list[float] = []

        pbar = tqdm(loader, desc=f"epoch {epoch:>3}/{epochs}", unit="batch", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Generate two augmented views per cascade in this batch
            originals = batch.to_data_list()
            view1_list, view2_list = [], []
            for d in originals:
                v1, v2 = augment_pair(d)
                view1_list.append(v1)
                view2_list.append(v2)

            from torch_geometric.data import Batch
            b1 = Batch.from_data_list(view1_list).to(dev)
            b2 = Batch.from_data_list(view2_list).to(dev)

            # Forward
            _, proj1, aux1 = model(b1)
            _, proj2, aux2 = model(b2)

            # Loss
            contrastive = info_nce_loss(proj1, proj2, temperature)
            aux = aux_weight * (aux1 + aux2) / 2
            var_loss = variance_weight * vicreg_variance_loss(proj1, proj2) if variance_weight > 0 else torch.tensor(0.0, device=dev)
            cov_loss = covariance_weight * vicreg_covariance_loss(proj1, proj2) if covariance_weight > 0 else torch.tensor(0.0, device=dev)
            loss = contrastive + aux + var_loss + cov_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_contrast += contrastive.item()
            epoch_aux += aux.item()
            epoch_var += var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss
            epoch_cov += cov_loss.item() if isinstance(cov_loss, torch.Tensor) else cov_loss
            n_batches += 1

            if batch_idx % log_grad_every == 0:
                grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        # LR schedule (with warmup)
        if epoch <= warmup:
            warmup_lr = lr * epoch / warmup
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_contrast = epoch_contrast / max(n_batches, 1)
        avg_aux = epoch_aux / max(n_batches, 1)
        avg_var = epoch_var / max(n_batches, 1)
        avg_cov = epoch_cov / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "loss": round(avg_loss, 5),
            "contrastive": round(avg_contrast, 5),
            "aux": round(avg_aux, 5),
            "variance": round(avg_var, 5),
            "covariance": round(avg_cov, 5),
            "lr": current_lr,
            "grad_norm_mean": round(sum(grad_norms) / max(len(grad_norms), 1), 4),
            "time_s": round(time.time() - t_start, 1),
        }
        log.append(epoch_log)

        # Progress
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}  "
                f"contrast={avg_contrast:.4f}  aux={avg_aux:.4f}  "
                f"lr={current_lr:.2e}  grad={epoch_log['grad_norm_mean']:.3f}"
            )

        # ── Best model checkpoint ────────────────────────────────────────
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            _save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        # ── Periodic checkpoint ──────────────────────────────────────────
        if epoch % ckpt_every == 0:
            _save_checkpoint(
                model, optimizer, epoch, avg_loss,
                ckpt_dir / f"epoch_{epoch}.pt",
            )

        # ── Always save last.pt (crash recovery) ────────────────────────
        _save_resumable(
            model, optimizer, scheduler, epoch, avg_loss,
            best_loss, epochs_without_improvement, log,
            last_ckpt,
        )

        # ── Early stopping ───────────────────────────────────────────────
        if epochs_without_improvement >= patience:
            print(
                f"[train] Early stopping at epoch {epoch} — "
                f"no improvement for {patience} epochs (best={best_loss:.4f})"
            )
            break

    # ── 8. Save final artefacts ──────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"[train] Done in {elapsed / 60:.1f} min.  Best loss: {best_loss:.4f}")

    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    _save_config(cfg, out_dir)

    _print_summary(log, elapsed)


# ── Augmentation pair generator ──────────────────────────────────────────────


def _build_augmentor(acfg: dict):
    """Return a callable (Data) → (view1, view2)."""
    min_nodes = acfg.get("min_nodes_after_aug", 3)
    feat_mask_ratio = acfg.get("feature_mask_ratio", 0.0)
    feat_dim = 6  # temporal, sent_label, sent_score, depth, in_deg, out_deg

    if feat_mask_ratio > 0:
        from blueccgl.augmentations.feature_masking import mask_features

    if acfg.get("strategy", "feed_aware") == "feed_aware":
        from blueccgl.augmentations.follower_subgraph import augment as aug_A
        from blueccgl.augmentations.temporal_prefix import augment as aug_B
        from blueccgl.augmentations.bridge_removal import augment as aug_C

        fol_ratio = acfg.get("follower_retain_ratio", 0.7)
        temp_ratio = acfg.get("temporal_prefix_ratio", 0.6)
        bridge_ratio = acfg.get("bridge_remove_ratio", 0.3)

        strategies = [
            lambda d: aug_A(d, retain_ratio=fol_ratio, min_nodes=min_nodes),
            lambda d: aug_B(d, prefix_ratio=temp_ratio, min_nodes=min_nodes),
            lambda d: aug_C(d, remove_ratio=bridge_ratio, min_nodes=min_nodes),
        ]
    else:
        # Ablation A2: random augmentation only
        from blueccgl.augmentations.random_aug import augment as aug_R
        strategies = [
            lambda d: aug_R(d, min_nodes=min_nodes),
            lambda d: aug_R(d, min_nodes=min_nodes),
        ]

    def pair(data):
        """Apply two different random strategies + feature masking; fall back to original."""
        s1, s2 = random.sample(range(len(strategies)), k=min(2, len(strategies)))
        v1 = strategies[s1](data) or data
        v2 = strategies[s2](data) or data
        # Apply independent feature masking to each view
        if feat_mask_ratio > 0:
            v1 = mask_features(v1, mask_ratio=feat_mask_ratio, feature_dim=feat_dim)
            v2 = mask_features(v2, mask_ratio=feat_mask_ratio, feature_dim=feat_dim)
        return v1, v2

    return pair


# ── Helpers ──────────────────────────────────────────────────────────────────


def _save_resumable(model, optimizer, scheduler, epoch, loss, best_loss,
                    epochs_without_improvement, log, path):
    """Save full training state for crash recovery."""
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "log": log,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, path)


def _save_checkpoint(model, optimizer, epoch, loss, path):
    state = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def _save_config(cfg, out_dir):
    (out_dir / "config_snapshot.yaml").write_text(yaml.dump(cfg, default_flow_style=False))


def _print_summary(log, elapsed):
    print(f"\n[train] ── Training Summary ──────────────────────────")
    print(f"  Epochs         : {len(log)}")
    print(f"  Final loss     : {log[-1]['loss']:.4f}")
    print(f"  Best loss      : {min(e['loss'] for e in log):.4f}")
    print(f"  Initial loss   : {log[0]['loss']:.4f}")
    print(f"  Total time     : {elapsed / 60:.1f} min")
    print(f"[train] ────────────────────────────────────────────────\n")
