"""Stage: extract_embeddings — Run trained BlueCCGL encoder over full dataset.

Reads:
  {output_dir}/train/checkpoints/best.pt
  {output_dir}/build_pyg_dataset/cascade_dataset.pt

Writes:
  {output_dir}/extract_embeddings/embeddings.h5
  {output_dir}/extract_embeddings/cascade_index.parquet
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import yaml

from blueccgl.models.blueccgl import BlueCCGL


_DIAG_SAMPLE = 8_000  # cap for expensive pairwise ops


def _embedding_diagnostics(embeddings: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    N = len(embeddings)
    idx = rng.choice(N, size=min(_DIAG_SAMPLE, N), replace=False)
    emb = embeddings[idx].astype(np.float64)

    # L2-normalise for cosine-based metrics
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.clip(norms, 1e-8, None)

    # Effective rank (Roy & Vetterli 2007): exp(H) where H = entropy of normalised singular values
    _, sv, _ = np.linalg.svd(emb_n, full_matrices=False)
    sv_sq = sv ** 2
    p = sv_sq / sv_sq.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    # Uniformity (Wang & Isola 2020): mean pairwise Gaussian kernel on unit hypersphere
    # log E[exp(-2 ||z_i - z_j||^2)] — lower (more negative) = more uniform
    diff = emb_n[:, None, :] - emb_n[None, :, :]          # [S, S, D]
    sq_dist = (diff ** 2).sum(-1)                           # [S, S]
    mask = np.triu(np.ones((len(emb_n), len(emb_n)), dtype=bool), k=1)
    uniformity = float(np.log(np.exp(-2 * sq_dist[mask]).mean()))

    # Alignment: mean ||z_i - z_j||^2 for positive pairs
    # Approximate with nearest-neighbour cosine sim as proxy (no pair labels here)
    cos_sim = emb_n @ emb_n.T                               # [S, S]
    np.fill_diagonal(cos_sim, -1.0)
    median_cos = float(np.median(cos_sim.max(axis=1)))      # proxy: median NN sim

    # Top-5 variance ratio
    var_ratio = float(sv_sq[:5].sum() / sv_sq.sum())

    _SEP = "─" * 57
    print(f"\n{_SEP}")
    print("  EMBEDDING DIAGNOSTICS")
    print(_SEP)

    def _fmt(val, lo, hi, invert=False):
        ok = (lo <= val <= hi) if not invert else not (lo <= val <= hi)
        return "OK" if ok else "WARN"

    print(f"  Effective rank        : {eff_rank:.1f} / {embeddings.shape[1]}  "
          f"[target >25]  {_fmt(eff_rank, 25, 1e9)}")
    print(f"  Uniformity (log)      : {uniformity:.3f}  "
          f"[target <-3.0]  {_fmt(uniformity, -1e9, -3.0)}")
    print(f"  Median NN cosine sim  : {median_cos:.3f}  "
          f"[proxy alignment — lower is better]")
    print(f"  Top-5 variance ratio  : {var_ratio:.3f}  "
          f"[target <0.6]  {_fmt(var_ratio, 0, 0.6)}")
    print(_SEP + "\n")


def run(cfg: dict, output_dir: Path, device: str) -> None:
    out_dir = output_dir / "extract_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # ── 1. Load dataset ───────────────────────────────────────────────────
    dataset_path = output_dir / "build_pyg_dataset" / "cascade_dataset.pt"
    print(f"[extract_embeddings] Loading dataset from {dataset_path}")
    data_list = torch.load(dataset_path, weights_only=False)
    print(f"[extract_embeddings] {len(data_list):,} cascades loaded")

    # ── 2. Load model ─────────────────────────────────────────────────────
    ckpt_path = output_dir / "train" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found at {ckpt_path}. "
            "Run --stage train first."
        )
    print(f"[extract_embeddings] Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=dev)

    # Use the config snapshot from training so the architecture always matches
    # the checkpoint — safe even when config.yaml has been updated for a new run.
    snapshot_path = output_dir / "train" / "config_snapshot.yaml"
    if snapshot_path.exists():
        with open(snapshot_path) as f:
            train_cfg = yaml.safe_load(f)
        print(f"[extract_embeddings] Using training config snapshot (pool={train_cfg['model'].get('pool')})")
    else:
        train_cfg = cfg

    model = BlueCCGL(train_cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"[extract_embeddings] Checkpoint: epoch={epoch}  loss={loss:.4f}")

    # ── 3. Extract embeddings ─────────────────────────────────────────────
    loader = DataLoader(data_list, batch_size=32, shuffle=False)

    all_embeddings: list[np.ndarray] = []
    all_cascade_ids: list[int] = []
    all_event_windows: list[int] = []
    all_cascade_sizes: list[int] = []
    all_cascade_depths: list[float] = []
    all_cascade_breadths: list[float] = []
    all_wiener_indices: list[float] = []
    all_follower_ratios: list[float] = []
    all_sent_neg: list[float] = []
    all_sent_neu: list[float] = []
    all_sent_pos: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            emb = model.encode(batch)  # [B, 64]
            B = emb.shape[0]

            all_embeddings.append(emb.cpu().numpy())
            all_cascade_ids.extend(batch.cascade_id.cpu().tolist())
            all_event_windows.extend(batch.event_window.cpu().tolist())
            all_cascade_sizes.extend(batch.cascade_size.cpu().tolist())
            all_cascade_depths.extend(batch.cascade_depth.cpu().tolist())
            all_cascade_breadths.extend(batch.cascade_breadth.cpu().tolist())
            all_wiener_indices.extend(batch.wiener_index.cpu().tolist())
            all_follower_ratios.extend(batch.follower_edge_ratio.cpu().tolist())

            sent = batch.sentiment_composition.view(B, 3).cpu().numpy()
            all_sent_neg.extend(sent[:, 0].tolist())
            all_sent_neu.extend(sent[:, 1].tolist())
            all_sent_pos.extend(sent[:, 2].tolist())

    embeddings = np.vstack(all_embeddings)  # [N, 64]
    print(f"[extract_embeddings] Embedding matrix shape: {embeddings.shape}")

    # ── 4. Sanity checks ──────────────────────────────────────────────────
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"[extract_embeddings] Embedding norms — "
        f"mean={norms.mean():.3f}  std={norms.std():.3f}  "
        f"min={norms.min():.3f}  max={norms.max():.3f}"
    )
    if norms.std() < 0.01:
        print(
            "[extract_embeddings] WARNING: near-zero norm variance — "
            "possible representation collapse. Check training loss."
        )

    # ── 4b. Embedding quality diagnostics ────────────────────────────────
    _embedding_diagnostics(embeddings)

    # ── 5. Save embeddings to HDF5 ────────────────────────────────────────
    emb_path = out_dir / "embeddings.h5"
    with h5py.File(emb_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip", chunks=True)
        f.create_dataset(
            "cascade_ids", data=np.array(all_cascade_ids, dtype=np.int64)
        )
    print(f"[extract_embeddings] Embeddings saved → {emb_path}")

    # ── 6. Save cascade metadata index ───────────────────────────────────
    df = pd.DataFrame(
        {
            "cascade_id": all_cascade_ids,
            "event_window": all_event_windows,
            "cascade_size": all_cascade_sizes,
            "cascade_depth": all_cascade_depths,
            "cascade_breadth": all_cascade_breadths,
            "wiener_index": all_wiener_indices,
            "follower_edge_ratio": all_follower_ratios,
            "sentiment_neg": all_sent_neg,
            "sentiment_neu": all_sent_neu,
            "sentiment_pos": all_sent_pos,
        }
    )
    assert len(df) == embeddings.shape[0], "Index row count != embedding count"
    idx_path = out_dir / "cascade_index.parquet"
    df.to_parquet(idx_path, index=False)
    print(f"[extract_embeddings] Index saved ({len(df):,} rows) → {idx_path}")
