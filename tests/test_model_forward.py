"""Tests for BlueCCGL model: forward pass, gradient flow, shape correctness."""

import torch
import pytest
from torch_geometric.data import Batch, Data

from blueccgl.models.blueccgl import BlueCCGL
from blueccgl.models.losses import info_nce_loss


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_cascade(n_nodes: int = 12, n_follower: int = 8, seed: int = 0) -> Data:
    """Construct a synthetic cascade tree for testing."""
    torch.manual_seed(seed)
    E = n_nodes - 1  # tree

    # Simple chain + branching: 0→1→2→...→n_follower, 0→n_follower+1, ...
    src, tgt = [], []
    for i in range(1, n_nodes):
        parent = max(0, i - 1) if i <= n_follower else 0
        src.append(parent)
        tgt.append(i)

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_type = torch.zeros(E, dtype=torch.long)
    edge_type[n_follower:] = 1  # non-follower edges

    x = torch.randn(n_nodes, 6)
    x[:, 0] = torch.arange(n_nodes, dtype=torch.float)  # temporal delta
    x[0, 0] = 0.0  # root

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        sentiment_composition=torch.tensor([0.3, 0.5, 0.2]),
        follower_edge_ratio=torch.tensor(n_follower / E, dtype=torch.float),
        event_window=torch.tensor(1, dtype=torch.long),
        cascade_depth=torch.tensor(n_follower, dtype=torch.long),
        cascade_breadth=torch.tensor(1, dtype=torch.long),
        wiener_index=torch.tensor(100.0),
        cascade_size=torch.tensor(n_nodes, dtype=torch.long),
        cascade_id=torch.tensor(42, dtype=torch.long),
    )


def _default_cfg(**overrides):
    cfg = {
        "model": {
            "encoder": "rgcn",
            "input_dim": 6,
            "hidden_dim": 32,
            "embedding_dim": 16,
            "rgcn_bases": 2,
            "rgcn_layers": 2,
            "rgcn_dropout": 0.0,
            "pool": "diffpool",
            "pool_ratios": [0.5, 0.25],
            "diffpool_max_nodes": 50,
            "diffpool_aux_weight": 0.1,
            "use_sentiment": True,
        }
    }
    cfg["model"].update(overrides)
    return cfg


def _batch(n_graphs: int = 4, n_nodes: int = 12) -> Batch:
    graphs = [_make_cascade(n_nodes=n_nodes, seed=i) for i in range(n_graphs)]
    return Batch.from_data_list(graphs)


# ── Model forward tests ─────────────────────────────────────────────────────


class TestRGCNDiffPool:
    """R-GCN encoder + DiffPool (default config)."""

    def test_forward_shape(self):
        model = BlueCCGL(_default_cfg())
        batch = _batch(4)
        emb, proj, aux = model(batch)
        assert emb.shape == (4, 16)
        assert proj.shape == (4, 16)
        assert aux.dim() == 0  # scalar

    def test_gradient_flow(self):
        model = BlueCCGL(_default_cfg())
        batch = _batch(4)
        emb, proj, aux = model(batch)
        loss = info_nce_loss(proj[:2], proj[2:]) + 0.1 * aux
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().max() > 0, f"Zero gradient for {name}"

    def test_no_nan_inf(self):
        model = BlueCCGL(_default_cfg())
        batch = _batch(8)
        emb, proj, aux = model(batch)
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()
        assert not torch.isnan(proj).any()

    def test_embedding_variance(self):
        """Embeddings should not collapse to a constant vector."""
        model = BlueCCGL(_default_cfg())
        batch = _batch(8, n_nodes=20)
        emb, _, _ = model(batch)
        assert emb.var(dim=0).mean() > 1e-8, "Embedding variance too low (possible collapse)"


class TestGCNMeanPool:
    """GCN encoder + mean pool (ablation A1 + A3)."""

    def test_forward_shape(self):
        model = BlueCCGL(_default_cfg(encoder="gcn", pool="mean"))
        batch = _batch(4)
        emb, proj, aux = model(batch)
        assert emb.shape == (4, 16)
        assert aux.item() == 0.0

    def test_gradient_flow(self):
        model = BlueCCGL(_default_cfg(encoder="gcn", pool="mean"))
        batch = _batch(4)
        _, proj, _ = model(batch)
        loss = info_nce_loss(proj[:2], proj[2:])
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().max() > 0, f"Zero gradient for {name}"


class TestNoSentiment:
    """Ablation A4: sentiment features zeroed, composition excluded."""

    def test_forward_runs(self):
        model = BlueCCGL(_default_cfg(use_sentiment=False))
        batch = _batch(4)
        emb, proj, aux = model(batch)
        assert emb.shape == (4, 16)


# ── InfoNCE loss tests ───────────────────────────────────────────────────────


class TestInfoNCE:
    def test_shape(self):
        z_i = torch.randn(8, 16)
        z_j = torch.randn(8, 16)
        loss = info_nce_loss(z_i, z_j, temperature=0.5)
        assert loss.dim() == 0

    def test_identical_views_low_loss(self):
        z = torch.randn(8, 16)
        loss = info_nce_loss(z, z, temperature=0.5)
        # Identical views → sim diagonal is 1/τ, off-diagonal noise → loss ≈ 0
        assert loss.item() < 1.0

    def test_gradient_flows(self):
        z_i = torch.randn(8, 16, requires_grad=True)
        z_j = torch.randn(8, 16, requires_grad=True)
        loss = info_nce_loss(z_i, z_j)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None
