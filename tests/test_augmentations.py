"""Tests for graph augmentation strategies."""

import torch
import pytest
from torch_geometric.data import Data

from blueccgl.augmentations.follower_subgraph import augment as aug_follower
from blueccgl.augmentations.temporal_prefix import augment as aug_temporal
from blueccgl.augmentations.bridge_removal import augment as aug_bridge
from blueccgl.augmentations.random_aug import augment as aug_random


# ── Fixture ──────────────────────────────────────────────────────────────────


def _cascade(n: int = 20, n_non_follower: int = 5) -> Data:
    """Build a test cascade with mixed edge types.

    Structure: chain 0→1→2→...→(n-n_nf-1) with FOLLOWER edges,
    then 0→(n-n_nf)→...→(n-1) with NON_FOLLOWER edges.
    """
    src, tgt, etypes = [], [], []
    # Follower chain
    for i in range(1, n - n_non_follower):
        src.append(i - 1)
        tgt.append(i)
        etypes.append(0)
    # Non-follower branches from root
    for i in range(n - n_non_follower, n):
        src.append(0)
        tgt.append(i)
        etypes.append(1)

    x = torch.zeros(n, 6)
    x[:, 0] = torch.arange(n, dtype=torch.float) * 0.1  # temporal delta
    x[:, 1] = 1.0  # neutral sentiment

    return Data(
        x=x,
        edge_index=torch.tensor([src, tgt], dtype=torch.long),
        edge_type=torch.tensor(etypes, dtype=torch.long),
        sentiment_composition=torch.tensor([0.3, 0.5, 0.2]),
        follower_edge_ratio=torch.tensor(0.75),
        event_window=torch.tensor(1, dtype=torch.long),
        cascade_depth=torch.tensor(n - n_non_follower - 1, dtype=torch.long),
        cascade_breadth=torch.tensor(n_non_follower, dtype=torch.long),
        wiener_index=torch.tensor(100.0),
        cascade_size=torch.tensor(n, dtype=torch.long),
        cascade_id=torch.tensor(99, dtype=torch.long),
    )


def _assert_valid(aug: Data, original: Data):
    """Common validity checks for an augmented graph."""
    N = aug.x.shape[0]
    E = aug.edge_index.shape[1]

    assert N >= 3, f"Too few nodes: {N}"
    assert E >= 1, f"No edges"
    assert aug.x.shape[1] == 6
    assert aug.edge_type.shape[0] == E
    assert (aug.edge_type >= 0).all() and (aug.edge_type <= 2).all()
    assert aug.x[0, 0].item() == 0.0, "Root temporal delta should be 0"

    # Connectedness (undirected BFS from root)
    adj: dict[int, list[int]] = {}
    for s, t in zip(aug.edge_index[0].tolist(), aug.edge_index[1].tolist()):
        adj.setdefault(s, []).append(t)
        adj.setdefault(t, []).append(s)
    visited: set[int] = set()
    q = [0]
    visited.add(0)
    while q:
        nd = q.pop(0)
        for nb in adj.get(nd, []):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    assert len(visited) == N, f"Disconnected: {len(visited)}/{N} reachable"

    # Graph-level attrs preserved
    assert aug.cascade_id.item() == original.cascade_id.item()


# ── Tests ────────────────────────────────────────────────────────────────────


class TestFollowerSubgraph:
    def test_reduces_to_follower_edges(self):
        torch.manual_seed(0)
        d = _cascade(20, 5)
        aug = aug_follower(d, retain_ratio=1.0)
        assert aug is not None
        # All remaining edges should be FOLLOWER
        assert (aug.edge_type == 0).all()
        _assert_valid(aug, d)

    def test_returns_none_if_no_follower_edges(self):
        d = _cascade(10, 9)  # only 1 follower edge → augmented may be too small
        result = aug_follower(d, retain_ratio=0.1, min_nodes=5)
        # Either None or valid
        if result is not None:
            _assert_valid(result, d)


class TestTemporalPrefix:
    def test_keeps_early_nodes(self):
        torch.manual_seed(0)
        d = _cascade(20, 5)
        aug = aug_temporal(d, prefix_ratio=0.5)
        assert aug is not None
        _assert_valid(aug, d)
        assert aug.x.shape[0] <= d.x.shape[0]

    def test_root_always_included(self):
        torch.manual_seed(1)
        d = _cascade(20, 5)
        aug = aug_temporal(d, prefix_ratio=0.3)
        if aug is not None:
            assert aug.x[0, 0].item() == 0.0


class TestBridgeRemoval:
    def test_removes_non_follower_entries(self):
        torch.manual_seed(0)
        d = _cascade(20, 5)
        aug = aug_bridge(d, remove_ratio=1.0)
        assert aug is not None
        _assert_valid(aug, d)
        # Should have removed non-follower entry nodes
        assert aug.x.shape[0] < d.x.shape[0]

    def test_returns_none_if_no_non_follower(self):
        d = _cascade(10, 0)  # all follower
        assert aug_bridge(d) is None


class TestRandomAug:
    def test_produces_valid_subgraph(self):
        torch.manual_seed(0)
        d = _cascade(20, 5)
        aug = aug_random(d, node_drop_ratio=0.2, edge_drop_ratio=0.2)
        assert aug is not None
        _assert_valid(aug, d)

    def test_root_preserved(self):
        torch.manual_seed(42)
        d = _cascade(20, 5)
        aug = aug_random(d, node_drop_ratio=0.5)
        if aug is not None:
            assert aug.x[0, 0].item() == 0.0
