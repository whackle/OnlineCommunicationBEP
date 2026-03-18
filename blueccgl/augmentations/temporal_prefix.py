"""Strategy B — Temporal prefix sampling.

Keep the earliest ``prefix_ratio`` fraction of nodes (by temporal_delta),
retaining only edges between surviving nodes.
"""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from blueccgl.augmentations import root_component, subgraph


def augment(
    data: Data,
    prefix_ratio: float = 0.6,
    min_nodes: int = 3,
) -> Data | None:
    """
    Args:
        data: Single-graph PyG Data (node 0 = root, temporal_delta in x[:, 0]).
        prefix_ratio: Fraction of earliest nodes to retain.
        min_nodes: Minimum viable size; returns ``None`` if violated.

    Returns:
        Augmented Data or ``None``.
    """
    N = data.x.shape[0]
    keep_count = max(min_nodes, math.ceil(N * prefix_ratio))
    keep_count = min(keep_count, N)

    # Sort by temporal_delta (feature dim 0); root always has delta=0 → first
    order = torch.argsort(data.x[:, 0])
    keep_set = set(order[:keep_count].tolist())
    keep_set.add(0)  # guarantee root

    keep_mask = torch.zeros(N, dtype=torch.bool)
    for v in keep_set:
        keep_mask[v] = True

    aug = subgraph(data, keep_mask)
    aug = root_component(aug)

    if aug.x.shape[0] < min_nodes:
        return None
    return aug
