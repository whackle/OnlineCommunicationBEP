"""Strategy A — Follower-subgraph extraction.

Keep only FOLLOWER-typed edges (edge_type == 0), randomly drop a fraction,
then retain the connected component containing the root.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from blueccgl.augmentations import root_component, subgraph


def augment(
    data: Data,
    retain_ratio: float = 0.7,
    min_nodes: int = 3,
) -> Data | None:
    """
    Args:
        data: Single-graph PyG Data.
        retain_ratio: Fraction of FOLLOWER edges to keep.
        min_nodes: Minimum viable subgraph size; returns ``None`` if violated.

    Returns:
        Augmented Data or ``None`` if result is too small.
    """
    follower_mask = data.edge_type == 0
    n_follower = follower_mask.sum().item()

    if n_follower == 0:
        return None  # no follower edges to work with

    # Randomly retain a subset of follower edges
    keep_count = max(1, int(n_follower * retain_ratio))
    follower_idx = follower_mask.nonzero(as_tuple=True)[0]
    perm = torch.randperm(n_follower)[:keep_count]
    selected = follower_idx[perm]

    # Build a new edge mask: only the selected follower edges survive
    edge_keep = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    edge_keep[selected] = True

    # Construct subgraph with only the kept edges
    new_edge_index = data.edge_index[:, edge_keep]
    new_edge_type = data.edge_type[edge_keep]

    # Identify which nodes are still connected
    surviving_nodes = set(new_edge_index[0].tolist() + new_edge_index[1].tolist())
    surviving_nodes.add(0)  # always keep root

    keep_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    for v in surviving_nodes:
        keep_mask[v] = True

    aug = subgraph(data, keep_mask)
    aug = root_component(aug)

    if aug.x.shape[0] < min_nodes:
        return None
    return aug
