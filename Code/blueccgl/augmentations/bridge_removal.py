"""Strategy C — Bridge-node removal.

Identify "bridge nodes" whose incoming edge is NON_FOLLOWER (edge_type == 1),
randomly remove a fraction of them, then keep the root's connected component.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from blueccgl.augmentations import root_component, subgraph


def augment(
    data: Data,
    remove_ratio: float = 0.3,
    min_nodes: int = 3,
) -> Data | None:
    """
    Args:
        data: Single-graph PyG Data.
        remove_ratio: Fraction of bridge nodes to remove.
        min_nodes: Minimum viable size; returns ``None`` if violated.

    Returns:
        Augmented Data or ``None``.
    """
    # Bridge nodes: target of a NON_FOLLOWER edge (edge_type == 1)
    non_follower_mask = data.edge_type == 1
    if non_follower_mask.sum().item() == 0:
        return None  # no bridge nodes to remove

    bridge_nodes = set(data.edge_index[1, non_follower_mask].tolist())
    bridge_nodes.discard(0)  # never remove root

    if not bridge_nodes:
        return None

    # Sample nodes to remove
    bridge_list = list(bridge_nodes)
    n_remove = max(1, int(len(bridge_list) * remove_ratio))
    perm = torch.randperm(len(bridge_list))[:n_remove]
    remove_set = {bridge_list[i] for i in perm.tolist()}

    # Keep everything except removed nodes
    N = data.x.shape[0]
    keep_mask = torch.ones(N, dtype=torch.bool)
    for v in remove_set:
        keep_mask[v] = False

    aug = subgraph(data, keep_mask)
    aug = root_component(aug)

    if aug.x.shape[0] < min_nodes:
        return None
    return aug
