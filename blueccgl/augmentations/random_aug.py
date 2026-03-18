"""Generic random augmentation — ablation A2.

Random node dropping (20%) + random edge dropping (20%).
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from blueccgl.augmentations import root_component, subgraph


def augment(
    data: Data,
    node_drop_ratio: float = 0.2,
    edge_drop_ratio: float = 0.2,
    min_nodes: int = 3,
) -> Data | None:
    """
    Args:
        data: Single-graph PyG Data.
        node_drop_ratio: Fraction of nodes to drop (root is protected).
        edge_drop_ratio: Fraction of edges to drop.
        min_nodes: Minimum viable size; returns ``None`` if violated.

    Returns:
        Augmented Data or ``None``.
    """
    N = data.x.shape[0]
    E = data.edge_index.shape[1]

    # ── Node dropping (protect root = node 0) ────────────────────────────
    keep_mask = torch.rand(N) >= node_drop_ratio
    keep_mask[0] = True  # always keep root

    if keep_mask.sum().item() < min_nodes:
        return None

    aug = subgraph(data, keep_mask)

    # ── Edge dropping ────────────────────────────────────────────────────
    E_new = aug.edge_index.shape[1]
    if E_new > 0:
        edge_keep = torch.rand(E_new) >= edge_drop_ratio
        if edge_keep.sum().item() > 0:
            aug = Data(
                x=aug.x,
                edge_index=aug.edge_index[:, edge_keep],
                edge_type=aug.edge_type[edge_keep],
                sentiment_composition=aug.sentiment_composition,
                follower_edge_ratio=aug.follower_edge_ratio,
                event_window=aug.event_window,
                cascade_depth=aug.cascade_depth,
                cascade_breadth=aug.cascade_breadth,
                wiener_index=aug.wiener_index,
                cascade_size=aug.cascade_size,
                cascade_id=aug.cascade_id,
            )

    aug = root_component(aug)

    if aug.x.shape[0] < min_nodes:
        return None
    return aug
