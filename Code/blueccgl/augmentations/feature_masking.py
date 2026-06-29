"""Feature masking augmentation — applied on top of structural augmentations.

Randomly zeroes a fraction of node feature dimensions per graph, forcing the
encoder to learn from topology rather than memorising feature statistics.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def mask_features(
    data: Data,
    mask_ratio: float = 0.25,
    feature_dim: int = 6,
) -> Data:
    """Zero out a random subset of feature dimensions for all nodes.

    Args:
        data: Single-graph PyG Data object.
        mask_ratio: Fraction of feature dimensions to zero (0.0–1.0).
        feature_dim: Total number of node feature dimensions.

    Returns:
        Data with masked node features (new tensor, not in-place).
    """
    if mask_ratio <= 0.0 or data.x.shape[0] == 0:
        return data

    n_mask = max(1, int(feature_dim * mask_ratio))
    dims_to_mask = torch.randperm(feature_dim)[:n_mask]

    x_new = data.x.clone()
    x_new[:, dims_to_mask] = 0.0

    # Return a shallow copy with only x replaced
    new_data = data.clone()
    new_data.x = x_new
    return new_data
