"""Hierarchical DiffPool with sentiment enrichment.

Two pooling levels (default ratios 0.5, 0.25) produce progressively coarser
graph representations.  At each level a global mean pool is extracted.
The three level pools are returned for concatenation with sentiment features
in the readout MLP.

Auxiliary losses (link-prediction + entropy) are summed and returned so the
caller can weight them via ``diffpool_aux_weight``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class HierarchicalDiffPool(nn.Module):
    """Two-level DiffPool operating on dense graph representations.

    Expects R-GCN-encoded node embeddings in *sparse* PyG format
    (flat ``x``, ``edge_index``, ``batch``).  Internally converts to dense
    tensors for the DiffPool operation.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        pool_ratios: list[float] | None = None,
        max_nodes: int = 1000,
    ):
        super().__init__()
        pool_ratios = pool_ratios or [0.5, 0.25]
        self.max_nodes = max_nodes

        k1 = math.ceil(max_nodes * pool_ratios[0])
        k2 = math.ceil(max_nodes * pool_ratios[1])

        # Level 1
        self.embed_gnn_1 = DenseGCNConv(hidden_dim, hidden_dim)
        self.pool_gnn_1 = DenseGCNConv(hidden_dim, k1)

        # Level 2
        self.embed_gnn_2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.pool_gnn_2 = DenseGCNConv(hidden_dim, k2)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_assignments: bool = False,
    ) -> tuple:
        """
        Args:
            x: [N_total, hidden_dim]  (sparse, after R-GCN encoding).
            edge_index: [2, E_total].
            batch: [N_total] graph-membership vector.
            return_assignments: If True, also return soft assignment matrices
                and the dense batch mask for XAI analysis.

        Returns (default):
            pool_0: [B, hidden_dim]  global mean pool of encoder output.
            pool_1: [B, hidden_dim]  global mean pool after DiffPool level 1.
            pool_2: [B, hidden_dim]  global mean pool after DiffPool level 2.
            aux_loss: scalar  (link-prediction + entropy, summed over levels).

        Returns (return_assignments=True), additionally:
            s1: [B, N_max, k1]  soft assignment matrix, Level 1.
            s2: [B, k1, k2]     soft assignment matrix, Level 2.
            mask: [B, N_max]    bool padding mask (True = real node).
        """
        # Sparse → dense
        x_dense, mask = to_dense_batch(x, batch)   # [B, N_max, F]
        adj = to_dense_adj(edge_index, batch)       # [B, N_max, N_max]

        # Level 0: global mean pool (pre-DiffPool)
        pool_0 = _masked_mean(x_dense, mask)        # [B, F]

        # Level 1
        z1 = F.relu(self.embed_gnn_1(x_dense, adj, mask))
        s1 = self.pool_gnn_1(x_dense, adj, mask)
        x1, adj1, link1, ent1 = dense_diff_pool(z1, adj, s1, mask)
        pool_1 = x1.mean(dim=1)                     # [B, F]

        # Level 2
        z2 = F.relu(self.embed_gnn_2(x1, adj1))
        s2 = self.pool_gnn_2(x1, adj1)
        x2, _adj2, link2, ent2 = dense_diff_pool(z2, adj1, s2)
        pool_2 = x2.mean(dim=1)                     # [B, F]

        aux_loss = link1 + ent1 + link2 + ent2

        if return_assignments:
            return pool_0, pool_1, pool_2, aux_loss, s1, s2, mask
        return pool_0, pool_1, pool_2, aux_loss


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Global mean pool respecting the padding mask.

    Args:
        x: [B, N, F].
        mask: [B, N] bool.
    """
    m = mask.unsqueeze(-1).float()                   # [B, N, 1]
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
