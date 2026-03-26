"""R-GCN encoder with basis decomposition and residual connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """N-layer R-GCN with residual connections (layers 1+).

    Uses basis decomposition to share parameters across relation types
    while still allowing per-type weight differentiation.

    Residual connections are applied from layer 1 onward (layer 0 changes
    dims from input_dim → hidden_dim so no skip there). This prevents
    over-smoothing on small graphs where 3 hops covers the entire graph.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_relations: int = 3,
        num_bases: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] node features.
            edge_index: [2, E] edge list.
            edge_type: [E] relation labels (0/1/2).

        Returns:
            [N, hidden_dim] node embeddings.
        """
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index, edge_type)
            if i > 0:                    # skip connection: dims match for layers 1+ (hidden→hidden)
                h = h + x
            if i < len(self.convs) - 1:
                x = F.relu(h)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = h                    # final layer: no activation (ReadoutMLP handles it)
        return x
