"""R-GCN encoder with basis decomposition for edge-type-aware message passing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """3-layer R-GCN: input_dim → hidden_dim (×num_layers).

    Uses basis decomposition to share parameters across relation types
    while still allowing per-type weight differentiation.
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
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
