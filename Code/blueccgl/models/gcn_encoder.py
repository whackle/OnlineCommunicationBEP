"""Vanilla GCN encoder — ablation A1 (no edge-type encoding)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """Same architecture as RGCNEncoder (with residual connections) but ignores edge types."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """edge_type is accepted but ignored (API compatibility with R-GCN)."""
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            if i > 0:                    # skip connection: dims match for layers 1+ (hidden→hidden)
                h = h + x
            if i < len(self.convs) - 1:
                x = F.relu(h)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = h                    # final layer: no activation
        return x
