"""Readout and projector MLPs for BlueCCGL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadoutMLP(nn.Module):
    """Two-layer MLP: concat(pool_levels, root, sentiment) → embedding.

    Input dimension is computed at construction time based on pooling mode,
    whether root-node readout is enabled, and whether sentiment is enabled.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        use_diffpool: bool = True,
        use_sentiment: bool = True,
        use_root: bool = False,
    ):
        super().__init__()
        n_pools = 3 if use_diffpool else 1
        root_dim = hidden_dim if use_root else 0
        sentiment_dim = 3 if use_sentiment else 0
        in_dim = n_pools * hidden_dim + root_dim + sentiment_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, pool_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pool_list: List of tensors to concatenate, each [B, *].
                       Typically [pool_0, pool_1, pool_2, sentiment_composition]
                       or [pool_0, sentiment_composition] for mean-pool mode.
        """
        h = torch.cat(pool_list, dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class ProjectorMLP(nn.Module):
    """Projector head for contrastive loss (discarded after pre-training).

    Configurable depth, width, and BatchNorm (VICReg / Barlow Twins best
    practice).  Deeper projectors absorb collapse, preserving information
    in the pre-projector embedding used for downstream clustering.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        use_bn: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
