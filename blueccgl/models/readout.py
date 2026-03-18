"""Readout and projector MLPs for BlueCCGL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadoutMLP(nn.Module):
    """Two-layer MLP: concat(pool_levels, sentiment) → embedding.

    Input dimension is computed at construction time based on pooling mode
    and whether sentiment enrichment is enabled.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        use_diffpool: bool = True,
        use_sentiment: bool = True,
    ):
        super().__init__()
        n_pools = 3 if use_diffpool else 1
        sentiment_dim = 3 if use_sentiment else 0
        in_dim = n_pools * hidden_dim + sentiment_dim

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

    Maps embedding_dim → hidden → embedding_dim.
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(z)))
