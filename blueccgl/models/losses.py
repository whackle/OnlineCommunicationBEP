"""Loss functions for BlueCCGL contrastive pre-training."""

import torch
import torch.nn.functional as F


def info_nce_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) loss.

    Args:
        z_i, z_j: [B, D] embeddings of two augmented views.
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Scalar loss.
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    B = z_i.shape[0]
    sim = torch.mm(z_i, z_j.T) / temperature  # [B, B]
    labels = torch.arange(B, device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss
