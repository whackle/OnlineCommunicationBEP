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


def vicreg_variance_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    target_std: float = 1.0,
) -> torch.Tensor:
    """VICReg-style variance regularisation (Bardes et al., 2022).

    Penalises any embedding dimension whose batch standard deviation falls
    below ``target_std``, preventing dimensional collapse.

    Args:
        z_i, z_j: [B, D] embedding batches from two views.
        target_std: Target per-dimension std (hinge threshold).

    Returns:
        Scalar variance loss (lower = dimensions are well-spread).
    """
    loss = torch.tensor(0.0, device=z_i.device)
    for z in (z_i, z_j):
        std = z.std(dim=0)  # [D]
        loss = loss + F.relu(target_std - std).mean()
    return loss / 2


def vicreg_covariance_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
) -> torch.Tensor:
    """VICReg-style covariance regularisation (Bardes et al., 2022).

    Penalises off-diagonal elements of the covariance matrix, forcing
    embedding dimensions to be decorrelated and preventing dimensional
    collapse where multiple dims encode the same direction.

    Args:
        z_i, z_j: [B, D] embedding batches from two views.

    Returns:
        Scalar covariance loss (lower = dimensions are decorrelated).
    """
    loss = torch.tensor(0.0, device=z_i.device)
    for z in (z_i, z_j):
        z_c = z - z.mean(dim=0)
        N, D = z_c.shape
        cov = (z_c.T @ z_c) / (N - 1)           # [D, D]
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        loss = loss + off_diag / D
    return loss / 2
