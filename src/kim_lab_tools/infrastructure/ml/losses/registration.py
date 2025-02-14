"""Registration loss functions."""

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM


class SSIMLoss(SSIM):
    """SSIM-based loss function."""

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            SSIM loss value.
        """
        return 100 * (1 - super().forward(img1, img2))


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """Compute deformation field smoothness loss.

    Args:
        flow: Deformation field tensor.

    Returns:
        Smoothness loss value.
    """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)


def create_grid(batch_size: int, shape: tuple) -> torch.Tensor:
    """Create sampling grid.

    Args:
        batch_size: Batch size.
        shape: Grid shape (H, W).

    Returns:
        Sampling grid tensor.
    """
    H, W = shape
    tensors = [torch.linspace(-1, 1, s) for s in [H, W]]
    grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
