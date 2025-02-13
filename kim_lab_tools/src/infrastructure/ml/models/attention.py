"""Attention module implementation."""

import torch.nn as nn


class AttentionBlock(nn.Module):
    """Attention block for focusing on relevant features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """Initialize attention block.

        Args:
            F_g: Number of feature maps in gating signal.
            F_l: Number of feature maps in input signal.
            F_int: Number of intermediate feature maps.
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """Forward pass.

        Args:
            g: Gating signal.
            x: Input signal.

        Returns:
            Attention weighted input.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
