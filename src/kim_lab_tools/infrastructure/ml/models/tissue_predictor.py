"""Tissue prediction model implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture."""

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        """Initialize block.

        Args:
            in_channels: Number of input channels.
            intermediate_channels: Number of intermediate channels.
            out_channels: Number of output channels.
            stride: Stride for first convolution.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class TissuePredictor(nn.Module):
    """ResNet-101 based model for tissue type prediction."""

    def __init__(self):
        """Initialize model."""
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Residual blocks
        self.conv2 = self._make_layer(64, 64, 256, blocks=3, stride=1)
        self.conv3 = self._make_layer(256, 128, 512, blocks=4, stride=2)
        self.conv4 = self._make_layer(512, 256, 1024, blocks=23, stride=2)
        self.conv5 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)

        self.fc = nn.Linear(2048, 3)

    def _make_layer(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a layer of residual blocks.

        Args:
            in_channels: Number of input channels.
            intermediate_channels: Number of intermediate channels.
            out_channels: Number of output channels.
            blocks: Number of residual blocks.
            stride: Stride for first block.

        Returns:
            Sequential container of blocks.
        """
        layers = []
        layers.append(
            ResidualBlock(
                in_channels, intermediate_channels, out_channels, stride=stride
            )
        )
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, intermediate_channels, out_channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return torch.flatten(x, 1)
