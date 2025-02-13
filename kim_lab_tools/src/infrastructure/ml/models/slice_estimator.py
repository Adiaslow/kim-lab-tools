"""Slice estimation model implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        """Initialize block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Convolution stride.
            downsample: Optional downsampling module.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out


class STN(nn.Module):
    """Spatial transformer network."""

    def __init__(self):
        """Initialize network."""
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        self.fc_loc1 = None
        self.fc_loc2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        xs = self.localization(x)
        xs_size = xs.size()

        if self.fc_loc1 is None or self.fc_loc2 is None:
            flatten_size = xs_size[1] * xs_size[2] * xs_size[3]
            self.fc_loc1 = nn.Linear(flatten_size, 32).to(xs.device)
            self.fc_loc2 = nn.Linear(32, 6).to(xs.device)
            self.fc_loc2.weight.data.zero_()
            self.fc_loc2.bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.fc_loc1(xs))
        theta = self.fc_loc2(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class SliceEstimator(nn.Module):
    """Model for estimating slice parameters."""

    def __init__(self, block: nn.Module, layers: list[int]):
        """Initialize model.

        Args:
            block: Residual block module.
            layers: List of layer sizes.
        """
        super().__init__()
        self.in_channels = 64
        self.stn = STN()

        # Initial layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=3)

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )

        # Output layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc_x = nn.Linear(512, 1)
        self.fc_y = nn.Linear(512, 1)
        self.fc_z = nn.Linear(512, 1)

    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a layer of blocks.

        Args:
            block: Block module.
            out_channels: Number of output channels.
            blocks: Number of blocks.
            stride: Stride for first block.

        Returns:
            Sequential container of blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor [z_depth, x_cut, y_cut].
        """
        x = self.stn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        x = self.global_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x_cut = self.fc_x(x)
        y_cut = self.fc_y(x)
        z_depth = self.fc_z(x)
        return torch.stack([z_depth, x_cut, y_cut], dim=1)
