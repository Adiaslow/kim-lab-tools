"""Brain registration UNet implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionBlock


class BrainRegUNet(nn.Module):
    """UNet model for brain image registration."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()

        # Encoder
        self.encoder1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder4 = self._conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder3 = self._conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder2 = self._conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.decoder1 = self._conv_block(64, 32)

        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Encoding
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoding with attention
        dec4 = self.upconv4(bottleneck)
        enc4 = self.crop_and_concat(enc4, dec4)
        att4 = self.att4(dec4, enc4)
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.crop_and_concat(enc3, dec3)
        att3 = self.att3(dec3, enc3)
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.crop_and_concat(enc2, dec2)
        att2 = self.att2(dec2, enc2)
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.crop_and_concat(enc1, dec1)
        att1 = self.att1(dec1, enc1)
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            Convolution block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def crop_and_concat(
        self, upsampled: torch.Tensor, bypass: torch.Tensor
    ) -> torch.Tensor:
        """Crop and concatenate feature maps.

        Args:
            upsampled: Upsampled feature maps.
            bypass: Bypass connection feature maps.

        Returns:
            Cropped and concatenated feature maps.
        """
        crop_size = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-crop_size, -crop_size, -crop_size, -crop_size))
        return bypass
