"""Standalone SAXNet teacher used by the 70/10/20 distillation pipeline.

This module is deliberately separate from the existing teacher implementation.
"""

import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attention)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))


class SAXBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = self.conv2(self.conv1(x))
        residual = self.channel_attention(residual)
        residual = self.spatial_attention(residual)
        return self.activation(x + residual)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sax = SAXBlock(out_channels, reduction)

    def forward(self, x):
        return self.sax(self.convs(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, reduction=16):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sax = SAXBlock(out_channels, reduction)

    def forward(self, x, skip):
        return self.sax(self.convs(torch.cat((self.up(x), skip), dim=1)))


class SAXNetV1(nn.Module):
    """3D U-Net teacher with channel and spatial attention at every scale."""

    def __init__(self, in_channels=4, out_channels=3, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.bottleneck = EncoderBlock(c4, c5)
        self.pool = nn.MaxPool3d(2)
        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.output = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        x = self.bottleneck(self.pool(skip4))
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output(x)

