from __future__ import annotations

import torch
import torch.nn as nn

from .common import LearnablePooling, ResidualNormalization


class FusionConv2d(nn.Module):
    """Multi-branch spatial layer from the supplied aligned-teacher notebook.

    Branches: 3x3, 3x1, 5x5 and 7x7.  Outputs are concatenated and fused by a
    1x1 convolution so the feature shape stays compatible with the student.
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride=(1, 1),
        groups: int = 1,
    ):
        super().__init__()
        self.conv3x3 = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.conv3x1 = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=(3, 1),
            stride=stride,
            padding=(1, 0),
            bias=False,
            groups=groups,
        )
        self.conv5x5 = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
            groups=groups,
        )
        self.conv7x7 = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
            groups=groups,
        )

        self.bn3x3 = nn.BatchNorm2d(output_channel)
        self.bn3x1 = nn.BatchNorm2d(output_channel)
        self.bn5x5 = nn.BatchNorm2d(output_channel)
        self.bn7x7 = nn.BatchNorm2d(output_channel)

        self.fuse_conv = nn.Conv2d(
            output_channel * 4,
            output_channel,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            self.bn3x3(self.conv3x3(x)),
            self.bn3x1(self.conv3x1(x)),
            self.bn5x5(self.conv5x5(x)),
            self.bn7x7(self.conv7x7(x)),
        ]
        return self.fuse_conv(torch.cat(branches, dim=1))


class AlignedLightEchoBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=(1, 1),
        expansion_factor: int = 6,
    ):
        super().__init__()
        self.use_skip = True
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.input_scaling = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.expand_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.expand_norm = nn.BatchNorm2d(out_channels)
        self.expand_activation = nn.LeakyReLU()

        self.spatial_conv = FusionConv2d(
            out_channels,
            out_channels,
            stride=stride,
            groups=out_channels,
        )
        self.spatial_norm = nn.BatchNorm2d(out_channels)
        self.spatial_activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_scaling(x) if self.use_skip else None
        out = self.input_norm(x)
        out = self.expand_conv(out)
        out = self.expand_activation(out)
        out = self.spatial_norm(out)
        out = self.spatial_conv(out)
        out = self.spatial_activation(out)
        return out + residual if self.use_skip else out


class AlignedLightEchoNet(nn.Module):
    """Higher-capacity network structurally aligned with the compact student."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(1)

        self.conv1 = FusionConv2d(1, 16, stride=(2, 2))
        self.norm1 = nn.BatchNorm2d(16)
        self.activation1 = nn.ReLU()

        self.conv2 = FusionConv2d(16, 32, stride=(2, 2))
        self.activation2 = nn.ReLU()

        self.stage1 = nn.Sequential(
            AlignedLightEchoBlock(32, 32),
            AlignedLightEchoBlock(32, 32),
            AlignedLightEchoBlock(32, 32),
        )
        self.stage1_norm = ResidualNormalization(32)

        self.stage2 = nn.Sequential(
            AlignedLightEchoBlock(32, 32),
            AlignedLightEchoBlock(32, 32),
            AlignedLightEchoBlock(32, 32),
        )
        self.stage2_norm = ResidualNormalization(32)

        self.stage3 = nn.Sequential(
            AlignedLightEchoBlock(32, 64),
        )
        self.stage3_norm = ResidualNormalization(64)

        self.pooling = LearnablePooling(64)
        self.head_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def _forward_features(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = []
        x = self.input_norm(x)

        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        features.append(x)

        x = self.stage1(x) + x
        x = self.stage1_norm(x)
        features.append(x)

        x = self.stage2(x) + x
        x = self.stage2_norm(x)
        features.append(x)

        x = self.stage3(x)
        x = self.stage3_norm(x)
        features.append(x)
        return x, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._forward_features(x)
        x = self.pooling(x)
        x = self.head_norm(x)
        x = self.dropout(x)
        return self.classifier(x)

    def forward_with_features(self, x: torch.Tensor):
        x, features = self._forward_features(x)
        pooled = self.pooling(x)
        pooled = self.head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, features
