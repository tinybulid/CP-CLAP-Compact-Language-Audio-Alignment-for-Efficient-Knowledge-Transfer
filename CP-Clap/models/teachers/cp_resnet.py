from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def calc_padding(kernel):
    if isinstance(kernel, int):
        return kernel // 3
    return tuple(k // 3 for k in kernel)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k1=3, k2=3, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            groups=groups,
            kernel_size=k1,
            stride=stride,
            padding=calc_padding(k1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            groups=groups,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    bias=False,
                    groups=groups,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu2(y + self.shortcut(x))


class CPResNetTeacher(nn.Module):
    """Compact residual external teacher from final-model-a.ipynb."""

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 32,
        channels_multiplier: int = 2,
        cut_channels_s2: int = 0,
        cut_channels_s3: int = 36,
        n_blocks=(2, 1, 1),
    ):
        super().__init__()
        channels = [
            base_channels,
            base_channels * channels_multiplier - cut_channels_s2,
            base_channels * channels_multiplier * channels_multiplier - cut_channels_s3,
        ]

        self.in_c = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            BasicBlock(channels[0], channels[0], k1=3, k2=1, groups=1),
            nn.MaxPool2d(2, 2),
            BasicBlock(channels[0], channels[0], k1=3, k2=1, groups=1),
        )

        stage2_blocks = []
        in_ch = channels[0]
        for i in range(n_blocks[1]):
            stage2_blocks.append(
                BasicBlock(in_ch, channels[1], k1=3, k2=1, groups=2)
            )
            in_ch = channels[1]
        self.stage2 = nn.Sequential(*stage2_blocks)

        stage3_blocks = []
        for i in range(n_blocks[2]):
            stage3_blocks.append(
                BasicBlock(in_ch, channels[2], k1=1, k2=1, groups=1)
            )
            in_ch = channels[2]
        self.stage3 = nn.Sequential(*stage3_blocks)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_ch, num_classes, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_c(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.feed_forward(x)
        return x.squeeze(-1).squeeze(-1)
