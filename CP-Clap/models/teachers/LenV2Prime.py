from __future__ import annotations

import torch
import torch.nn as nn

from ..common import ScalarResidualNormalization, LearnablePooling
from .repconv import NotebookRepConv2d


class MehdyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), expansion_factor=6):
        super().__init__()
        self.use_skip = in_channels <= out_channels
        mid_channels = in_channels * expansion_factor

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.scaleinput = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.act1 = nn.LeakyReLU()

        self.norm2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = NotebookRepConv2d(
            mid_channels, mid_channels, stride=stride, groups=mid_channels,
            use_branch_bn=False,
        )
        self.act2 = nn.LeakyReLU()

        self.norm3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.dropout3 = nn.Dropout2d(0.1)
        self.norm4 = ScalarResidualNormalization(out_channels)

    def forward(self, x):
        scaled = self.scaleinput(x) if self.use_skip else None
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(self.norm2(out)))
        out = self.dropout3(self.conv3(self.norm3(out)))
        if scaled is not None:
            out = out + scaled
        return self.norm4(out)


class DSFMehdyTeacher(nn.Module):
    """Second DSFlexiNet external teacher architecture from final-model-a.ipynb."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.initial_conv1 = NotebookRepConv2d(1, 16, stride=(2, 2))
        self.rn1 = ScalarResidualNormalization(16)
        self.act1 = nn.ReLU()

        self.initial_conv2 = NotebookRepConv2d(16, 16, stride=(2, 2))
        self.rn2 = ScalarResidualNormalization(16)
        self.act2 = nn.ReLU()

        self.stage1 = nn.Sequential(
            MehdyBlock(16, 16),
            MehdyBlock(16, 16),
            MehdyBlock(16, 16),
        )
        self.stage2 = nn.Sequential(
            MehdyBlock(16, 16),
            MehdyBlock(16, 16),
        )
        self.stage3 = nn.Sequential(MehdyBlock(16, 16))

        self.avgpool = LearnablePooling(16, residual_norm=False)
        self.bn_final = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x, recording_device=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.act1(self.rn1(self.initial_conv1(x)))
        x = self.act2(self.rn2(self.initial_conv2(x)))
        x = self.stage1(x) + x
        x = self.stage2(x) + x
        x = self.stage3(x)
        x = self.avgpool(x)
        x = self.bn_final(x)
        x = self.dropout(x)
        x = self.act3(self.fc1(x))
        return self.fc2(x)
