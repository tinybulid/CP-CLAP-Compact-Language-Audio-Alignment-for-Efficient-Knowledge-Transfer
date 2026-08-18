from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ScalarResidualNormalization
from .repconv import NotebookRepConv2d


class AmirPooling(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_channels // 2
        self.bn_input = nn.BatchNorm2d(in_channels)
        self.attn_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn_attn = nn.BatchNorm2d(hidden_dim)
        self.attn_score = nn.Conv2d(hidden_dim, in_channels, 1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        scores = self.attn_score(self.act(self.bn_attn(self.attn_conv(self.bn_input(x)))))
        b, c, h, w = x.shape
        weights = F.softmax(scores.view(b, c, -1), dim=-1).view(b, c, h, w)
        a = (x * weights).view(b, c, -1).sum(-1)
        g = self.gap(x).view(b, c)
        return torch.cat([a, g], dim=1)


class AmirBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), expansion_factor=6):
        super().__init__()
        mid_channels = out_channels
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.scaleinput = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout2d(0.0)

        self.norm2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = NotebookRepConv2d(
            mid_channels, mid_channels, stride=stride, groups=mid_channels,
            use_branch_bn=False,
        )
        self.act2 = nn.LeakyReLU()
        self.norm3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, 1, bias=False, groups=mid_channels
        )
        self.dropout3 = nn.Dropout2d(0.15)
        self.norm4 = ScalarResidualNormalization(out_channels)

    def forward(self, x):
        scaled = self.scaleinput(x)
        out = self.norm1(x)
        out = self.dropout1(self.act1(self.conv1(out)))
        out = self.act2(self.conv2(self.norm2(out)))
        out = self.dropout3(self.act2(self.conv3(self.norm3(out))))
        return self.norm4(out + scaled)


class DSFAmirTeacher(nn.Module):
    """First DSFlexiNet external teacher architecture from final-model-a.ipynb."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.initial_conv1 = NotebookRepConv2d(1, 16, stride=(2, 2))
        self.rn1 = nn.BatchNorm2d(1)
        self.act1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.1)

        self.initial_conv2 = NotebookRepConv2d(16, 32, stride=(2, 2))
        self.rn2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()

        self.stage1 = nn.Sequential(
            AmirBlock(32, 32),
            AmirBlock(32, 32),
            AmirBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            AmirBlock(64, 64),
            AmirBlock(64, 64),
            AmirBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            AmirBlock(96, 128),
            AmirBlock(128, 128),
            AmirBlock(128, 128),
        )

        self.avgpool_3 = AmirPooling(128)
        self.avgpool_2 = AmirPooling(64)
        self.avgpool_1 = AmirPooling(32)
        self.bn_final = nn.BatchNorm1d((128 + 64 + 32) * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear((128 + 64 + 32) * 2, num_classes)

    def forward(self, x, recording_device=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.rn1(x)
        x = self.act1(self.initial_conv1(x))
        x = self.rn2(x)
        x = self.dropout_1(self.act2(self.initial_conv2(x)))

        x1o = self.stage1(x)
        x1 = torch.cat([x, x1o], dim=1)
        x2o = self.stage2(x1)
        x2 = torch.cat([x, x2o], dim=1)
        x3o = self.stage3(x2)

        pooled = torch.cat(
            [self.avgpool_3(x3o), self.avgpool_2(x2o), self.avgpool_1(x1o)],
            dim=1,
        )
        pooled = self.dropout(pooled)
        pooled = self.bn_final(pooled)
        return self.fc1(pooled)
