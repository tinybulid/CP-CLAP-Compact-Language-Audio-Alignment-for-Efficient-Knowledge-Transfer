from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNormalization(nn.Module):
    """Per-channel residual scaling plus instance normalization."""

    def __init__(self, num_features: int):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.ones(num_features, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda_param * x + self.instance_norm(x)


class ScalarResidualNormalization(nn.Module):
    """Scalar variant retained for external teacher compatibility."""

    def __init__(self, num_features: int):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.ones(1))
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda_param * x + self.instance_norm(x)


class LearnablePooling(nn.Module):
    """Attention pooling concatenated with global average pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int | None = None,
        residual_norm: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_channels // 2
        norm = ResidualNormalization if residual_norm else nn.BatchNorm2d

        self.bn_input = norm(in_channels)
        self.attn_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_attn = norm(hidden_dim)
        self.attn_score = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.bn_input(x)
        attn = self.activation(self.bn_attn(self.attn_conv(x_norm)))
        scores = self.attn_score(attn)

        b, c, h, w = x.size()
        weights = F.softmax(scores.view(b, c, -1), dim=-1).view(b, c, h, w)
        attn_pooled = (x * weights).sum(dim=(2, 3))
        gap_pooled = self.global_avg_pool(x).flatten(1)
        return torch.cat([attn_pooled, gap_pooled], dim=1)
