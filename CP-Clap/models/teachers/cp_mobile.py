from __future__ import annotations

import torch
import torch.nn as nn


def make_divisible(v: float, divisor: int = 8, min_value=None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        activation=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        super().__init__(*layers)


class GlobalResponseNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class CPMobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate, stride):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)
        self.block = nn.Sequential(
            ConvBNAct(in_channels, exp_channels, 1),
            ConvBNAct(
                exp_channels,
                exp_channels,
                3,
                stride=stride,
                padding=1,
                groups=exp_channels,
            ),
            ConvBNAct(exp_channels, out_channels, 1, activation=False),
        )
        self.after_block_norm = GlobalResponseNorm()
        self.after_block_activation = nn.ReLU()
        self.use_shortcut = in_channels == out_channels and stride in (1, (1, 1))

    def forward(self, x):
        result = self.block(x)
        if self.use_shortcut:
            result = result + x
        result = self.after_block_norm(result)
        return self.after_block_activation(result)


class CPMobileTeacher(nn.Module):
    """Mobile external teacher architecture from final-model-a.ipynb."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base_channels = make_divisible(32, 8)
        stage_channels = [
            base_channels,
            make_divisible(base_channels * 2.0, 8),
            make_divisible(base_channels * 4.0, 8),
            make_divisible(base_channels * 8.0, 8),
        ]
        self.in_c = nn.Sequential(
            ConvBNAct(1, stage_channels[0] // 4, 3, stride=2, padding=1),
            ConvBNAct(
                stage_channels[0] // 4,
                stage_channels[0],
                3,
                stride=2,
                padding=1,
            ),
        )

        strides = {2: (2, 2), 4: (2, 1)}
        counts = [3, 3, 2]
        blocks = []
        in_ch = stage_channels[0]
        block_id = 0
        for stage_id, count in enumerate(counts):
            out_ch = stage_channels[stage_id + 1]
            for _ in range(count):
                block_id += 1
                stride = strides.get(block_id, (1, 1))
                blocks.append(CPMobileBlock(in_ch, out_ch, 3.0, stride))
                in_ch = out_ch
        self.stages = nn.Sequential(*blocks)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_ch, num_classes, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_c(x)
        x = self.stages(x)
        x = self.feed_forward(x)
        return x.squeeze(-1).squeeze(-1)
