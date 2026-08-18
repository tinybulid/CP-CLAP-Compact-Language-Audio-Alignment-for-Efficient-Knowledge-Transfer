from __future__ import annotations

import torch
import torch.nn as nn


class NotebookRepConv2d(nn.Module):
    """Additive 3x3 + 1x3 + 3x1 + 1x1 layer used by external teachers."""

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride=(1, 1),
        groups: int = 1,
        use_branch_bn: bool = False,
    ):
        super().__init__()
        self.use_branch_bn = use_branch_bn
        self.conv1 = nn.Conv2d(
            input_channel, output_channel, 3, stride=stride, padding=1,
            bias=False, groups=groups
        )
        self.conv2 = nn.Conv2d(
            input_channel, output_channel, (1, 3), stride=stride, padding=(0, 1),
            bias=False, groups=groups
        )
        self.conv3 = nn.Conv2d(
            input_channel, output_channel, (3, 1), stride=stride, padding=(1, 0),
            bias=False, groups=groups
        )
        self.conv4 = nn.Conv2d(
            input_channel, output_channel, 1, stride=stride, padding=0,
            bias=False, groups=groups
        )
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.bn4 = nn.BatchNorm2d(output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x),
        ]
        if self.use_branch_bn:
            branches = [
                self.bn1(branches[0]),
                self.bn2(branches[1]),
                self.bn3(branches[2]),
                self.bn4(branches[3]),
            ]
        return branches[0] + branches[1] + branches[2] + branches[3]

    def merged_conv(self) -> nn.Conv2d:
        main = self.conv1.weight.data
        k2 = torch.zeros_like(main)
        k3 = torch.zeros_like(main)
        k4 = torch.zeros_like(main)
        k2[:, :, 1, :] = self.conv2.weight.data.squeeze(2)
        k3[:, :, :, 1] = self.conv3.weight.data.squeeze(3)
        k4[:, :, 1, 1] = self.conv4.weight.data.squeeze(3).squeeze(2)

        merged = nn.Conv2d(
            self.conv1.in_channels,
            self.conv1.out_channels,
            self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            groups=self.conv1.groups,
            bias=False,
        ).to(self.conv1.weight.device)
        with torch.no_grad():
            merged.weight.copy_(main + k2 + k3 + k4)
        return merged


def reparametrize_external_teacher(module: nn.Module) -> nn.Module:
    """Replace compatible additive convolution blocks recursively."""
    for name, child in list(module.named_children()):
        if isinstance(child, NotebookRepConv2d) and not child.use_branch_bn:
            setattr(module, name, nn.Sequential(child.merged_conv()))
        else:
            reparametrize_external_teacher(child)
    return module
