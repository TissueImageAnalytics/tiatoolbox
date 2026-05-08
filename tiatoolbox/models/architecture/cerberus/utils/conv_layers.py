"""Minimal convolution blocks required by the Cerberus ResNet-34 decoder."""

from __future__ import annotations

import torch
from torch import nn


class Conv2d(nn.Module):
    """Convolution wrapper preserving checkpoint module names."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: int,
        *,
        pad: bool = True,
    ) -> None:
        """Initialize the convolution layer."""
        super().__init__()
        pad_size = int(ksize // 2) if pad else 0
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            ksize,
            stride=1,
            padding=pad_size,
            bias=True,
        )

    def forward(self, prev_feat: torch.Tensor) -> torch.Tensor:
        """Apply convolution."""
        return self.conv(prev_feat)


class _ConvLayer(nn.Module):
    """Conv-BN-ReLU block used by the released Cerberus decoder."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: int,
        *,
        pad: bool = True,
    ) -> None:
        """Initialize the convolution, batch normalization, and activation."""
        super().__init__()
        pad_size = int(ksize // 2) if pad else 0
        self.preact = False
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, padding=pad_size, bias=True)

    def forward(self, prev_feat: torch.Tensor) -> torch.Tensor:
        """Apply convolution followed by batch norm and ReLU."""
        feat = self.conv(prev_feat)
        feat = self.bn(feat)
        return self.relu(feat)


class ConvBlock(nn.Module):
    """A sequence of Cerberus convolution layers."""

    def __init__(
        self,
        in_ch: int,
        unit_ch: list[int],
        ksize: int,
        *,
        pad: bool = True,
    ) -> None:
        """Initialize the convolution block."""
        super().__init__()
        self.nr_layers = len(unit_ch)
        self.block = nn.ModuleList()
        for idx in range(self.nr_layers):
            self.block.append(_ConvLayer(in_ch, unit_ch[idx], ksize, pad=pad))
            in_ch = unit_ch[idx]

    def forward(self, prev_feat: torch.Tensor) -> torch.Tensor:
        """Apply each convolution layer in order."""
        feat = prev_feat
        for idx in range(self.nr_layers):
            feat = self.block[idx](feat)
        return feat


class PytorchBase(nn.Module):
    """Sequential wrapper preserving original checkpoint key prefix ``x``."""

    def __init__(self, *args: nn.Module) -> None:
        """Initialize the sequential wrapper."""
        super().__init__()
        self.x = nn.Sequential(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wrapped modules."""
        return self.x(x)
