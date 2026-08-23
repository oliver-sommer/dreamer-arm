"""Primitive neural-network layers and shared initialization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init as nn_init


def weight_init_(module: nn.Module, fan_type: str = "in") -> None:
    """Initialize weights with fan-scaled truncated normals and zero biases."""
    if isinstance(module, nn.RMSNorm):
        with torch.no_grad():
            module.weight.fill_(1.0)
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)
    fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
    with torch.no_grad():
        std = 1.1368 * float(np.sqrt(1.0 / fan))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0) -> None:
        super().__init__()
        assert in_ch % blocks == 0 and out_ch % blocks == 0
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        return x.reshape(*batch_shape, self.out_ch) + self.bias


class Conv2dSamePad(nn.Conv2d):
    """Conv2d with TensorFlow-compatible SAME padding."""

    @staticmethod
    def _calc_same_pad(i: int, k: int, s: int, d: int) -> int:
        ceil = (i + s - 1) // s
        return max((ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = self._calc_same_pad(height, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(width, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h or pad_w:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over the channel dimension of B,C,H,W tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype: torch.dtype | None = None) -> None:
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _StochReshape(nn.Module):
    def __init__(self, stoch: int, discrete: int) -> None:
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)
