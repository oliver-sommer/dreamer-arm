"""Shape-only tests for the conv encoder / decoder.

These tests catch the common failure mode where conv strides or paddings
get tweaked and break round-trip shape compatibility between the
:class:`ConvEncoder` and :class:`ConvDecoder`.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dreamer_arm.architecture.networks import ConvEncoder


def test_conv_encoder_shape() -> None:
    cfg = SimpleNamespace(
        act="SiLU",
        norm=True,
        kernel_size=5,
        minres=4,
        depth=4,
        mults=[1, 1, 1, 1],
    )
    enc = ConvEncoder(cfg, input_shape=(64, 64, 3))
    x = torch.randn(2, 64, 64, 3)
    y = enc(x)
    assert y.ndim == 2
    assert y.shape[0] == 2
    assert y.shape[1] == enc.out_dim
