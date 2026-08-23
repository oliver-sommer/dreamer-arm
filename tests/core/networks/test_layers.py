import torch

from dreamer_arm.core.networks.layers import BlockLinear, Conv2dSamePad, RMSNorm2D, weight_init_


def test_block_linear_preserves_batch_shape() -> None:
    layer = BlockLinear(in_ch=8, out_ch=12, blocks=4)
    weight_init_(layer)
    assert layer(torch.randn(2, 3, 8)).shape == (2, 3, 12)


def test_same_padding_uses_ceil_divided_output() -> None:
    layer = Conv2dSamePad(3, 5, kernel_size=5, stride=2)
    assert layer(torch.randn(2, 3, 9, 10)).shape == (2, 5, 5, 5)


def test_rms_norm_2d_preserves_shape() -> None:
    tensor = torch.randn(2, 4, 5, 6)
    assert RMSNorm2D(4)(tensor).shape == tensor.shape
