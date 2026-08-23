import numpy as np
import pytest
import torch

from dreamer_arm.utils.tensor import compute_global_norm, compute_rms, rpad, tensorstats, to_f32, to_i32, to_np


def test_tensor_conversion_helpers() -> None:
    tensor = torch.tensor([1, 2], dtype=torch.int64)
    assert to_f32(tensor).dtype == torch.float32
    assert to_i32(tensor).dtype == torch.int32
    np.testing.assert_array_equal(to_np(tensor), np.array([1, 2]))


def test_padding_and_norm_helpers() -> None:
    tensor = torch.tensor([3.0, 4.0])
    assert rpad(tensor, 2).shape == (2, 1, 1)
    assert compute_global_norm([tensor, None]).item() == 5.0
    assert compute_rms([tensor]).item() == pytest.approx(5.0 / np.sqrt(2))


def test_tensorstats_names() -> None:
    stats = tensorstats(torch.tensor([1.0, 2.0]), "value")
    assert set(stats) == {"value_mean", "value_std", "value_min", "value_max"}
