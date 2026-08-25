from __future__ import annotations

import numpy as np
import pytest

from dreamer_arm.envs.action import ACTION_SPEC, ActionSpec


def test_action_spec_declares_cartesian_rate_and_gripper_layout() -> None:
    space = ACTION_SPEC.make_space()
    assert space.shape == (4,)
    assert space.dtype == np.float32
    np.testing.assert_array_equal(space.low, np.full(4, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(space.high, np.full(4, 1.0, dtype=np.float32))
    assert slice(0, 3) == ActionSpec.CARTESIAN
    assert ActionSpec.GRIPPER == 3
    assert ActionSpec.GRIPPER_OPEN == -1.0
    assert ActionSpec.GRIPPER_CLOSED == 1.0


def test_action_spec_coerces_dtype_and_clips_bounds() -> None:
    action = ACTION_SPEC.coerce([2.0, -2.0, 0.25, 0.5])
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, [1.0, -1.0, 0.25, 0.5])
    assert ACTION_SPEC.make_space().contains(action)


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(3),
        np.zeros((1, 4)),
        np.array([0.0, 0.0, np.nan, 0.0]),
        np.array([0.0, np.inf, 0.0, 0.0]),
    ],
)
def test_action_spec_rejects_invalid_backend_commands(action: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ACTION_SPEC.coerce(action)
