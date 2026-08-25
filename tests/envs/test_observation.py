from __future__ import annotations

import numpy as np
import pytest

from dreamer_arm.envs.observation import ObservationSpec


def test_observation_spec_builds_base_contract() -> None:
    spec = ObservationSpec((32, 48))
    space = spec.make_space()
    assert set(space.spaces) == {"scene", "proprio"}
    assert space["scene"].shape == (32, 48, 3)
    assert space["scene"].dtype == np.uint8
    assert space["proprio"].shape == (10,)
    assert space["proprio"].dtype == np.float32


def test_observation_spec_builds_and_validates_full_observation() -> None:
    spec = ObservationSpec((8, 8), wrist_image=True, task_count=3)
    joints = np.arange(6, dtype=np.float32)
    tool = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    scene = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)

    observation = spec.make(
        scene=scene,
        joint_positions=joints,
        gripper_open=1.5,
        tool_position=tool,
        wrist_image=wrist,
        task_index=2,
    )

    assert spec.make_space().contains(observation)
    np.testing.assert_array_equal(observation["proprio"][spec.JOINTS], joints)
    assert observation["proprio"][spec.GRIPPER_OPEN] == 1.0
    np.testing.assert_array_equal(observation["proprio"][spec.TOOL_POSITION], tool)
    np.testing.assert_array_equal(observation["task_id"], [0.0, 0.0, 1.0])
    assert observation["scene"] is scene
    assert observation["wrist_image"] is wrist


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": (0, 8)},
        {"image_size": (8, 8), "task_count": 0},
    ],
)
def test_observation_spec_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ObservationSpec(**kwargs)  # type: ignore[arg-type]


def test_observation_spec_rejects_missing_or_malformed_measurements() -> None:
    spec = ObservationSpec((8, 8), wrist_image=True, task_count=2)
    base = {
        "scene": np.zeros((8, 8, 3), dtype=np.uint8),
        "joint_positions": np.zeros(6, dtype=np.float32),
        "gripper_open": 0.5,
        "tool_position": np.zeros(3, dtype=np.float32),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "task_index": 0,
    }

    with pytest.raises(ValueError, match="joint_positions"):
        spec.make(**{**base, "joint_positions": np.zeros(5)})
    with pytest.raises(ValueError, match="finite"):
        spec.make(**{**base, "tool_position": np.array([0.0, np.nan, 0.0])})
    with pytest.raises(ValueError, match="uint8"):
        spec.make(**{**base, "scene": np.zeros((8, 8, 3), dtype=np.float32)})
    with pytest.raises(ValueError, match="wrist_image"):
        spec.make(**{**base, "wrist_image": None})
    with pytest.raises(ValueError, match="task_index"):
        spec.make(**{**base, "task_index": 2})
