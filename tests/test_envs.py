"""Smoke tests for the Meta-World env factory.

These are marked ``slow`` because MuJoCo + Meta-World take noticeable time to
import and compile models on first use. They run only with ``pixi run pytest
-m slow``; the default ``test`` task skips them.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Single-task Meta-World (MT1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metaworld_single_task_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:door-open", seed=0, time_limit=10)
    obs, info = env.reset()
    assert set(obs.keys()) >= {"image", "state", "is_first", "is_last", "is_terminal"}
    assert "task_id" not in obs  # single-task: no one-hot
    assert obs["image"].dtype == np.uint8
    assert obs["image"].shape == (64, 64, 3)
    assert obs["state"].shape == (39,)  # Meta-World's fixed-size obs
    assert obs["is_first"]
    assert info.get("task") == "door-open"
    # Sawyer EE action: (x, y, z, gripper)
    assert env.action_space.shape == (4,)

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


# ---------------------------------------------------------------------------
# Multi-task one-hot conditioning
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metaworld_task_id_one_hot() -> None:
    """A pinned task index emits a one-hot ``task_id`` of length num_tasks."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:door-open", seed=0, time_limit=5, task_idx=3, num_tasks=10)
    obs, _ = env.reset()
    assert "task_id" in obs
    assert obs["task_id"].shape == (10,)
    assert int(np.argmax(obs["task_id"])) == 3
    assert obs["task_id"].sum() == pytest.approx(1.0)
    # task_id is carried through step unchanged (task is pinned per env).
    obs, *_ = env.step(env.action_space.sample())
    assert int(np.argmax(obs["task_id"])) == 3
    env.close()


@pytest.mark.slow
def test_metaworld_mt_env_num_guard() -> None:
    """env_num not a multiple of the task count raises a clear error."""
    from dreamer_arm.envs.factory import make_vector_env

    with pytest.raises(ValueError, match="multiple"):
        make_vector_env("metaworld:MT10", num_envs=7)
