"""Smoke tests for env factories.

These are marked ``slow`` because MuJoCo + dm_control take noticeable time to
import and compile models on first use. They run only with ``pixi run pytest
-m slow``; the default ``test`` task skips them.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_yam_reach_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("yam:reach", seed=0, time_limit=10)
    obs, _info = env.reset()
    assert set(obs.keys()) >= {"image", "state", "target", "is_first", "is_last", "is_terminal"}
    assert obs["image"].dtype == np.uint8
    assert obs["image"].shape == (64, 64, 3)
    assert obs["is_first"]

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(act)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.slow
def test_yam_pick_place_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("yam:pick_place", seed=0, time_limit=10)
    obs, info = env.reset()
    assert set(obs.keys()) >= {
        "image",
        "state",
        "object",
        "goal",
        "is_first",
        "is_last",
        "is_terminal",
    }
    assert "target" not in obs
    assert obs["object"].shape == (3,)
    assert obs["goal"].shape == (3,)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["is_first"]
    assert info["success"] is False
    assert env.action_space.shape == (7,)

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isfinite(reward) and reward >= 0.0
        if terminated or truncated:
            break
    env.close()


@pytest.mark.slow
def test_yam_pick_place_object_rests_on_floor() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("yam:pick_place", seed=1, time_limit=60)
    env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(40):
        obs, *_ = env.step(zero)
    z = float(obs["object"][2])
    # Object should rest near floor_top (~0.01 m), not tunnel or fly away.
    assert -0.02 < z < 0.06, f"unexpected object z={z}"
    env.close()


@pytest.mark.slow
def test_yam_pick_place_spawn_separation() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("yam:pick_place", seed=2, time_limit=10)
    obs, _ = env.reset()
    sep = float(np.linalg.norm(obs["object"][:2] - obs["goal"][:2]))
    assert sep > 0.08, f"object and goal too close at reset: {sep:.3f} m"
    env.close()


@pytest.mark.slow
def test_dmc_cartpole_swingup_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("dmc:cartpole_swingup", seed=0, time_limit=10)
    obs, _info = env.reset()
    assert "image" in obs
    assert obs["image"].dtype == np.uint8
    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, _terminated, _truncated, _info = env.step(act)
        assert np.isfinite(reward)
    env.close()
