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
