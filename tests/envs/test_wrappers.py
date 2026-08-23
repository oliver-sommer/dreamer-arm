from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

from dreamer_arm.envs.wrappers import ActionRatePenalty, SyncVectorEnv, TimeLimit


def _make_mock_env() -> Any:
    env = MagicMock(spec=gymnasium.Env)
    env.observation_space = spaces.Dict(
        {
            "scene": spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
    )
    env.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def _reset(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return {
            "scene": np.zeros((8, 8, 3), dtype=np.uint8),
            "proprio": np.zeros(10, dtype=np.float32),
        }, {}

    def _step(action: Any, **kwargs: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        return (
            {
                "scene": np.ones((8, 8, 3), dtype=np.uint8),
                "proprio": np.ones(10, dtype=np.float32),
            },
            1.0,
            False,
            False,
            {},
        )

    env.reset = MagicMock(side_effect=_reset)
    env.step = MagicMock(side_effect=_step)
    env.close = MagicMock()
    return env


def _make_done_env(done_on_step: int = 1) -> Any:
    env = _make_mock_env()
    step_count = 0

    def _reset(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        nonlocal step_count
        step_count = 0
        return {
            "scene": np.zeros((8, 8, 3), np.uint8),
            "proprio": np.zeros(10, np.float32),
        }, {"success": False}

    def _step(action: Any, **kwargs: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        nonlocal step_count
        step_count += 1
        done = step_count >= done_on_step
        obs = {
            "scene": np.zeros((8, 8, 3), np.uint8),
            "proprio": np.zeros(10, np.float32),
        }
        return obs, 1.0, done, False, {"success": done}

    env.reset = MagicMock(side_effect=_reset)
    env.step = MagicMock(side_effect=_step)
    return env


def test_sync_vec_env_reset_shape() -> None:
    count = 3
    vec = SyncVectorEnv([_make_mock_env for _ in range(count)], action_repeat=1)
    obs = vec.reset()
    assert obs["scene"].shape == (count, 8, 8, 3)
    assert obs["proprio"].shape == (count, 10)
    assert vec.observation_space["scene"].shape == (8, 8, 3)
    assert vec.observation_space["proprio"].shape == (10,)
    vec.close()


def test_sync_vec_env_auto_reset() -> None:
    env0 = _make_done_env()
    env1 = _make_mock_env()
    vec = SyncVectorEnv([lambda: env0, lambda: env1], action_repeat=1)
    vec.reset()
    obs, _, terms, truncs, info = vec.step(np.zeros((2, 4), dtype=np.float32))
    assert terms[0] or truncs[0]
    assert not (terms[1] or truncs[1])
    assert obs["proprio"][0].sum() == 0.0
    assert obs["proprio"][1].sum() == 10.0
    assert "final_info" in info
    vec.close()


def test_sync_vec_env_final_info() -> None:
    env0 = _make_done_env()
    env1 = _make_mock_env()
    vec = SyncVectorEnv([lambda: env0, lambda: env1], action_repeat=1)
    vec.reset()
    *_, info = vec.step(np.zeros((2, 4), dtype=np.float32))
    assert info["final_info"][0]["success"]
    assert info["final_info"][1] is None


def test_sync_vec_env_action_repeat() -> None:
    env = _make_mock_env()
    vec = SyncVectorEnv([lambda: env], action_repeat=3)
    vec.reset()
    _, rewards, _, _, _ = vec.step(np.zeros((1, 4), dtype=np.float32))
    assert float(rewards[0]) == pytest.approx(3.0)
    assert env.step.call_count == 3
    vec.close()


def test_sync_vec_env_done_skips_remaining_repeats() -> None:
    env = _make_done_env()
    vec = SyncVectorEnv([lambda: env], action_repeat=5)
    vec.reset()
    _, rewards, terms, _, _ = vec.step(np.zeros((1, 4), dtype=np.float32))
    assert terms[0]
    assert float(rewards[0]) == pytest.approx(1.0)
    assert env.step.call_count == 1
    vec.close()


def test_time_limit_truncates() -> None:
    env = TimeLimit(_make_mock_env(), time_limit=3)
    env.reset()
    for _ in range(2):
        *_, truncated, _ = env.step(np.zeros(4))
        assert not truncated
    *_, truncated, _ = env.step(np.zeros(4))
    assert truncated


def test_action_rate_penalty_zero_cost() -> None:
    env = ActionRatePenalty(_make_mock_env(), action_rate_cost=0.0, action_mag_cost=0.0)
    env.reset()
    _, reward, _, _, _ = env.step(np.ones(4, dtype=np.float32))
    assert float(reward) == pytest.approx(1.0)


def test_action_rate_penalty_nonzero() -> None:
    env = ActionRatePenalty(_make_mock_env(), action_rate_cost=1.0, action_mag_cost=0.0)
    env.reset()
    _, first_reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
    _, second_reward, _, _, _ = env.step(np.ones(4, dtype=np.float32))
    assert float(first_reward) == pytest.approx(1.0)
    assert float(second_reward) < 1.0
