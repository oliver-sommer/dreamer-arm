"""Tests for the extracted evaluation rollout (dreamer_arm.inference.evaluate).

The same function backs the training loop's periodic eval and the standalone
entrypoint, so these cover both callers.  Reuses the trainer's mock objects so
no MuJoCo or GPU is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dreamer_arm.core.frozen import freeze_clone
from dreamer_arm.inference.evaluate import (
    ACTION_TRACE_STEPS,
    EVAL_SEED,
    EvalResult,
    _actor_parameter_metrics,
    evaluate,
)
from dreamer_arm.training.trainer import OnlineTrainer
from tests.training.test_trainer import (
    _make_trainer_cfg,
    _MockAgent,
    _MockBuffer,
    _MockLogger,
    _MockVectorEnv,
)


class _SeedRecordingEnv(_MockVectorEnv):
    """Records the seeds it is reset with."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        self.reset_seeds.append(seed)
        return super().reset(seed=seed)


def test_evaluate_returns_success_metrics() -> None:
    envs = _MockVectorEnv(num_envs=2, done_every=3)
    result = evaluate(_MockAgent(num_envs=2), envs, episodes=2)

    assert isinstance(result, EvalResult)
    # The mock env reports task_name="mock" and success=False for every episode.
    assert result.metrics["eval/success/mock"] == 0.0
    assert result.metrics["eval/success_mean"] == 0.0


def test_evaluate_namespaces_every_metric_under_eval() -> None:
    result = evaluate(_MockAgent(num_envs=2), _MockVectorEnv(num_envs=2, done_every=3), episodes=2)
    assert result.metrics
    assert all(name.startswith("eval/") for name in result.metrics)


def test_evaluate_sanitises_task_names() -> None:
    """W&B panel names cannot contain the '-' Meta-World uses in task ids."""

    class _DashedEnv(_MockVectorEnv):
        def step(self, actions):
            obs, rew, terms, truncs, info = super().step(actions)
            info["final_info"] = [{"success": True, "task_name": "door-open v3"} if t else None for t in terms]
            return obs, rew, terms, truncs, info

    result = evaluate(_MockAgent(num_envs=2), _DashedEnv(num_envs=2, done_every=3), episodes=2)
    assert "eval/success/door_open_v3" in result.metrics
    assert result.metrics["eval/success/door_open_v3"] == 1.0


def test_evaluate_uses_a_fixed_seed() -> None:
    """Eval must be comparable across checkpoints, so resets are seeded."""
    envs = _SeedRecordingEnv(num_envs=2, done_every=3)
    evaluate(_MockAgent(num_envs=2), envs, episodes=2)
    assert envs.reset_seeds == [EVAL_SEED]


def test_evaluate_rounds_up_to_whole_rounds() -> None:
    """Episodes are collected in parallel, so each env runs ceil(episodes/N)."""
    envs = _MockVectorEnv(num_envs=4, done_every=2)
    result = evaluate(_MockAgent(num_envs=4), envs, episodes=5)
    # ceil(5/4) = 2 rounds across 4 envs = 8 episodes, all from task "mock".
    assert result.metrics["eval/success/mock"] == 0.0
    assert result.metrics["eval/episodes_completed"] == 8.0
    assert result.metrics["eval/task_count"] == 1.0


def test_evaluate_logs_rewards_and_first_twenty_deterministic_actions() -> None:
    envs = _MockVectorEnv(num_envs=2, done_every=25)
    result = evaluate(_MockAgent(num_envs=2), envs, episodes=2)

    assert result.metrics["eval/return/mock"] == 25.0
    assert result.metrics["eval/return_mean"] == 25.0
    assert not any(key.startswith("eval/action_trace/") for key in result.metrics)
    assert result.action_trace is not None
    assert len(result.action_trace.rows) == ACTION_TRACE_STEPS
    assert result.action_trace.columns[:6] == [
        "task",
        "timestep",
        "action_x",
        "action_y",
        "action_z",
        "action_gripper",
    ]
    assert all(row[0] == "mock" for row in result.action_trace.rows)
    assert all(row[2:6] == [0.0, 0.0, 0.0, 0.0] for row in result.action_trace.rows)


def test_actor_parameter_metrics_prove_live_frozen_sync_and_weight_changes() -> None:
    class _ActorAgent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ac = nn.Module()
            self.ac.actor = nn.Linear(3, 2)
            self.ac._frozen_actor = freeze_clone(self.ac.actor)

    agent = _ActorAgent()
    before = _actor_parameter_metrics(agent)
    with torch.no_grad():
        agent.ac.actor.weight.add_(0.25)
    after = _actor_parameter_metrics(agent)

    assert after["eval/actor_param_checksum"] != before["eval/actor_param_checksum"]
    assert after["eval/actor_param_norm"] != before["eval/actor_param_norm"]
    assert after["eval/actor_live_frozen_max_diff"] == 0.0
    assert after["eval/actor_live_frozen_shared_fraction"] == 1.0


@pytest.mark.parametrize("episodes", [1, 2, 4])
def test_evaluate_always_completes_at_least_one_round(episodes: int) -> None:
    result = evaluate(_MockAgent(num_envs=2), _MockVectorEnv(num_envs=2, done_every=3), episodes)
    assert "eval/success_mean" in result.metrics


def test_evaluate_reports_no_video_without_scene_frames() -> None:
    """The mock env emits an all-zero scene, so frames are captured."""
    result = evaluate(_MockAgent(num_envs=2), _MockVectorEnv(num_envs=2, done_every=3), episodes=2)
    assert result.video is not None
    assert result.video.ndim == 4  # (T, H, W, C)


def test_trainer_forwards_eval_metrics_to_the_logger() -> None:
    """The in-loop eval path must log exactly what evaluate() returns."""
    envs = _MockVectorEnv(num_envs=2, done_every=3)
    logger = _MockLogger()
    trainer = OnlineTrainer(
        _make_trainer_cfg(eval_episode_num=2),
        _MockBuffer(),
        logger,
        envs,
        envs,
    )
    trainer._run_eval(_MockAgent(num_envs=2), env_step=100)

    logged = dict(logger.recorded)
    assert logged["eval/success_mean"] == 0.0
    assert logged["eval/success/mock"] == 0.0
    assert logger.videos == ["eval/video"]
    assert logger.tables == ["eval/action_trace"]
