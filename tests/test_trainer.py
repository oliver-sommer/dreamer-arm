"""Tests for the online training loop (dreamer_arm.train.trainer).

Uses lightweight mock objects so no MuJoCo or GPU is required.
"""

from __future__ import annotations

import itertools
import os
import tempfile
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.train.trainer import OnlineTrainer, TrainerConfig

# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


def _make_trainer_cfg(**overrides: Any) -> TrainerConfig:
    defaults: dict[str, Any] = dict(
        steps=10,
        pretrain=0,
        train_ratio=0.0,  # no updates by default in unit tests
        batch_size=2,
        batch_length=4,
        action_repeat=1,
        eval_every=9999,
        eval_episode_num=0,  # disable eval
        update_log_every=9999,
        checkpoint_every=9999,
        checkpoint_path="/tmp/test_checkpoint.pt",
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


class _MockVectorEnv:
    """Minimal SyncVectorEnv-like object for trainer tests."""

    def __init__(
        self,
        num_envs: int = 2,
        obs_keys: list[str] | None = None,
        done_every: int = 5,  # env terminates every N steps
    ) -> None:
        from gymnasium import spaces

        self.num_envs = num_envs
        self._done_every = done_every
        self._step_counts = [0] * num_envs

        # Per-env shapes (no leading N) — matches SyncVectorEnv contract
        obs_space_dict: dict[str, spaces.Space[Any]] = {
            "scene": spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_space_dict)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        N = self.num_envs
        self._step_counts = [0] * N
        return {
            "scene": np.zeros((N, 8, 8, 3), dtype=np.uint8),
            "proprio": np.zeros((N, 4), dtype=np.float32),
        }

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        N = self.num_envs
        for i in range(N):
            self._step_counts[i] += 1

        terms = np.array([self._step_counts[i] >= self._done_every for i in range(N)], dtype=bool)
        truncs = np.zeros(N, dtype=bool)
        rewards = np.ones(N, dtype=np.float32)

        # Auto-reset done envs
        for i in range(N):
            if terms[i]:
                self._step_counts[i] = 0

        obs = {
            "scene": np.zeros((N, 8, 8, 3), dtype=np.uint8),
            "proprio": np.ones((N, 4), dtype=np.float32),
        }
        fin_info = [{"success": False, "task_name": "mock"} if terms[i] else None for i in range(N)]
        info: dict[str, Any] = {
            "final_observation": {k: np.zeros_like(v) for k, v in obs.items()},
            "final_info": fin_info,
        }
        return obs, rewards, terms, truncs, info

    def close(self) -> None:
        pass


class _MockBuffer:
    """Captures add_transition calls."""

    def __init__(self, batch_length: int = 4) -> None:
        self._transitions: list[TensorDict] = []
        self.batch_length = batch_length

    def add_transition(self, data: TensorDict) -> None:
        self._transitions.append(data)

    def sample(self) -> Any:
        raise NotImplementedError("MockBuffer.sample not used in unit tests")

    def __len__(self) -> int:
        return 0  # always too small → no training updates


class _MockAgent:
    """Minimal Dreamer-like agent."""

    def __init__(self, num_envs: int = 2, act_dim: int = 4) -> None:
        self.device = torch.device("cpu")
        self._N = num_envs
        self._act_dim = act_dim
        self._stoch_shape = (4, 4)  # tiny
        self._deter_dim = 8

    def get_initial_state(self, batch_size: int) -> TensorDict:
        return TensorDict(
            {
                "stoch": torch.zeros(batch_size, *self._stoch_shape),
                "deter": torch.zeros(batch_size, self._deter_dim),
                "prev_action": torch.zeros(batch_size, self._act_dim),
            },
            batch_size=(batch_size,),
        )

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        state: TensorDict,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, TensorDict]:
        N = state.batch_size[0]
        action = torch.zeros(N, self._act_dim)
        next_state = TensorDict(
            {
                "stoch": torch.zeros(N, *self._stoch_shape),
                "deter": torch.zeros(N, self._deter_dim),
                "prev_action": action,
            },
            batch_size=(N,),
        )
        return action, next_state

    def update(self, buffer: Any) -> dict[str, torch.Tensor]:
        return {"loss/dyn": torch.tensor(0.5)}

    def checkpoint_state(self) -> dict[str, Any]:
        return {"dummy": "state"}

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        pass


class _MockLogger:
    """Captures scalar calls."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float]] = []

    def scalar(self, name: str, value: Any) -> None:
        self.scalars.append((name, float(value)))

    def scalars(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            self.scalar(k, v)

    def write(self, step: int, fps: bool = False) -> None:
        pass

    def keepalive(self, step: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_transition_episode_ids_monotonic() -> None:
    """Episode ids in the buffer must be strictly increasing when envs reset."""
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=3)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=15)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    episode_ids: list[int] = []
    for td in buffer._transitions:
        ids = td["episode"].numpy().tolist()
        episode_ids.extend(ids)

    # All episode ids should be non-negative
    assert all(e >= 0 for e in episode_ids)

    # Find per-env transitions and check that ids only increase per env slot
    n_transitions = len(buffer._transitions)
    for env_idx in range(N):
        ids_for_env = [
            int(buffer._transitions[t]["episode"][env_idx]) for t in range(n_transitions)
        ]
        # ids should be non-decreasing
        for prev, curr in itertools.pairwise(ids_for_env):
            assert curr >= prev, f"Episode id went backward: {prev} → {curr}"


def test_is_first_after_done() -> None:
    """is_first must be True for the step immediately after a done."""
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=2)  # done every 2 steps
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=12)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    # Find transitions where is_last is True; the NEXT transition for that env
    # should have is_first True.
    n = len(buffer._transitions)
    for t in range(n - 1):
        td = buffer._transitions[t]
        td_next = buffer._transitions[t + 1]
        for i in range(N):
            if bool(td["is_last"][i]):
                assert bool(td_next["is_first"][i]), (
                    f"env {i} step {t}: is_last=True but next is_first=False"
                )


def test_reward_shape() -> None:
    """Reward stored in the buffer must have shape (N, 1)."""
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=5)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    for td in buffer._transitions:
        assert td["reward"].shape == (N, 1), f"reward shape: {td['reward'].shape}"


def test_episode_score_logged() -> None:
    """episode/score is logged each time an episode finishes."""
    N = 2
    # done_every=3: each env finishes after 3 steps; with steps=9 → 3 resets per env
    envs = _MockVectorEnv(num_envs=N, done_every=3)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=9)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    score_logs = [v for name, v in logger.scalars if name == "episode/score"]
    # With 9 steps / 3 steps per episode * 2 envs, expect ~6 episode ends
    assert len(score_logs) >= 1, "Expected at least one episode/score log"


def test_checkpoint_round_trip() -> None:
    """Save a checkpoint and verify the envelope has the expected keys."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        N = 1
        envs = _MockVectorEnv(num_envs=N, done_every=99)
        buffer = _MockBuffer()
        logger = _MockLogger()
        agent = _MockAgent(num_envs=N)

        cfg = _make_trainer_cfg(
            steps=3,
            checkpoint_every=2,
            checkpoint_path=ckpt_path,
        )
        trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
        trainer.begin(agent)

        assert os.path.exists(ckpt_path), "Checkpoint file should exist"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "agent" in ckpt, "Checkpoint missing 'agent' key"
        assert "step" in ckpt, "Checkpoint missing 'step' key"
        assert isinstance(ckpt["step"], int)
    finally:
        os.unlink(ckpt_path)


def test_train_ratio_accumulation() -> None:
    """With train_ratio=1.0 and a full buffer, each env step calls agent.update once."""
    N = 2
    update_calls: list[int] = []

    class _CountingAgent(_MockAgent):
        def update(self, buffer: Any) -> dict[str, torch.Tensor]:
            update_calls.append(1)
            return {"loss/dyn": torch.tensor(0.0)}

    class _FullBuffer(_MockBuffer):
        """Always has enough data to trigger training."""

        def __len__(self) -> int:
            return 9999

    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _FullBuffer()
    logger = _MockLogger()
    agent = _CountingAgent(num_envs=N)

    n_steps = 10
    cfg = _make_trainer_cfg(steps=n_steps * N, train_ratio=1.0)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    # Each outer step covers N env steps; train_ratio=1.0 → N credits per step
    # → N updates per step, n_steps steps → n_steps * N total update calls
    expected = n_steps * N
    assert len(update_calls) == expected, (
        f"Expected {expected} update calls, got {len(update_calls)}"
    )


def test_no_updates_during_prefill() -> None:
    """agent.update should NOT be called while buffer is empty (prefill phase)."""
    N = 2
    update_calls: list[int] = []

    class _CountingAgent(_MockAgent):
        def update(self, buffer: Any) -> dict[str, torch.Tensor]:
            update_calls.append(1)
            return {}

    # Buffer always too small → never triggers training
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _MockBuffer()  # len always 0
    logger = _MockLogger()
    agent = _CountingAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=20 * N, train_ratio=2.0)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    assert len(update_calls) == 0, (
        f"Should not call update during prefill, got {len(update_calls)} calls"
    )
