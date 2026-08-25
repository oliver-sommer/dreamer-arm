"""Tests for the online training loop (dreamer_arm.training.trainer).

Uses lightweight mock objects so no MuJoCo or GPU is required.
"""

from __future__ import annotations

import itertools
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.training.trainer import OnlineTrainer, TrainerConfig


def _make_trainer_cfg(**overrides: Any) -> TrainerConfig:
    defaults: dict[str, Any] = dict(  # noqa: C408
        steps=10,
        pretrain=0,
        replay_ratio=0.0,  # no replay updates by default in unit tests
        batch_size=2,
        batch_length=4,
        action_repeat=1,
        eval_every=9999,
        eval_episode_num=0,  # disable eval
        update_log_every=9999,
        checkpoint_every=9999,
        # Writable but shared: tests that assert on checkpoint contents pass
        # their own tmp_path.  Only created if a test actually writes.
        checkpoint_dir=str(Path(tempfile.gettempdir()) / "dreamer-arm-test-checkpoints"),
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
        info: dict[str, Any] = {"final_info": fin_info}
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
        self.replay_cache_keys = ("stoch", "deter")

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
    """Captures scalar / video calls.

    ``recorded`` holds the flat (name, value) log; the list is deliberately not
    named ``scalars`` because that would shadow the ``scalars()`` method the
    trainer calls.
    """

    def __init__(self) -> None:
        self.recorded: list[tuple[str, float]] = []
        self.videos: list[str] = []
        self.tables: list[str] = []

    def scalar(self, name: str, value: Any) -> None:
        self.recorded.append((name, float(value)))

    def scalars(self, values: dict[str, Any], defer: bool = False) -> None:
        for k, v in values.items():
            self.scalar(k, v)

    def video(self, name: str, frames: Any) -> None:
        self.videos.append(name)

    def table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        self.tables.append(name)

    def write(self, step: int, fps: bool = False) -> None:
        pass

    def keepalive(self, step: int) -> None:
        pass


def test_transition_episode_ids_monotonic() -> None:
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
        ids_for_env = [int(buffer._transitions[t]["episode"][env_idx]) for t in range(n_transitions)]
        # ids should be non-decreasing
        for prev, curr in itertools.pairwise(ids_for_env):
            assert curr >= prev, f"Episode id went backward: {prev} → {curr}"


def test_is_first_after_done() -> None:
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
                assert bool(td_next["is_first"][i]), f"env {i} step {t}: is_last=True but next is_first=False"


def test_reward_shape() -> None:
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
    N = 2
    # done_every=3: each env finishes after 3 steps; with steps=9 → 3 resets per env
    envs = _MockVectorEnv(num_envs=N, done_every=3)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=9)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    score_logs = [v for name, v in logger.recorded if name == "episode/score"]
    # With 9 steps / 3 steps per episode * 2 envs, expect ~6 episode ends
    assert len(score_logs) >= 1, "Expected at least one episode/score log"


def test_episode_reward_components_logged() -> None:
    class _RewardDiagnosticEnv(_MockVectorEnv):
        def step(self, actions):
            obs, rewards, terms, truncs, info = super().step(actions)
            for fin in info["final_info"]:
                if fin is not None:
                    fin["reward_diag"] = {"grasp_reward": 0.25}
            return obs, rewards, terms, truncs, info

    envs = _RewardDiagnosticEnv(num_envs=2, done_every=2)
    logger = _MockLogger()
    trainer = OnlineTrainer(_make_trainer_cfg(steps=4), _MockBuffer(), logger, envs, eval_envs=None)
    trainer.begin(_MockAgent())

    assert ("episode/reward_grasp_reward", 0.25) in logger.recorded


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    N = 1
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=3, checkpoint_every=2, checkpoint_dir=str(tmp_path))
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    latest = tmp_path / "latest.pt"
    assert latest.exists(), "latest.pt should exist"
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    assert "agent" in ckpt, "Checkpoint missing 'agent' key"
    assert "step" in ckpt, "Checkpoint missing 'step' key"
    assert isinstance(ckpt["step"], int)
    # Trainer-owned counters ride along so a resume can restore them.
    assert ckpt["trainer"]["next_ep_id"] >= N
    assert "best_success" in ckpt["trainer"]
    # No .tmp left behind by the atomic write.
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_ratio_uses_sampled_transition_semantics() -> None:
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
    cfg = _make_trainer_cfg(steps=n_steps * N, replay_ratio=4.0)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    # Each update samples batch_size * batch_length = 8 transitions. With 20
    # collected transitions and replay_ratio=4, the standard Dreamer budget is
    # 20 * 4 / 8 = 10 optimizer updates (not 80 calls).
    expected = 10
    assert len(update_calls) == expected, f"Expected {expected} update calls, got {len(update_calls)}"


def test_no_updates_during_prefill() -> None:
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

    cfg = _make_trainer_cfg(steps=20 * N, replay_ratio=2.0)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    assert len(update_calls) == 0, f"Should not call update during prefill, got {len(update_calls)} calls"


def test_rollout_only_state_not_written_to_buffer() -> None:
    """A world model with rollout-only state (no replay cache) must not leak
    it into the buffer -- only `replay_cache_keys` are ever stored per step.
    """
    N = 2

    class _WindowedAgent(_MockAgent):
        """Mimics DINO-WM: rollout state carries a context window, but
        nothing is cached to the replay buffer (`replay_cache_keys = ()`).
        """

        def __init__(self, num_envs: int = 2, act_dim: int = 4) -> None:
            super().__init__(num_envs, act_dim)
            self.replay_cache_keys = ()

        def get_initial_state(self, batch_size: int) -> TensorDict:
            return TensorDict(
                {
                    "context": torch.zeros(batch_size, 3, 8),
                    "prev_action": torch.zeros(batch_size, self._act_dim),
                },
                batch_size=(batch_size,),
            )

        @torch.no_grad()
        def act(
            self, obs: dict[str, torch.Tensor], state: TensorDict, eval_mode: bool = False
        ) -> tuple[torch.Tensor, TensorDict]:
            n = state.batch_size[0]
            action = torch.zeros(n, self._act_dim)
            next_state = TensorDict(
                {"context": torch.ones(n, 3, 8), "prev_action": action},
                batch_size=(n,),
            )
            return action, next_state

    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _MockBuffer()
    logger = _MockLogger()
    agent = _WindowedAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=10)
    trainer = OnlineTrainer(cfg, buffer, logger, envs, eval_envs=None)
    trainer.begin(agent)

    assert len(buffer._transitions) > 0
    for td in buffer._transitions:
        assert "context" not in td, "rollout-only state key leaked into the replay buffer"


class _FilledBuffer(_MockBuffer):
    """Buffer that reports enough fill to leave prefill immediately."""

    def __len__(self) -> int:
        return 10_000


def test_heartbeat_reports_prefill_progress(caplog: Any) -> None:
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    # heartbeat_secs tiny → fires on every loop iteration.
    cfg = _make_trainer_cfg(steps=4 * N, heartbeat_secs=1e-9)
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    with caplog.at_level("INFO", logger="dreamer_arm.training.trainer"):
        trainer.begin(agent)

    assert any("prefill" in r.message and "transitions" in r.message for r in caplog.records), (
        f"no prefill heartbeat in: {[r.message for r in caplog.records]}"
    )


def test_heartbeat_reports_updates_once_training_starts(caplog: Any) -> None:
    """Once training starts the heartbeat reports the running update count,
    which is what distinguishes a slow world model from a hang."""
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=4 * N, replay_ratio=4.0, heartbeat_secs=1e-9)
    trainer = OnlineTrainer(cfg, _FilledBuffer(), _MockLogger(), envs, eval_envs=None)
    with caplog.at_level("INFO", logger="dreamer_arm.training.trainer"):
        trainer.begin(agent)

    assert any("working" in r.message and "updates" in r.message for r in caplog.records), (
        f"no working heartbeat in: {[r.message for r in caplog.records]}"
    )


def test_heartbeat_disabled_by_zero(caplog: Any) -> None:
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=4 * N, replay_ratio=4.0, heartbeat_secs=0.0)
    trainer = OnlineTrainer(cfg, _FilledBuffer(), _MockLogger(), envs, eval_envs=None)
    with caplog.at_level("INFO", logger="dreamer_arm.training.trainer"):
        trainer.begin(agent)

    # "working" and "N/M transitions" are heartbeat-only phrasings; the one-off
    # "collecting ...-transition prefill" banner is not a heartbeat.
    assert not any("working" in r.message for r in caplog.records)


def test_pretrain_runs_once_after_prefill() -> None:
    """`pretrain` warms the model on the collected prefill.

    It used to run before any collection, where its own buffer-size guard was
    always false -- so the knob silently did nothing.
    """
    N = 2
    update_calls: list[int] = []

    class _CountingAgent(_MockAgent):
        def update(self, buffer: Any) -> dict[str, torch.Tensor]:
            update_calls.append(1)
            return {}

    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _CountingAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=6 * N, replay_ratio=0.0, pretrain=5)
    trainer = OnlineTrainer(cfg, _FilledBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer.begin(agent)

    assert len(update_calls) == 5, f"expected 5 pretrain updates, got {len(update_calls)}"


def test_archives_only_on_keep_every_grid(tmp_path: Path) -> None:
    N = 1
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(
        steps=12,
        checkpoint_every=2,
        checkpoint_keep_every=6,
        checkpoint_dir=str(tmp_path),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer.begin(agent)

    assert (tmp_path / "latest.pt").exists()
    archived = sorted(p.name for p in (tmp_path / "checkpoints").glob("step_*.pt"))
    # checkpoint_every=2 crosses the keep_every=6 watermark at steps 6 and 12.
    assert archived == ["step_000000006.pt", "step_000000012.pt"], archived


def test_no_archives_when_keep_every_zero(tmp_path: Path) -> None:
    N = 1
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(
        steps=12,
        checkpoint_every=2,
        checkpoint_keep_every=0,
        checkpoint_dir=str(tmp_path),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer.begin(agent)

    assert (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_archive_grid_survives_uneven_env_count(tmp_path: Path) -> None:
    """The keep_every grid is a watermark, not step % keep_every == 0.

    env_step advances by N per iteration, so it only lands exactly on a
    multiple of keep_every when N divides the interval.  With N=3 and
    keep_every=10 it never will; the watermark must still fire once per
    crossing rather than never firing (or firing every iteration).
    """
    N = 3
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(
        steps=33,  # env_step visits 3, 6, ..., 33 -- crosses 10 and 20 and 30
        checkpoint_every=3,
        checkpoint_keep_every=10,
        checkpoint_dir=str(tmp_path),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer.begin(agent)

    archived = sorted(p.name for p in (tmp_path / "checkpoints").glob("step_*.pt"))
    assert len(archived) == 3, archived  # one per watermark crossing, not one per checkpoint


class _ScriptedSuccessEnv(_MockVectorEnv):
    """Reports a caller-controlled success value on every episode completion."""

    success: bool = False

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, rewards, terms, truncs, info = super().step(actions)
        for fin in info["final_info"]:
            if fin is not None:
                fin["success"] = self.success
        return obs, rewards, terms, truncs, info


def test_best_checkpoint_written_on_improvement_only(tmp_path: Path) -> None:
    envs = _ScriptedSuccessEnv(num_envs=2, done_every=3)
    cfg = _make_trainer_cfg(eval_episode_num=2, checkpoint_dir=str(tmp_path))
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    agent = _MockAgent(num_envs=2)

    envs.success = True
    trainer._run_eval(agent, env_step=100)
    assert (tmp_path / "best.pt").exists()
    first_mtime = (tmp_path / "best.pt").stat().st_mtime_ns
    assert trainer._best_success == 1.0

    # A worse eval must not overwrite best.pt.
    envs.success = False
    trainer._run_eval(agent, env_step=200)
    assert trainer._best_success == 1.0
    assert (tmp_path / "best.pt").stat().st_mtime_ns == first_mtime


def test_best_checkpoint_survives_missing_success_key(monkeypatch: Any, tmp_path: Path) -> None:
    """evaluate() omits eval/success_mean entirely when no episode completed.

    _run_eval must not crash on the missing key, and must not write best.pt.
    Stubs evaluate() directly rather than driving an env loop, since the real
    evaluate() cannot return early without at least one completed episode.
    """
    from dreamer_arm.inference.evaluate import EvalResult

    envs = _MockVectorEnv(num_envs=2, done_every=3)
    cfg = _make_trainer_cfg(eval_episode_num=2, checkpoint_dir=str(tmp_path))
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    agent = _MockAgent(num_envs=2)

    monkeypatch.setattr(
        "dreamer_arm.training.trainer.evaluate",
        lambda *a, **k: EvalResult(metrics={"eval/ctrl_singularity": 0.1}, video=None),
    )
    trainer._run_eval(agent, env_step=100)

    assert not (tmp_path / "best.pt").exists()
    assert trainer._best_success == -math.inf


def test_trainer_state_round_trips_through_resume(tmp_path: Path) -> None:
    """next_ep_id and best_success survive a checkpoint save/restore cycle.

    Ids must continue past what the first run already allocated -- restarting
    at N would make SliceSampler treat old and new transitions (sharing an id)
    as one trajectory and splice across the join.
    """
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=2)  # forces episode-id churn
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(steps=20, checkpoint_every=20, checkpoint_dir=str(tmp_path))
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer._best_success = 0.75  # simulate an eval having already run
    trainer.begin(agent)

    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    saved_next_id = ckpt["trainer"]["next_ep_id"]
    assert saved_next_id > N  # episodes actually turned over during the run
    assert ckpt["trainer"]["best_success"] == 0.75

    # A fresh trainer (new process, in effect) resuming with that state must
    # not reissue any id the first run already used.
    resumed = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    resumed.begin(agent, start_step=20, trainer_state=ckpt["trainer"])
    assert resumed._next_ep_id >= saved_next_id
    assert resumed._best_success == 0.75


def test_eval_at_start_runs_before_any_collection(monkeypatch: Any) -> None:
    """With eval_at_start, the first eval must happen before any transition
    is stored -- a baseline read on the policy exactly as begin() received it.
    """
    from dreamer_arm.inference.evaluate import EvalResult

    call_order: list[str] = []

    def fake_evaluate(agent: Any, envs: Any, episodes: int) -> Any:
        call_order.append("eval")
        return EvalResult(metrics={"eval/success_mean": 0.0}, video=None)

    monkeypatch.setattr("dreamer_arm.training.trainer.evaluate", fake_evaluate)

    class _TrackingBuffer(_MockBuffer):
        def add_transition(self, data: TensorDict) -> None:
            call_order.append("transition")
            super().add_transition(data)

    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    buffer = _TrackingBuffer()
    agent = _MockAgent(num_envs=N)

    cfg = _make_trainer_cfg(
        steps=4 * N,
        eval_episode_num=2,
        eval_every=9999,  # would not otherwise fire within this short run
        eval_at_start=True,
    )
    trainer = OnlineTrainer(cfg, buffer, _MockLogger(), envs, envs)
    trainer.begin(agent)

    assert call_order[0] == "eval", call_order
    assert "transition" in call_order
    assert call_order.index("eval") < call_order.index("transition")


def test_eval_at_start_false_waits_for_normal_cadence(monkeypatch: Any) -> None:
    from dreamer_arm.inference.evaluate import EvalResult

    eval_calls = 0

    def fake_evaluate(agent: Any, envs: Any, episodes: int) -> Any:
        nonlocal eval_calls
        eval_calls += 1
        return EvalResult(metrics={"eval/success_mean": 0.0}, video=None)

    monkeypatch.setattr("dreamer_arm.training.trainer.evaluate", fake_evaluate)

    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)
    cfg = _make_trainer_cfg(
        steps=4 * N,
        eval_episode_num=2,
        eval_every=9999,  # never crossed within this short run
        eval_at_start=False,
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    trainer.begin(agent)

    assert eval_calls == 0


def test_eval_warmup_steps_do_not_wait_for_episode_boundaries(monkeypatch: Any) -> None:
    """Long synchronized episodes must not delay or coalesce milestones."""
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=2)
    agent = _MockAgent(num_envs=N)
    cfg = _make_trainer_cfg(
        steps=16,
        eval_episode_num=2,
        eval_every=9999,
        eval_at_start=False,
        eval_warmup_steps=(4, 10),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    eval_steps: list[int] = []
    monkeypatch.setattr(trainer, "_run_eval", lambda _agent, env_step: eval_steps.append(env_step))

    trainer.begin(agent)

    # 4 is an episode boundary while 10 is mid-episode. Both run at their
    # requested step rather than coalescing at the boundary at step 12.
    assert eval_steps == [4, 10]


def test_eval_warmup_steps_before_resume_are_not_replayed(monkeypatch: Any) -> None:
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=1)
    agent = _MockAgent(num_envs=N)
    cfg = _make_trainer_cfg(
        steps=18,
        eval_episode_num=2,
        eval_every=9999,
        eval_at_start=False,
        eval_warmup_steps=(4, 10, 12),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    eval_steps: list[int] = []
    monkeypatch.setattr(trainer, "_run_eval", lambda _agent, env_step: eval_steps.append(env_step))

    trainer.begin(agent, start_step=10)

    assert eval_steps == [12]


def test_eval_warmup_and_periodic_trigger_coalesce(monkeypatch: Any) -> None:
    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=1)
    agent = _MockAgent(num_envs=N)
    cfg = _make_trainer_cfg(
        steps=10,
        eval_episode_num=2,
        eval_every=4,
        eval_at_start=False,
        eval_warmup_steps=(4,),
    )
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    eval_steps: list[int] = []
    monkeypatch.setattr(trainer, "_run_eval", lambda _agent, env_step: eval_steps.append(env_step))

    trainer.begin(agent)

    assert eval_steps == [4, 8]


def test_eval_at_start_is_a_noop_when_eval_disabled(monkeypatch: Any) -> None:
    called = False

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("evaluate() should never be called")

    monkeypatch.setattr("dreamer_arm.training.trainer.evaluate", fail_if_called)

    N = 2
    envs = _MockVectorEnv(num_envs=N, done_every=99)
    agent = _MockAgent(num_envs=N)

    # eval_episode_num=0 disables eval outright.
    cfg = _make_trainer_cfg(steps=4 * N, eval_episode_num=0, eval_at_start=True)
    trainer = OnlineTrainer(cfg, _MockBuffer(), _MockLogger(), envs, envs)
    trainer.begin(agent)
    assert not called

    # eval_envs=None disables eval outright, even with eval_episode_num > 0.
    cfg2 = _make_trainer_cfg(steps=4 * N, eval_episode_num=2, eval_at_start=True)
    trainer2 = OnlineTrainer(cfg2, _MockBuffer(), _MockLogger(), envs, eval_envs=None)
    trainer2.begin(agent)
    assert not called
