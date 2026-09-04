"""Online Dreamer training loop.

``OnlineTrainer.begin`` is the only public entry point.  It runs the standard
DreamerV3 collection → update → eval → checkpoint cycle until ``cfg.steps``
environment steps have been taken.

Design notes
------------
* The trainer counts **post-action-repeat** env steps (one ``SyncVectorEnv.step``
  = 1 agent step = ``action_repeat`` inner physics steps).
* Replay ratio follows the standard Dreamer definition: replayed transition
  samples per newly collected transition. Each update consumes
  ``batch_size * batch_length`` samples, so update-call credit is
  ``replay_ratio / batch_steps`` per environment step.
* Eval reuses the train envs (``eval_envs is train_envs``), so no extra EGL
  contexts are created. Fast eval follows ``eval_every`` and the early warmup
  milestones; the larger fixed-seed pass follows ``robust_eval_every``, writes
  the same ``eval/*`` series, and alone selects ``best.pt``. A shared-env
  evaluation resets the interrupted training
  rollout and allocates fresh episode ids, preventing long synchronized MT
  episodes from delaying and coalescing early evals.
  ``eval_at_start`` additionally seeds this trigger before the loop's first
  iteration, so a run's first eval reads the policy exactly as ``begin``
  received it -- untrained on a fresh run, whatever ``resume`` restored on a
  resumed one -- rather than waiting ``eval_every`` steps for a baseline.  The
  rollout itself lives in :mod:`dreamer_arm.inference.evaluate`.
* Episode ids are monotonically-increasing int32 counters shared across all
  envs; the SliceSampler uses them to avoid splicing across resets.  The counter
  lives on the trainer and is checkpointed, so a run resumed onto a restored
  replay buffer continues past the ids already stored rather than reusing them.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.inference.evaluate import evaluate
from dreamer_arm.training.checkpoint import CheckpointManager
from dreamer_arm.training.metrics import compact_eval_metrics, compact_train_histograms, compact_train_metrics
from dreamer_arm.utils.logging import phase, set_phase

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Hyperparameters for the online training loop."""

    steps: int  # total environment steps to train for
    prefill: int  # transitions to collect before the first update
    pretrain: int  # world-model-only updates once the prefill is collected
    replay_ratio: float  # replayed transition samples per new env transition
    batch_size: int  # stored here for documentation / config round-trip
    batch_length: int  # stored here for documentation / config round-trip
    action_repeat: int  # inner steps per outer step (owned by SyncVectorEnv)
    eval_every: int  # eval after this many env steps
    eval_episode_num: int  # episodes to collect during each eval pass
    robust_eval_every: int  # larger fixed-seed eval after this many env steps
    robust_eval_episode_num: int  # episodes to collect during the larger eval
    update_log_every: int  # write logger scalars every N env steps
    checkpoint_every: int  # refresh latest.pt every N env steps
    checkpoint_dir: str  # directory holding latest.pt / best.pt / checkpoints/
    diagnostic_log_every: int  # expensive model/optimizer diagnostics cadence
    checkpoint_keep_every: int  # also archive on this grid (0 = no archives)
    checkpoint_buffer: bool  # persist the replay buffer beside latest.pt
    eval_at_start: bool  # run one eval pass before the first training step
    eval_warmup_steps: tuple[int, ...]  # additional one-shot eval milestones
    heartbeat_secs: float  # console liveness line cadence (0 disables)


class OnlineTrainer:
    """Runs the online DreamerV3 training loop."""

    def __init__(
        self,
        cfg: TrainerConfig,
        buffer: Any,
        logger: Any,
        train_envs: Any,
        eval_envs: Any | None,
    ) -> None:
        self._cfg = cfg
        self._buffer = buffer
        self._logger = logger
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._N = train_envs.num_envs
        self._checkpoints = CheckpointManager(cfg.checkpoint_dir, buffer, cfg.checkpoint_buffer)
        self._hb_time = time.time()
        # Best eval score seen so far, and the next unused episode id. Both are
        # carried in the checkpoint: see begin() and _save_checkpoint().
        self._best_success = -math.inf
        self._next_ep_id = 0
        self._wm_pretrain_complete = False
        self._episode_metrics: dict[str, list[float]] = defaultdict(list)

    def begin(self, agent: Any, start_step: int = 0, trainer_state: Mapping[str, Any] | None = None) -> None:
        """Run or resume training, including the checkpointed episode-id and best-score state."""
        set_phase("train")
        if trainer_state:
            # Episode ids must continue past anything already in the buffer.
            # Restarting them at N would make SliceSampler treat old and new
            # transitions as one trajectory and splice across the join.
            self._next_ep_id = int(trainer_state.get("next_ep_id", self._next_ep_id))
            if trainer_state.get("best_metric") in {"eval/success_mean", "eval_robust/success_mean"}:
                self._best_success = float(trainer_state.get("best_success", self._best_success))
            self._wm_pretrain_complete = bool(trainer_state.get("wm_pretrain_complete", start_step > 0))
        elif start_step > 0:
            self._wm_pretrain_complete = True
        cfg = self._cfg
        N = self._N
        device = agent.device
        obs_keys = sorted(self._train_envs.observation_space.spaces.keys())

        obs_np = self._train_envs.reset(seed=None)
        state = agent.get_initial_state(N)

        # World-model latent keys the buffer caches and writes back after each
        # update (e.g. RSSM's stoch/deter). Not every key of `state` -- a
        # world model may carry rollout-only context (e.g. DINO-WM's token
        # history window) that must never be written into the replay buffer.
        cache_keys = agent.replay_cache_keys
        # Placeholder zeros for a transition's cached latent before the first
        # update writes a real one back (see ReplayBuffer.update_initial_state).
        # Shape/dtype are fixed by the model and never change across resets or
        # steps, so build these once on CPU -- where the buffer stores them --
        # rather than re-deriving them from the (device-resident) `state` on
        # every step and every eval-boundary reset.
        cache_zeros = {k: torch.zeros(state[k].shape, dtype=state[k].dtype) for k in cache_keys}

        is_first = np.ones(N, dtype=bool)

        # Monotonically-increasing episode ids, allocated from the instance
        # counter so a resumed run continues past the ids already in the buffer.
        episode_ids = np.arange(self._next_ep_id, self._next_ep_id + N, dtype=np.int32)
        self._next_ep_id += N

        ep_return = np.zeros(N, dtype=np.float32)
        ep_len = np.zeros(N, dtype=np.int32)

        batch_steps = cfg.batch_size * cfg.batch_length
        # At least one full training batch, while still guaranteeing every
        # vector slot has a long enough contiguous trajectory to sample.
        prefill_min = max(cfg.prefill, cfg.batch_size * (cfg.batch_length + 1), N * (cfg.batch_length + 1))
        train_budget = 0.0

        env_step = start_step

        # Eval scheduling.  is_first is already all-True above, so seeding
        # eval_pending here fires the deferred-eval check on the very first
        # loop iteration, before any action is taken or transition stored --
        # a baseline read on the (possibly untrained) policy.  Gated the same
        # way as the periodic trigger below, so eval_at_start is a no-op
        # when eval is disabled entirely rather than a surprise first pass.
        eval_enabled = cfg.eval_episode_num > 0 and self._eval_envs is not None
        robust_eval_enabled = cfg.robust_eval_episode_num > 0 and self._eval_envs is not None
        eval_pending = bool(cfg.eval_at_start and eval_enabled)
        robust_eval_pending = False
        _last_eval_trigger = (start_step // cfg.eval_every) * cfg.eval_every
        _last_robust_eval_trigger = (start_step // cfg.robust_eval_every) * cfg.robust_eval_every
        # Normalise here as well as validating the Hydra config because unit
        # tests and library users construct TrainerConfig directly.  bisect
        # semantics are implemented by advancing past every milestone at or
        # below start_step: a resumed run evaluates its restored policy via
        # eval_at_start, but must not replay historical warmup evaluations.
        _eval_warmup_steps = tuple(sorted({int(step) for step in cfg.eval_warmup_steps if int(step) > 0}))
        _next_warmup = 0
        while _next_warmup < len(_eval_warmup_steps) and _eval_warmup_steps[_next_warmup] <= start_step:
            _next_warmup += 1
        _last_ckpt = (start_step // cfg.checkpoint_every) * cfg.checkpoint_every
        _last_log = (start_step // cfg.update_log_every) * cfg.update_log_every
        _last_diagnostic = (
            (start_step // cfg.diagnostic_log_every) * cfg.diagnostic_log_every
            if cfg.diagnostic_log_every > 0
            else start_step
        )
        # Archives ride a coarser grid than latest.pt.  Watermarked, not
        # `env_step % keep_every == 0`: env_step advances by N per iteration and
        # only lands on exact multiples when N divides the interval.
        _keep_every = cfg.checkpoint_keep_every
        _last_archive = (start_step // _keep_every) * _keep_every if _keep_every > 0 else 0

        # Liveness markers: collection runs fast, but once training updates
        # start fps drops sharply, so the first periodic step-line can be minutes
        # away with nothing printed in between -- hence the time-based heartbeat.
        log.info(
            "collecting %d-transition prefill, then training to %d env steps "
            "(step line every %d steps, heartbeat every %.0fs)",
            prefill_min,
            cfg.steps,
            cfg.update_log_every,
            cfg.heartbeat_secs,
        )
        _training_started = False
        updates = 0
        ratio_updates = 0
        ratio_env_steps = 0
        progress_time = time.time()
        progress_updates = 0
        self._hb_time = time.time()

        while env_step < cfg.steps:
            # Run at the first loop boundary after the requested step. Waiting
            # for an episode boundary makes synchronized MT runs coalesce early
            # milestones: with MT50, a 250-policy-step episode spans 12,500
            # global steps. The episode-id seam and is_first below make the
            # interrupted fragment safe for trajectory sampling.
            if eval_pending or robust_eval_pending:
                # A 50-episode pass subsumes a coincident fast pass. Both write
                # the same eval/* series, so running both would waste rollout
                # time and make the smaller estimate immediately disappear.
                if robust_eval_pending:
                    self._run_eval(agent, env_step, robust=True)
                else:
                    self._run_eval(agent, env_step, robust=False)
                eval_pending = False
                robust_eval_pending = False
                if self._eval_envs is self._train_envs:
                    # Evaluation reset the shared envs. Start new training
                    # trajectories rather than splicing them onto the fragments
                    # collected before the evaluation.
                    obs_np = self._train_envs.reset(seed=None)
                    state = agent.get_initial_state(N)
                    is_first = np.ones(N, dtype=bool)
                    episode_ids = np.arange(self._next_ep_id, self._next_ep_id + N, dtype=np.int32)
                    self._next_ep_id += N
                    # The in-flight episodes were abandoned; carrying partial
                    # totals over would inflate the next completed episode.
                    ep_return[:] = 0.0
                    ep_len[:] = 0

            # Build the CPU-side obs tensors once and reuse them for the buffer
            # write below, instead of wrapping the same numpy arrays a second
            # time -- torch.from_numpy is zero-copy, so obs_cpu is cheap, and
            # the buffer store happens before obs_np is reassigned to the next
            # observation, so reusing it here is safe.
            obs_cpu: dict[str, torch.Tensor] = {k: torch.from_numpy(obs_np[k]) for k in obs_keys}
            is_first_cpu = torch.from_numpy(is_first)
            obs_torch: dict[str, torch.Tensor] = {k: v.to(device, non_blocking=True) for k, v in obs_cpu.items()}
            obs_torch["is_first"] = is_first_cpu.to(device, non_blocking=True)

            # No gradient is ever needed here (frozen.py already sets
            # requires_grad=False on every rollout-view param); torch.no_grad()
            # is belt-and-braces, matching inference/evaluate.py's act() call.
            with torch.no_grad():
                action_t, next_state = agent.act(obs_torch, state, eval_mode=False)
            action_np = action_t.detach().cpu().numpy()  # (N, act_dim) -- the one D2H copy of the action

            obs_next_np, rewards, terms, truncs, info = self._train_envs.step(action_np)
            done = terms | truncs  # (N,) bool
            transition_info = info.get("transition", {})

            def transition_array(
                name: str, shape: tuple[int, ...], dtype: Any, source: dict[str, Any] = transition_info
            ) -> np.ndarray:
                value = np.asarray(source.get(name, np.zeros((N, *shape), dtype=dtype)), dtype=dtype)
                expected = (N, *shape)
                if value.shape != expected:
                    raise ValueError(f"transition info {name!r} must have shape {expected}, got {value.shape}")
                return value

            td = TensorDict(
                {
                    **obs_cpu,
                    "action": torch.from_numpy(action_np),  # reuses the D2H copy made above
                    "reward": torch.from_numpy(rewards).unsqueeze(-1),  # (N,1)
                    "success": torch.from_numpy(transition_array("success", (), np.float32)).unsqueeze(-1),
                    "ctrl_valid": torch.from_numpy(transition_array("ctrl_valid", (), bool)).unsqueeze(-1),
                    "ctrl_clamp": torch.from_numpy(transition_array("ctrl_clamp", (3,), np.float32)),
                    "ctrl_retained_xyz": torch.from_numpy(transition_array("ctrl_retained_xyz", (3,), np.float32)),
                    "ctrl_achieved_xyz": torch.from_numpy(transition_array("ctrl_achieved_xyz", (3,), np.float32)),
                    "is_first": is_first_cpu,  # (N,) bool
                    "is_last": torch.from_numpy(done),  # (N,) bool
                    "is_terminal": torch.from_numpy(terms),  # (N,) bool
                    **cache_zeros,
                    "episode": torch.from_numpy(episode_ids).to(torch.int32),
                },
                batch_size=(N,),
            )
            self._buffer.add_transition(td)

            env_step += N
            ep_return += rewards
            ep_len += 1

            if done.any():
                self._log_completed_episodes(done, info["final_info"], ep_return, ep_len)
            for i in range(N):
                if done[i]:
                    ep_return[i] = 0.0
                    ep_len[i] = 0

            is_first = done.copy()
            state = next_state
            # Zero prev_action for restarted envs so the RSSM doesn't condition
            # on the last action of the old episode.
            if done.any():
                done_t = torch.from_numpy(done).to(device)
                state["prev_action"][done_t] = 0.0
                for i in range(N):
                    if done[i]:
                        episode_ids[i] = self._next_ep_id
                        self._next_ep_id += 1

            obs_np = obs_next_np

            if len(self._buffer) >= prefill_min:
                if not _training_started:
                    log.info("prefill complete at step %d; training updates started", env_step)
                    _training_started = True
                    if not self._wm_pretrain_complete:
                        log.info("running %d world-model-only pretrain updates", cfg.pretrain)
                        for _ in range(cfg.pretrain):
                            metrics = agent.update_world_model(self._buffer)
                            updates += 1
                            self._logger.scalars(compact_train_metrics(metrics), defer=True)
                            self._heartbeat(env_step, updates, prefill_min)
                        self._wm_pretrain_complete = True
                ratio_env_steps += N
                train_budget += cfg.replay_ratio * N / batch_steps
                diagnostics_due = (
                    cfg.diagnostic_log_every > 0 and env_step - _last_diagnostic >= cfg.diagnostic_log_every
                )
                while train_budget >= 1.0:
                    metrics = agent.update(self._buffer, diagnostics=diagnostics_due)
                    train_budget -= 1.0
                    updates += 1
                    ratio_updates += 1
                    # defer=True: keep these as tensors until the next
                    # logger.write() instead of syncing every update -- only
                    # the last update's metrics before a write are ever read,
                    # so an eager sync here would pay a device->host stall
                    # once per update just to discard nearly all of them.
                    # See WandbLogger._flush_pending.
                    compact_metrics = compact_train_metrics(metrics, diagnostics=diagnostics_due)
                    self._logger.scalars(
                        compact_metrics,
                        defer=True,
                        latest=frozenset({"train/optimizer/learning_rate"}),
                    )
                    if diagnostics_due:
                        for name, values in compact_train_histograms(metrics).items():
                            self._logger.histogram(name, values)
                        _last_diagnostic = (env_step // cfg.diagnostic_log_every) * cfg.diagnostic_log_every
                        diagnostics_due = False
                    # Beat between updates too: one update can outlast the interval.
                    self._heartbeat(env_step, updates, prefill_min)

            if env_step - _last_log >= cfg.update_log_every:
                _last_log = (env_step // cfg.update_log_every) * cfg.update_log_every
                now = time.time()
                if ratio_env_steps:
                    actual_ratio = ratio_updates * batch_steps / ratio_env_steps
                    self._logger.scalar("train/progress/replay_ratio", actual_ratio)
                self._logger.scalar("train/progress/update_step", updates)
                self._logger.scalar("train/progress/buffer_size", len(self._buffer))
                capacity = int(getattr(self._buffer, "capacity", 0))
                if capacity > 0:
                    self._logger.scalar("train/progress/buffer_fill_fraction", len(self._buffer) / capacity)
                self._logger.scalar("train/progress/wm_pretrain_complete", float(self._wm_pretrain_complete))
                self._logger.scalar(
                    "system/updates_per_sec", (updates - progress_updates) / max(now - progress_time, 1e-8)
                )
                self._flush_episode_metrics()
                self._logger.write(env_step, fps=True)
                progress_time = now
                progress_updates = updates

            if eval_enabled and env_step - _last_eval_trigger >= cfg.eval_every:
                _last_eval_trigger = (env_step // cfg.eval_every) * cfg.eval_every
                eval_pending = True

            if robust_eval_enabled and env_step - _last_robust_eval_trigger >= cfg.robust_eval_every:
                _last_robust_eval_trigger = (env_step // cfg.robust_eval_every) * cfg.robust_eval_every
                robust_eval_pending = True

            # One-shot early-training evaluations.  Advance across every
            # crossed milestone so an uneven vector-env step cannot trigger
            # the same one twice.  A bool is intentional: if a warmup and
            # periodic trigger become due before the same safe boundary, one
            # evaluation represents both instead of running identical passes.
            if eval_enabled:
                while _next_warmup < len(_eval_warmup_steps) and env_step >= _eval_warmup_steps[_next_warmup]:
                    eval_pending = True
                    _next_warmup += 1

            if env_step - _last_ckpt >= cfg.checkpoint_every:
                _last_ckpt = (env_step // cfg.checkpoint_every) * cfg.checkpoint_every
                archive = _keep_every > 0 and env_step - _last_archive >= _keep_every
                if archive:
                    _last_archive = (env_step // _keep_every) * _keep_every
                self._checkpoints.save(agent, env_step, self._trainer_state(), archive=archive)

            self._heartbeat(env_step, updates, prefill_min)
            self._logger.keepalive(env_step)

        self._flush_episode_metrics()
        self._logger.write(env_step, fps=True)

    def _log_completed_episodes(
        self,
        done: np.ndarray,
        final_info: list[dict[str, Any] | None],
        episode_returns: np.ndarray,
        episode_lengths: np.ndarray,
    ) -> None:
        """Log a compact aggregate summary for episodes ending together.

        MT10 environments are synchronized, so ten episodes normally finish
        on one vector step. Writing the same scalar name in a loop makes the
        logger's dict retain only env 9. Per-task success and return live in
        deterministic evaluation; duplicating every controller/reward field
        here was the largest source of W&B metric proliferation.
        """
        values: dict[str, list[float]] = defaultdict(list)
        retained_ctrl_diags = {
            "frac_ws_clamped",
            "frac_lag_clamped",
            "frac_joint_limit_clamped",
            "frac_undertracking",
        }

        def add(name: str, value: float) -> None:
            values[name].append(value)

        for index in np.flatnonzero(done):
            i = int(index)
            fin = final_info[i]
            add("episode/return_mean", float(episode_returns[i]))
            add("episode/length_mean", float(episode_lengths[i]))
            if fin is None:
                continue
            if "success" in fin:
                add("episode/success_rate", float(fin["success"]))
            for key, value in fin.get("ctrl_diag", {}).items():
                if key in retained_ctrl_diags:
                    public_key = {
                        "frac_ws_clamped": "workspace_clamp_rate",
                        "frac_lag_clamped": "lag_clamp_rate",
                        "frac_joint_limit_clamped": "joint_limit_clamp_rate",
                        "frac_undertracking": "undertracking_rate",
                    }[key]
                    add(f"episode/controller_{public_key}", float(value))

        for name, samples in values.items():
            self._episode_metrics[name].extend(samples)

    def _flush_episode_metrics(self) -> None:
        """Emit one episode-weighted mean for every completion since the last write."""

        for name, samples in self._episode_metrics.items():
            if samples:
                self._logger.scalar(name, float(np.mean(samples)))
        self._episode_metrics.clear()

    def _heartbeat(self, env_step: int, updates: int, prefill_min: int) -> None:
        """Print a console liveness line if ``heartbeat_secs`` has elapsed.

        ``update_log_every`` is a *step* cadence, which says nothing about wall
        time: one ``agent.update`` costs milliseconds for the RSSM but seconds
        for DINO-WM on MPS, so a healthy run's first step-line can be 20+
        minutes out.  Console-only -- it never calls ``logger.write``, so it
        adds no W&B steps.
        """
        if self._cfg.heartbeat_secs <= 0:
            return
        now = time.time()
        elapsed = now - self._hb_time
        if elapsed < self._cfg.heartbeat_secs:
            return

        fill = len(self._buffer)
        if fill < prefill_min:
            log.info(
                "prefill %d/%d transitions (%.0f%%)  step %d",
                fill,
                prefill_min,
                100.0 * fill / prefill_min,
                env_step,
            )
        else:
            log.info("working  step %d  updates %d  buffer %d", env_step, updates, fill)

        self._hb_time = now

    def _run_eval(self, agent: Any, env_step: int, *, robust: bool = False) -> None:
        """Evaluate the current policy and log the result.

        The rollout itself lives in :func:`dreamer_arm.inference.evaluate.evaluate`
        so the standalone eval entrypoint runs exactly the same code; this
        method forwards its compact dashboard subset to the logger.
        """
        episodes = self._cfg.robust_eval_episode_num if robust else self._cfg.eval_episode_num
        with phase("eval"):
            result = evaluate(
                agent,
                self._eval_envs,
                episodes,
                capture_artifacts=not robust,
            )
            if result.video is not None:
                self._logger.video("eval/video", result.video)
            if result.action_trace is not None:
                self._logger.table(
                    "eval/action_trace",
                    result.action_trace.columns,
                    result.action_trace.rows,
                )
            if result.task_metrics is not None:
                self._logger.table(
                    "eval/task_metrics",
                    result.task_metrics.columns,
                    result.task_metrics.rows,
                )
            logged_metrics = compact_eval_metrics(result.metrics)
            self._logger.scalars(logged_metrics)
            self._logger.write(env_step, fps=False)

            # best.pt is promoted from the larger eval here, not on the checkpoint
            # cadence, so a peak between two checkpoints is not lost.
            # `evaluate` omits the key entirely when no episode completed.
            success = logged_metrics.get("eval/success_mean") if robust else None
            if success is not None and success > self._best_success:
                self._best_success = float(success)
                self._checkpoints.save_best(agent, env_step, self._trainer_state(), self._best_success)

    def _trainer_state(self) -> dict[str, int | float | bool | str]:
        return {
            "next_ep_id": int(self._next_ep_id),
            "best_success": float(self._best_success),
            "best_metric": "eval/success_mean",
            "wm_pretrain_complete": bool(self._wm_pretrain_complete),
        }
