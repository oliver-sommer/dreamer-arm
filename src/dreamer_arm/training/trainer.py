"""Online Dreamer training loop.

``OnlineTrainer.begin`` is the only public entry point.  It runs the standard
DreamerV3 collection → update → eval → checkpoint cycle until ``cfg.steps``
environment steps have been taken.

Design notes
------------
* The trainer counts **post-action-repeat** env steps (one ``SyncVectorEnv.step``
  = 1 agent step = ``action_repeat`` inner physics steps).
* Train ratio is fractional: each env step accrues ``train_ratio`` training
  credits; whole credits are drained via ``agent.update`` calls.
* Eval reuses the train envs (``eval_envs is train_envs``), so no extra EGL
  contexts are created.  Eval is triggered when the step counter crosses a
  multiple of ``eval_every``, then deferred until an env is at an episode
  boundary (``is_first.any()``).  The rollout itself lives in
  :mod:`dreamer_arm.inference.evaluate`.
* Episode ids are monotonically-increasing int32 counters shared across all
  envs; the SliceSampler uses them to avoid splicing across resets.  The counter
  lives on the trainer and is checkpointed, so a run resumed onto a restored
  replay buffer continues past the ids already stored rather than reusing them.
* Checkpoints are atomic: ``torch.save`` to ``.tmp``, then ``os.replace``.  Three
  kinds land under ``checkpoint_dir``:

  - ``latest.pt`` -- refreshed every ``checkpoint_every`` steps, and the only one
    accompanied by a replay-buffer dump (``latest_buffer/``) when
    ``checkpoint_buffer`` is set, so resuming from it starts warm.
  - ``checkpoints/step_<N>.pt`` -- archived on the coarser
    ``checkpoint_keep_every`` grid, giving a history to roll back to.
  - ``best.pt`` -- highest ``eval/success_mean`` so far, written from the eval
    path rather than the checkpoint cadence.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.inference.evaluate import evaluate
from dreamer_arm.utils.logging import phase, set_phase

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Hyperparameters for the online training loop."""

    steps: int  # total environment steps to train for
    pretrain: int  # extra agent.update calls once the prefill is collected
    train_ratio: float  # update calls per env step (fractional)
    batch_size: int  # stored here for documentation / config round-trip
    batch_length: int  # stored here for documentation / config round-trip
    action_repeat: int  # inner steps per outer step (owned by SyncVectorEnv)
    eval_every: int  # eval after this many env steps
    eval_episode_num: int  # episodes to collect during each eval pass
    update_log_every: int  # write logger scalars every N env steps
    checkpoint_every: int  # refresh latest.pt every N env steps
    checkpoint_dir: str  # directory holding latest.pt / best.pt / checkpoints/
    checkpoint_keep_every: int = 0  # also archive on this grid (0 = no archives)
    checkpoint_buffer: bool = False  # persist the replay buffer beside latest.pt
    heartbeat_secs: float = 30.0  # console liveness line cadence (0 disables)


class OnlineTrainer:
    """Runs the online DreamerV3 training loop.

    Args:
        cfg:        Hyper-parameter configuration.
        buffer:     Replay buffer (``ReplayBuffer`` from ``core/buffer.py``).
        logger:     ``WandbLogger`` instance.
        train_envs: ``SyncVectorEnv`` used for both training and eval.
        eval_envs:  Same as ``train_envs`` (or ``None`` to skip eval).
    """

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
        self._hb_time = time.time()
        self._hb_step = 0
        self._hb_updates = 0
        # Best eval score seen so far, and the next unused episode id. Both are
        # carried in the checkpoint: see begin() and _save_checkpoint().
        self._best_success = -math.inf
        self._next_ep_id = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def begin(self, agent: Any, start_step: int = 0, trainer_state: Mapping[str, Any] | None = None) -> None:
        """Run the training loop.

        Args:
            agent:         Dreamer agent with ``get_initial_state``, ``act``,
                           ``update``, ``checkpoint_state``.
            start_step:    Resume from this env-step count (crash recovery).
            trainer_state: The ``"trainer"`` block of a checkpoint, restoring the
                           episode-id counter and best eval score.  Required for
                           correctness when resuming onto a restored replay
                           buffer -- see ``_next_ep_id`` below.
        """
        set_phase("train")
        if trainer_state:
            # Episode ids must continue past anything already in the buffer.
            # Restarting them at N would make SliceSampler treat old and new
            # transitions as one trajectory and splice across the join.
            self._next_ep_id = int(trainer_state.get("next_ep_id", self._next_ep_id))
            self._best_success = float(trainer_state.get("best_success", self._best_success))
        cfg = self._cfg
        N = self._N
        device = agent.device
        obs_keys = sorted(self._train_envs.observation_space.spaces.keys())

        # ---- initial collection state ----
        obs_np = self._train_envs.reset(seed=None)
        state = agent.get_initial_state(N)

        # World-model latent keys the buffer caches and writes back after each
        # update (e.g. RSSM's stoch/deter). Not every key of `state` -- a
        # world model may carry rollout-only context (e.g. DINO-WM's token
        # history window) that must never be written into the replay buffer.
        cache_keys = agent.replay_cache_keys
        cache_zeros = {k: torch.zeros_like(state[k]) for k in cache_keys}

        # is_first flags: first obs after reset is the start of episode
        is_first = np.ones(N, dtype=bool)

        # Monotonically-increasing episode ids, allocated from the instance
        # counter so a resumed run continues past the ids already in the buffer.
        episode_ids = np.arange(self._next_ep_id, self._next_ep_id + N, dtype=np.int32)
        self._next_ep_id += N

        # Per-env running stats
        ep_return = np.zeros(N, dtype=np.float32)
        ep_len = np.zeros(N, dtype=np.int32)

        # Minimum buffer fill before training starts
        prefill_min = N * (cfg.batch_length + 1)
        # Fractional training-step budget (drain via whole update() calls)
        train_budget = 0.0

        # Total (post-repeat) env steps taken
        env_step = start_step

        # Eval scheduling
        eval_pending = False
        _last_eval_trigger = (start_step // cfg.eval_every) * cfg.eval_every
        # Checkpoint / log watermarks
        _last_ckpt = (start_step // cfg.checkpoint_every) * cfg.checkpoint_every
        _last_log = (start_step // cfg.update_log_every) * cfg.update_log_every
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
        self._hb_time = time.time()
        self._hb_step = env_step
        self._hb_updates = 0

        # ====================================================================
        # Main loop
        # ====================================================================
        while env_step < cfg.steps:
            # ---- deferred eval: fire once any env hits an episode boundary ----
            # When all envs are in phase (the common case) they reset together,
            # so .any() == .all() here. But if an env ends out of phase (e.g. an
            # early termination), requiring .all() could stall forever and eval
            # would silently never run again; .any() stays robust to desync.
            if eval_pending and is_first.any():
                self._run_eval(agent, env_step)
                eval_pending = False
                # Resync: train_envs was used for eval; reset it so collection
                # resumes from clean episode starts.
                obs_np = self._train_envs.reset(seed=None)
                state = agent.get_initial_state(N)
                cache_zeros = {k: torch.zeros_like(state[k]) for k in cache_keys}
                is_first = np.ones(N, dtype=bool)
                # Bump episode ids for all envs after eval
                episode_ids = np.arange(self._next_ep_id, self._next_ep_id + N, dtype=np.int32)
                self._next_ep_id += N
                # The in-flight episodes were abandoned mid-way; carrying their
                # partial totals over would inflate the next episode's score.
                ep_return[:] = 0.0
                ep_len[:] = 0

            # ---- act ----
            obs_torch: dict[str, torch.Tensor] = {k: torch.from_numpy(obs_np[k]).to(device) for k in obs_keys}
            obs_torch["is_first"] = torch.from_numpy(is_first).to(device)

            action_t, next_state = agent.act(obs_torch, state, eval_mode=False)
            action_np = action_t.detach().cpu().numpy()  # (N, act_dim)

            # ---- step ----
            obs_next_np, rewards, terms, truncs, info = self._train_envs.step(action_np)
            done = terms | truncs  # (N,) bool

            # ---- store transition ----
            obs_td: dict[str, torch.Tensor] = {k: torch.from_numpy(obs_np[k]) for k in obs_keys}
            td = TensorDict(
                {
                    **obs_td,
                    "action": action_t.detach().cpu(),
                    "reward": torch.from_numpy(rewards).unsqueeze(-1),  # (N,1)
                    "is_first": torch.from_numpy(is_first),  # (N,) bool
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

            # ---- episode logging ----
            for i in range(N):
                if done[i]:
                    self._logger.scalar("episode/score", float(ep_return[i]))
                    self._logger.scalar("episode/length", int(ep_len[i]))
                    # sticky success from final_info
                    fin = info["final_info"][i]
                    if fin is not None and "success" in fin:
                        self._logger.scalar("episode/success", float(fin["success"]))
                    # EEController episode diagnostics (YAM only): singularity,
                    # orientation fighting, joint-velocity saturation, and the
                    # achieved-vs-commanded TCP ratio (the stuck signal).
                    if fin is not None and "ctrl_diag" in fin:
                        for k, v in fin["ctrl_diag"].items():
                            self._logger.scalar(f"episode/ctrl_{k}", float(v))
                    ep_return[i] = 0.0
                    ep_len[i] = 0

            # ---- bookkeeping ----
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

            cache_zeros = {k: torch.zeros_like(state[k]) for k in cache_keys}
            obs_np = obs_next_np

            # ---- training updates ----
            if len(self._buffer) >= prefill_min:
                if not _training_started:
                    log.info("prefill complete at step %d; training updates started", env_step)
                    _training_started = True
                    # Warm the model on the prefill before collection continues
                    # at the online train_ratio.
                    for _ in range(cfg.pretrain):
                        agent.update(self._buffer)
                        updates += 1
                        self._heartbeat(env_step, updates, prefill_min)
                train_budget += cfg.train_ratio * N
                while train_budget >= 1.0:
                    metrics = agent.update(self._buffer)
                    train_budget -= 1.0
                    updates += 1
                    # Prefix the agent's "loss/*" keys with "train/"
                    for k, v in metrics.items():
                        key = f"train/{k}" if not k.startswith("train/") else k
                        self._logger.scalar(key, v)
                    # Beat between updates too: one update can outlast the interval.
                    self._heartbeat(env_step, updates, prefill_min)

            # ---- periodic logging ----
            if env_step - _last_log >= cfg.update_log_every:
                _last_log = (env_step // cfg.update_log_every) * cfg.update_log_every
                self._logger.write(env_step, fps=True)

            # ---- eval trigger ----
            if (
                cfg.eval_episode_num > 0
                and self._eval_envs is not None
                and env_step - _last_eval_trigger >= cfg.eval_every
            ):
                _last_eval_trigger = (env_step // cfg.eval_every) * cfg.eval_every
                eval_pending = True

            # ---- checkpoint ----
            if env_step - _last_ckpt >= cfg.checkpoint_every:
                _last_ckpt = (env_step // cfg.checkpoint_every) * cfg.checkpoint_every
                archive = _keep_every > 0 and env_step - _last_archive >= _keep_every
                if archive:
                    _last_archive = (env_step // _keep_every) * _keep_every
                self._save_checkpoint(agent, env_step, archive=archive)

            self._heartbeat(env_step, updates, prefill_min)
            self._logger.keepalive(env_step)

        # Final flush
        self._logger.write(env_step, fps=True)

    # ------------------------------------------------------------------
    # Progress heartbeat
    # ------------------------------------------------------------------

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
            done = updates - self._hb_updates
            # "0 updates" is itself the signal: one update outlasting the interval.
            pace = f"{elapsed / done:.1f}s/update" if done else f"0 updates in {elapsed:.0f}s"
            log.info(
                "working  step %d  updates %d (%s)  %.1f env steps/s  buffer %d",
                env_step,
                updates,
                pace,
                (env_step - self._hb_step) / elapsed,
                fill,
            )

        self._hb_time, self._hb_step, self._hb_updates = now, env_step, updates

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _run_eval(self, agent: Any, env_step: int) -> None:
        """Evaluate the current policy and log the result.

        The rollout itself lives in :func:`dreamer_arm.inference.evaluate.evaluate`
        so the standalone eval entrypoint runs exactly the same code; this
        method only forwards the metrics to the logger.
        """
        with phase("eval"):
            result = evaluate(agent, self._eval_envs, self._cfg.eval_episode_num)
            if result.video is not None:
                self._logger.video("eval/video", result.video)
            self._logger.scalars(result.metrics)
            self._logger.write(env_step, fps=False)

            # best.pt is promoted here, not on the checkpoint cadence: eval is
            # the only success signal and it runs more often than checkpointing,
            # so a peak between two checkpoints would otherwise be lost.
            # `evaluate` omits the key entirely when no episode completed.
            success = result.metrics.get("eval/success_mean")
            if success is not None and success > self._best_success:
                self._best_success = float(success)
                self._save_best(agent, env_step, self._best_success)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _payload(self, agent: Any, env_step: int) -> dict[str, Any]:
        """Everything needed to resume: agent state plus trainer-owned counters."""
        return {
            "agent": agent.checkpoint_state(),
            "step": env_step,
            "trainer": {
                "next_ep_id": int(self._next_ep_id),
                "best_success": float(self._best_success),
            },
        }

    def _write(self, payload: dict[str, Any], path: Path) -> None:
        """Write ``payload`` to ``path`` atomically (``.tmp`` then ``os.replace``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def _save_checkpoint(self, agent: Any, env_step: int, archive: bool = False) -> None:
        """Refresh ``latest.pt``, optionally archiving a copy under ``checkpoints/``.

        The archive is a byte copy of the file just written rather than a second
        ``torch.save``, so the two cannot drift.
        """
        root = Path(self._cfg.checkpoint_dir)
        latest = root / "latest.pt"
        self._write(self._payload(agent, env_step), latest)
        log.info("checkpoint saved at step %d → %s", env_step, latest)

        if archive:
            kept = root / "checkpoints" / f"step_{env_step:09d}.pt"
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest, kept)
            log.info("archived checkpoint → %s", kept)

        if self._cfg.checkpoint_buffer:
            self._save_buffer(latest)

    def _save_buffer(self, latest: Path) -> None:
        """Persist the replay buffer beside ``latest.pt`` so resume starts warm.

        Named by convention (``<stem>_buffer``) so the resume path can find it
        without extra config, and so ``best.pt`` / archives -- which carry no
        buffer -- simply find nothing and start cold.

        Best-effort: a failed buffer dump must not lose the agent checkpoint that
        was just written successfully.
        """
        target = latest.with_name(f"{latest.stem}_buffer")
        try:
            self._buffer.save(target)
        except Exception:  # noqa: BLE001 - never let this kill a training run
            log.exception("replay buffer save failed (%s); continuing without it", target)

    def _save_best(self, agent: Any, env_step: int, success: float) -> None:
        """Record a new best-scoring policy (weights only, no replay buffer)."""
        path = Path(self._cfg.checkpoint_dir) / "best.pt"
        self._write(self._payload(agent, env_step), path)
        log.info("new best eval/success_mean %.4f at step %d → %s", success, env_step, path)
