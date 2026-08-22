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
* Episode ids are monotonically-increasing per-process int32 counters shared
  across all envs; the SliceSampler uses them to avoid splicing across resets.
* Checkpoints are atomic: ``torch.save`` to ``.tmp``, then ``os.replace``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.inference.evaluate import evaluate

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Hyperparameters for the online training loop."""

    steps: int  # total environment steps to train for
    pretrain: int  # agent.update calls to run before collection
    train_ratio: float  # update calls per env step (fractional)
    batch_size: int  # stored here for documentation / config round-trip
    batch_length: int  # stored here for documentation / config round-trip
    action_repeat: int  # inner steps per outer step (owned by SyncVectorEnv)
    eval_every: int  # eval after this many env steps
    eval_episode_num: int  # episodes to collect during each eval pass
    update_log_every: int  # write logger scalars every N env steps
    checkpoint_every: int  # save checkpoint every N env steps
    checkpoint_path: str  # path for the checkpoint file


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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def begin(self, agent: Any, start_step: int = 0) -> None:
        """Run the training loop.

        Args:
            agent:      Dreamer agent with ``get_initial_state``, ``act``,
                        ``update``, ``checkpoint_state``.
            start_step: Resume from this env-step count (crash recovery).
        """
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

        # Monotonically-increasing episode ids (unique across process lifetime)
        episode_ids = np.arange(N, dtype=np.int32)
        _next_ep_id = N

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

        # ---- pretrain ----
        for _ in range(cfg.pretrain):
            if len(self._buffer) >= prefill_min:
                agent.update(self._buffer)

        # Liveness markers: collection runs fast, but once training updates
        # start fps drops sharply, so the first periodic step-line can be minutes
        # away with nothing printed in between.
        log.info(
            "collecting %d-transition prefill, then training to %d env steps (step line every %d steps)",
            prefill_min,
            cfg.steps,
            cfg.update_log_every,
        )
        _training_started = False

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
                episode_ids = np.arange(_next_ep_id, _next_ep_id + N, dtype=np.int32)
                _next_ep_id += N

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
                        episode_ids[i] = _next_ep_id
                        _next_ep_id += 1

            cache_zeros = {k: torch.zeros_like(state[k]) for k in cache_keys}
            obs_np = obs_next_np

            # ---- training updates ----
            if len(self._buffer) >= prefill_min:
                if not _training_started:
                    log.info("prefill complete at step %d; training updates started", env_step)
                    _training_started = True
                train_budget += cfg.train_ratio * N
                while train_budget >= 1.0:
                    metrics = agent.update(self._buffer)
                    train_budget -= 1.0
                    # Prefix the agent's "loss/*" keys with "train/"
                    for k, v in metrics.items():
                        key = f"train/{k}" if not k.startswith("train/") else k
                        self._logger.scalar(key, v)

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
                self._save_checkpoint(agent, env_step)

            self._logger.keepalive(env_step)

        # Final flush
        self._logger.write(env_step, fps=True)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _run_eval(self, agent: Any, env_step: int) -> None:
        """Evaluate the current policy and log the result.

        The rollout itself lives in :func:`dreamer_arm.inference.evaluate.evaluate`
        so the standalone eval entrypoint runs exactly the same code; this
        method only forwards the metrics to the logger.
        """
        result = evaluate(agent, self._eval_envs, self._cfg.eval_episode_num)
        if result.video is not None:
            self._logger.video("eval/video", result.video)
        self._logger.scalars(result.metrics)
        self._logger.write(env_step, fps=False)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, agent: Any, env_step: int) -> None:
        """Atomically save ``{agent: ..., step: env_step}`` to disk."""
        path = Path(self._cfg.checkpoint_path)
        tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.checkpoint_state(), "step": env_step}, tmp)
        tmp.replace(path)
        log.info("checkpoint saved at step %d → %s", env_step, path)
