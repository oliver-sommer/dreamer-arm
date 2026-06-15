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
  multiple of ``eval_every``, then deferred until every env is at an episode
  boundary (``is_first.all()``).
* Episode ids are monotonically-increasing per-process int32 counters shared
  across all envs; the SliceSampler uses them to avoid splicing across resets.
* Checkpoints are atomic: ``torch.save`` to ``.tmp``, then ``os.replace``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

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
        buffer:     Replay buffer (``ReplayBuffer`` from ``data/buffer.py``).
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

        # RSSM latent seed shapes
        stoch_zeros = torch.zeros_like(state["stoch"])
        deter_zeros = torch.zeros_like(state["deter"])

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

        # ====================================================================
        # Main loop
        # ====================================================================
        while env_step < cfg.steps:
            # ---- deferred eval: fire once all envs hit an episode boundary ----
            if eval_pending and is_first.all():
                self._run_eval(agent, obs_keys, device, env_step)
                eval_pending = False
                # Resync: train_envs was used for eval; reset it so collection
                # resumes from clean episode starts.
                obs_np = self._train_envs.reset(seed=None)
                state = agent.get_initial_state(N)
                stoch_zeros = torch.zeros_like(state["stoch"])
                deter_zeros = torch.zeros_like(state["deter"])
                is_first = np.ones(N, dtype=bool)
                # Bump episode ids for all envs after eval
                episode_ids = np.arange(_next_ep_id, _next_ep_id + N, dtype=np.int32)
                _next_ep_id += N

            # ---- act ----
            obs_torch: dict[str, torch.Tensor] = {
                k: torch.from_numpy(obs_np[k]).to(device) for k in obs_keys
            }
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
                    "stoch": stoch_zeros,
                    "deter": deter_zeros,
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

            stoch_zeros = torch.zeros_like(state["stoch"])
            deter_zeros = torch.zeros_like(state["deter"])
            obs_np = obs_next_np

            # ---- training updates ----
            if len(self._buffer) >= prefill_min:
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

    def _run_eval(
        self,
        agent: Any,
        obs_keys: list[str],
        device: Any,
        env_step: int,
    ) -> None:
        """Run ``eval_episode_num`` episodes with ``eval_mode=True``.

        Reuses ``self._eval_envs`` (= train_envs).  Logs per-task success rates
        and the mean success.
        """
        cfg = self._cfg
        N = self._N
        num_rounds = max(1, math.ceil(cfg.eval_episode_num / N))

        # Per-task success tracking {task_name: [successes]}
        task_success: dict[str, list[float]] = {}

        obs_np = self._eval_envs.reset(seed=None)
        state = agent.get_initial_state(N)
        is_first = np.ones(N, dtype=bool)
        ep_return = np.zeros(N, dtype=np.float32)
        completed = np.zeros(N, dtype=np.int32)  # episodes done per env slot

        video_frames: list[np.ndarray] = []
        video_done = False

        while completed.min() < num_rounds:
            if not video_done and "scene" in obs_np:
                video_frames.append(obs_np["scene"][0])

            obs_torch: dict[str, torch.Tensor] = {
                k: torch.from_numpy(obs_np[k]).to(device) for k in obs_keys
            }
            obs_torch["is_first"] = torch.from_numpy(is_first).to(device)

            with torch.no_grad():
                action_t, next_state = agent.act(obs_torch, state, eval_mode=True)
            action_np = action_t.detach().cpu().numpy()

            obs_next_np, rewards, terms, truncs, info = self._eval_envs.step(action_np)
            done = terms | truncs
            ep_return += rewards

            for i in range(N):
                if done[i] and completed[i] < num_rounds:
                    fin = info["final_info"][i]
                    task_name = fin.get("task_name", f"env_{i}") if fin is not None else f"env_{i}"
                    success = float(fin.get("success", 0.0)) if fin is not None else 0.0
                    task_success.setdefault(task_name, []).append(success)
                    completed[i] += 1

            if not video_done and done[0]:
                video_done = True

            is_first = done.copy()
            state = next_state
            if done.any():
                done_t = torch.from_numpy(done).to(device)
                state["prev_action"][done_t] = 0.0

            obs_np = obs_next_np

        if video_frames:
            self._logger.video("eval/video", np.stack(video_frames))

        # Log per-task and mean success
        all_successes: list[float] = []
        for task_name, successes in task_success.items():
            mean_s = float(np.mean(successes))
            safe_name = task_name.replace("-", "_").replace(" ", "_")
            self._logger.scalar(f"eval/success/{safe_name}", mean_s)
            all_successes.extend(successes)

        if all_successes:
            self._logger.scalar("eval/success_mean", float(np.mean(all_successes)))

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
        os.replace(tmp, path)
        log.info("checkpoint saved at step %d → %s", env_step, path)
