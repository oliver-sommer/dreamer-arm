"""Online Dreamer trainer: interleaves env steps with world-model updates.

The training loop runs N parallel envs through a :class:`SyncVectorEnv`,
pushes each transition into the trajectory replay buffer, and every
``batch_steps / train_ratio`` env steps takes ``update_num`` optimiser
steps on the Dreamer agent. Eval rolls a fixed number of episodes with the
deterministic actor mode and logs rendered video + return.

This is the W&B-only, single-process port of the reference repo's
``r2dreamer/trainer.py`` — IsaacLab and TensorBoard branches are dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from tensordict import TensorDict

from dreamer_arm.data.buffer import ReplayBuffer
from dreamer_arm.envs.wrappers import SyncVectorEnv
from dreamer_arm.train.logger import WandbLogger
from dreamer_arm.utils.modules import Every, Once


@dataclass
class TrainerConfig:
    """Hyperparameters that drive the online training loop."""

    steps: int
    """Total environment steps to run (counts in env-side steps, not grad steps)."""

    pretrain: int
    """Number of model updates to do once before the loop starts."""

    train_ratio: float
    """env-data steps / model-update steps; e.g. 32 means 1 update per 32 env steps."""

    batch_size: int
    batch_length: int
    action_repeat: int

    eval_every: int
    eval_episode_num: int
    update_log_every: int


class _AgentProto(Protocol):
    """Minimal interface the trainer needs from :class:`Dreamer`."""

    device: torch.device
    act_dim: int

    def get_initial_state(self, batch_size: int) -> TensorDict: ...

    def act(
        self,
        obs: Mapping[str, torch.Tensor],
        state: TensorDict,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, TensorDict]: ...

    def update(self, replay_buffer: ReplayBuffer) -> dict[str, torch.Tensor]: ...

    def train(self) -> Any: ...
    def eval(self) -> Any: ...


class OnlineTrainer:
    """Online RL loop: env rollout → buffer → grad step.

    ``train_envs`` and ``eval_envs`` are :class:`SyncVectorEnv` instances —
    one for collection, optionally one for periodic evaluation.
    """

    def __init__(
        self,
        config: TrainerConfig,
        replay_buffer: ReplayBuffer,
        logger: WandbLogger,
        train_envs: SyncVectorEnv,
        eval_envs: SyncVectorEnv | None = None,
    ) -> None:
        self.config = config
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs

        # train_ratio is in data-steps (env_steps * action_repeat), not env steps.
        batch_steps = config.batch_size * config.batch_length
        self._updates_needed = Every(int(batch_steps / config.train_ratio * config.action_repeat))
        self._should_pretrain = Once()
        self._should_log = Every(config.update_log_every)
        self._should_eval = Every(config.eval_every)

    # ------------------------------------------------------------------- train

    def begin(self, agent: _AgentProto) -> None:
        """Main loop: collect one env step, maybe update, log, repeat."""
        device = agent.device
        envs = self.train_envs
        n = envs.num_envs

        step = len(self.replay_buffer) * self.config.action_repeat
        update_count = 0

        agent_state = agent.get_initial_state(n)
        act = agent_state["prev_action"].clone()

        # First reset — gives us is_first=True everywhere.
        obs_np, _ = envs.reset(seed=0)
        returns = np.zeros(n, dtype=np.float32)
        lengths = np.zeros(n, dtype=np.int32)
        video_cache: list[np.ndarray] = []
        train_metrics: dict[str, torch.Tensor] = {}
        metric_sums: dict[str, float] = {}
        window_updates: int = 0

        # Episode IDs are kept constant across resets so SliceSampler can group
        # multi-episode windows; the RSSM resets internally on is_first.
        episode_ids = torch.arange(n, dtype=torch.int32, device=device)

        while step < self.config.steps:
            # ---- eval ----
            if (
                self._should_eval(step)
                and self.config.eval_episode_num > 0
                and self.eval_envs is not None
            ):
                self.eval(agent, step)

            # ---- env step ----
            obs_t = self._obs_to_tensor(obs_np, device)
            act, agent_state = agent.act(obs_t, agent_state, eval_mode=False)
            act_np = act.detach().cpu().numpy()

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(act_np)
            done_np = terminated_np | truncated_np

            # ---- record + push to buffer ----
            reward_t = torch.from_numpy(reward_np.astype(np.float32)).to(device)
            transition = TensorDict(
                {
                    **{k: torch.from_numpy(np.asarray(v)).to(device) for k, v in obs_np.items()},
                    "action": act.detach() * (~torch.from_numpy(done_np).to(device)).unsqueeze(-1),
                    "reward": reward_t.unsqueeze(-1),
                    "stoch": agent_state["stoch"],
                    "deter": agent_state["deter"],
                    "episode": episode_ids,
                },
                batch_size=(n,),
            )
            if "scene" in obs_np:
                video_cache.append(obs_np["scene"][0])
            self.replay_buffer.add_transition(transition.detach())

            # ---- bookkeeping ----
            returns += reward_np
            lengths += 1
            step += int(n) * self.config.action_repeat

            # ---- episode-end logging ----
            for i, d in enumerate(done_np):
                if not d:
                    continue
                if i == 0 and video_cache:
                    self.logger.video("train/scene_video", np.stack(video_cache, axis=0)[None])
                    video_cache = []
                self.logger.scalar("episode/score", float(returns[i]))
                self.logger.scalar("episode/length", float(lengths[i]))
                fin = infos[i].get("final_info", {})
                success = float(bool(fin.get("success", False)))
                self.logger.scalar("episode/success", success)
                # Per-task breakdown for multi-task (MT*) runs; the env tags
                # each episode with its task name via info["task"].
                task = fin.get("task")
                if task is not None:
                    self.logger.scalar(f"episode/success/{task}", success)
                    self.logger.scalar(f"episode/score/{task}", float(returns[i]))
                self.logger.write(step + i)
                returns[i] = 0.0
                lengths[i] = 0

            obs_np = next_obs_np

            # ---- model updates ----
            min_buffer = (self.config.batch_length + 1) * n
            if len(self.replay_buffer) > min_buffer:
                update_num = (
                    self.config.pretrain if self._should_pretrain() else self._updates_needed(step)
                )
                for _ in range(update_num):
                    train_metrics = agent.update(self.replay_buffer)
                    for name, value in train_metrics.items():
                        v = (
                            value.detach().cpu().item()
                            if isinstance(value, torch.Tensor)
                            else float(value)
                        )
                        metric_sums[name] = metric_sums.get(name, 0.0) + v
                update_count += update_num
                window_updates += update_num

                if window_updates > 0 and self._should_log(step):
                    for name, total in metric_sums.items():
                        self.logger.scalar(f"train/{name}", total / window_updates)
                    self.logger.scalar("train/opt/updates", float(update_count))
                    self.logger.write(step, fps=True)
                    metric_sums.clear()
                    window_updates = 0

    # -------------------------------------------------------------------- eval

    @torch.no_grad()
    def eval(self, agent: _AgentProto, train_step: int) -> None:
        """Roll ``eval_episode_num`` episodes with the deterministic actor."""
        if self.eval_envs is None:
            return
        envs = self.eval_envs
        n = envs.num_envs
        device = agent.device
        agent.eval()

        obs_np, _ = envs.reset(seed=12345)
        agent_state = agent.get_initial_state(n)
        returns = np.zeros(n, dtype=np.float32)
        steps = np.zeros(n, dtype=np.int32)
        successes = np.zeros(n, dtype=bool)
        done_once = np.zeros(n, dtype=bool)
        eval_tasks: list[str | None] = [None] * n

        # Collect one episode's worth of frames from up to eval_episode_num envs.
        # In MT runs each env is pinned to a different task, so this shows one
        # video stream per task side-by-side in W&B.  The length is determined
        # by env 0's episode so all streams are the same number of frames.
        n_video = min(n, self.config.eval_episode_num)
        videos: list[list[np.ndarray]] = [[] for _ in range(n_video)]
        wrist_videos: list[list[np.ndarray]] = [[] for _ in range(n_video)]

        while not done_once.all():
            # Record before stepping to capture the current obs, not post-reset.
            # Stop appending once env 0 finishes so all streams stay the same length.
            if not done_once[0]:
                for vi in range(n_video):
                    if "scene" in obs_np:
                        videos[vi].append(obs_np["scene"][vi])
                    if "wrist_image" in obs_np:
                        wrist_videos[vi].append(obs_np["wrist_image"][vi])
            obs_t = self._obs_to_tensor(obs_np, device)
            act, agent_state = agent.act(obs_t, agent_state, eval_mode=True)
            obs_np, reward_np, terminated_np, truncated_np, eval_infos = envs.step(
                act.detach().cpu().numpy()
            )
            done_np = terminated_np | truncated_np
            active = ~done_once
            returns += reward_np * active
            steps += active.astype(np.int32)
            for i, d in enumerate(done_np):
                if d and not done_once[i]:
                    fin = eval_infos[i].get("final_info", {})
                    successes[i] = bool(fin.get("success", False))
                    eval_tasks[i] = fin.get("task")
            done_once |= done_np

        self.logger.scalar("episode/eval_score", float(returns.mean()))
        self.logger.scalar("episode/eval_length", float(steps.mean()))
        self.logger.scalar("episode/eval_success", float(successes.mean()))
        # Per-task eval success for multi-task (MT*) runs.
        for task in {t for t in eval_tasks if t is not None}:
            mask = np.array([t == task for t in eval_tasks], dtype=bool)
            self.logger.scalar(f"episode/eval_success/{task}", float(successes[mask].mean()))
        # Stack per-env frame lists → (n_video, T, H, W, C); logger tiles horizontally.
        if videos[0]:
            self.logger.video(
                "eval/scene_video",
                np.stack([np.stack(v, axis=0) for v in videos], axis=0),
            )
        if wrist_videos[0]:
            self.logger.video(
                "eval/wrist_video",
                np.stack([np.stack(v, axis=0) for v in wrist_videos], axis=0),
            )
        self.logger.write(train_step)
        agent.train()

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _obs_to_tensor(
        obs: Mapping[str, np.ndarray], device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in obs.items()}
