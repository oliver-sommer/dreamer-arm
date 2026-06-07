"""Gymnasium wrappers shared by every Dreamer environment.

The Dreamer agent expects each observation to be a dict with at least:

* ``image``: ``(H, W, 3)`` uint8 — what the encoder sees.
* one or more proprioceptive float32 vectors (optional, env-specific).
* ``is_first`` / ``is_last`` / ``is_terminal``: bool flags used by the replay
  buffer to mark episode boundaries.

The ``DreamerObsWrapper`` here adds those flags around any inner env that
already produces an image + state dict. The other wrappers in this file are
small, mechanical helpers (action repeat, time limit, synchronous
vectorisation) that we keep in-tree rather than pulling extra deps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

ObsDict = dict[str, np.ndarray]


# --------------------------------------------------------------------- flags


class DreamerObsWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Add ``is_first``/``is_last``/``is_terminal`` bool flags to obs dicts.

    Required because Dreamer's replay buffer keys episode-boundary handling
    off the obs dict itself (so a sampled slice carries its own metadata).
    Assumes the inner env's observation space is already a ``gym.spaces.Dict``
    containing an ``image`` entry.
    """

    def __init__(self, env: gym.Env) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                "DreamerObsWrapper requires a Dict observation space; "
                f"got {type(env.observation_space).__name__}"
            )
        spaces = dict(env.observation_space.spaces)
        for flag in ("is_first", "is_last", "is_terminal"):
            spaces[flag] = gym.spaces.Box(0, 1, shape=(), dtype=np.bool_)  # type: ignore[arg-type]
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs: Any) -> tuple[ObsDict, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._tag(obs, is_first=True, is_last=False, is_terminal=False), info

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._tag(
            obs,
            is_first=False,
            is_last=bool(terminated or truncated),
            is_terminal=bool(terminated),
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def _tag(obs: ObsDict, *, is_first: bool, is_last: bool, is_terminal: bool) -> ObsDict:
        obs = dict(obs)
        obs["is_first"] = np.array(is_first, dtype=np.bool_)
        obs["is_last"] = np.array(is_last, dtype=np.bool_)
        obs["is_terminal"] = np.array(is_terminal, dtype=np.bool_)
        return obs


# ------------------------------------------------------------- action repeat


class ActionRepeat(gym.Wrapper):  # type: ignore[type-arg]
    """Repeat each action ``k`` times, summing reward across the sub-steps.

    Standard Dreamer trick: cuts the effective control rate by ``k`` so the
    world model sees larger between-step changes.
    """

    def __init__(self, env: gym.Env, k: int) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        if k < 1:
            raise ValueError(f"action repeat must be >=1; got {k}")
        self._k = k

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        obs: ObsDict = {}
        for _ in range(self._k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# --------------------------------------------------------------- time limit


class TimeLimit(gym.Wrapper):  # type: ignore[type-arg]
    """Cut episodes off at ``max_steps`` env steps and flag the truncation."""

    def __init__(self, env: gym.Env, max_steps: int) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        self._max_steps = max_steps
        self._step = 0

    def reset(self, **kwargs: Any) -> tuple[ObsDict, dict[str, Any]]:
        self._step = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._max_steps:
            truncated = True
        return obs, float(reward), bool(terminated), bool(truncated), info


# ------------------------------------------------------- action-rate penalty


class ActionRatePenalty(gym.Wrapper):  # type: ignore[type-arg]
    """Subtract a jerk/smoothness penalty from the reward at each step.

    Penalises large changes between consecutive actions (action-rate cost) and,
    optionally, large action magnitudes.  Both are standard sim-to-real tricks
    that encourage the policy to produce smooth, hardware-friendly trajectories
    instead of bang-bang control.

    Penalty applied to the reward::

        penalty = rate_cost * ||a_t - a_{t-1}||^2 + mag_cost * ||a_t||^2

    No penalty is applied on the very first step of an episode (``is_first``).
    When both costs are zero the wrapper is a strict no-op and adds no overhead.

    **Scaling guidance (Meta-World):** task rewards are 0-10 per inner step,
    action-repeat=2 -> 0-20 per wrapped step.  Per-dim ``(da)^2 <= 4`` over the
    4-D action -> ``sum((da)^2) <= 16``.  So ``rate_cost ~= 0.02`` yields at most
    ~0.3 penalty vs ~10-20 task reward -- gentle enough not to swamp the task,
    strong enough to discourage reversals.  Start there and watch
    ``action_rate_cost`` vs success-rate in W&B.

    Diagnostics surfaced in ``info``:
    * ``info["task_reward"]``     — reward before penalty.
    * ``info["action_rate_cost"]``— penalty actually subtracted this step.
    """

    def __init__(self, env: gym.Env, rate_cost: float, mag_cost: float = 0.0) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        self._rate_cost = float(rate_cost)
        self._mag_cost = float(mag_cost)
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[ObsDict, dict[str, Any]]:
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        task_reward = float(reward)

        penalty = 0.0
        if self._rate_cost != 0.0 and self._prev_action is not None:
            penalty += self._rate_cost * float(np.sum((action - self._prev_action) ** 2))
        if self._mag_cost != 0.0:
            penalty += self._mag_cost * float(np.sum(action**2))

        self._prev_action = np.array(action, copy=True)

        info["task_reward"] = task_reward
        info["action_rate_cost"] = penalty
        return obs, task_reward - penalty, terminated, truncated, info


# --------------------------------------------------------------- vectorisation


class SyncVectorEnv:
    """Minimal synchronous vector env over a list of factories.

    We deliberately don't use ``gymnasium.vector.SyncVectorEnv`` because we
    want the *raw* dict obs (one entry per env, stacked on a leading batch
    axis) rather than gymnasium's flattened tuple format. This is the
    interface the Dreamer trainer expects.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]) -> None:  # type: ignore[type-arg]
        if not env_fns:
            raise ValueError("SyncVectorEnv needs at least one env factory")
        self._envs = [fn() for fn in env_fns]
        self.num_envs = len(self._envs)
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    def reset(self, *, seed: int | list[int] | None = None) -> tuple[ObsDict, list[dict[str, Any]]]:
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
        obs_list = []
        infos = []
        for env, s in zip(self._envs, seeds, strict=True):
            obs, info = env.reset(seed=s) if s is not None else env.reset()
            obs_list.append(obs)
            infos.append(info)
        return _stack_obs(obs_list), infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[ObsDict, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if len(actions) != self.num_envs:
            raise ValueError(f"actions has {len(actions)} rows but {self.num_envs} envs")
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []
        for i, (env, act) in enumerate(zip(self._envs, actions, strict=True)):
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                # Auto-reset to keep the replay-buffer-friendly stream contiguous.
                terminal_info = info
                next_obs, info = env.reset()
                info["final_observation"] = obs
                info["final_info"] = terminal_info
                obs = next_obs
                # is_first flag for the new episode is set in DreamerObsWrapper.reset.
            obs_list.append(obs)
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            infos.append(info)
        return _stack_obs(obs_list), rewards, terminated, truncated, infos

    def close(self) -> None:
        for env in self._envs:
            env.close()


def _stack_obs(obs_list: list[ObsDict]) -> ObsDict:
    """Stack a list of dict obs into a single dict with leading batch axis."""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}
