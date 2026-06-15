"""Environment wrappers for the Dreamer training loop.

``TimeLimit``       - truncates an episode after ``time_limit`` steps.
``ActionRatePenalty``- subtracts a jerk / magnitude penalty from reward.
``SyncVectorEnv``   - batches N envs, owns ``action_repeat``, auto-resets,
                      stashes ``final_observation`` / ``final_info``.

``SyncVectorEnv`` is the object that ``make_vector_env`` returns; it is the
only object the trainer and agent interact with.  Its ``step`` drives *one*
agent decision (which internally repeats the wrapped env ``action_repeat``
times and sums the reward).

Auto-reset contract
-------------------
When any env reaches a done (terminated | truncated), ``SyncVectorEnv.step``
saves the terminal obs / info in ``info["final_observation"]`` and
``info["final_info"]``, calls ``env.reset()``, and returns the RESET obs as
the live obs for that env — the trainer's next act call will see ``is_first``.
The done status is latched for the remainder of the ``action_repeat`` sequence
so the env is not stepped after done (Meta-World raises if you do).
"""

from __future__ import annotations

import contextlib
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TimeLimit(gymnasium.Wrapper):  # type: ignore[misc]
    """Truncate an episode after ``time_limit`` steps (counting agent steps)."""

    def __init__(self, env: gymnasium.Env, time_limit: int) -> None:  # type: ignore[misc]
        super().__init__(env)
        self._time_limit = time_limit
        self._step_count = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self._time_limit:
            truncated = True
        return obs, rew, terminated, truncated, info


# ---------------------------------------------------------------------------
# ActionRatePenalty
# ---------------------------------------------------------------------------


class ActionRatePenalty(gymnasium.Wrapper):  # type: ignore[misc]
    """Subtract a sim-to-real jerk / magnitude penalty from reward.

    The penalty is zero when both costs are 0 (no-op for training-only runs).
    """

    def __init__(
        self,
        env: gymnasium.Env,  # type: ignore[misc]
        action_rate_cost: float = 0.0,
        action_mag_cost: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._rate_cost = action_rate_cost
        self._mag_cost = action_mag_cost
        self._prev_action: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        obs, rew, terminated, truncated, info = self.env.step(action)

        penalty = 0.0
        if self._rate_cost > 0.0 and self._prev_action is not None:
            jerk = float(np.sum(np.abs(action - self._prev_action)))
            penalty += self._rate_cost * jerk
        if self._mag_cost > 0.0:
            penalty += self._mag_cost * float(np.sum(action**2))

        self._prev_action = action.copy()
        return obs, float(rew) - penalty, terminated, truncated, info


# ---------------------------------------------------------------------------
# SyncVectorEnv
# ---------------------------------------------------------------------------


class SyncVectorEnv:
    """Synchronous vectorised env with auto-reset and action repeat.

    Args:
        env_fns:       Callables each returning a gymnasium.Env (no args).
        action_repeat: Number of inner ``env.step`` calls per outer step.
                       Rewards are summed; only the final obs is returned.
    """

    def __init__(
        self,
        env_fns: list[Any],
        action_repeat: int = 1,
    ) -> None:
        self._envs: list[gymnasium.Env] = [fn() for fn in env_fns]  # type: ignore[misc]
        self._action_repeat = action_repeat
        self._num_envs = len(self._envs)

        # Infer obs keys from the first env's observation_space
        ref_obs_space = self._envs[0].observation_space
        assert isinstance(ref_obs_space, spaces.Dict)
        self._obs_keys: list[str] = sorted(ref_obs_space.spaces.keys())

        # Expose the single-env observation space (per-env shapes, no leading N).
        # The agent/encoder reads this to determine feature dims; N is the batch dim
        # that appears at runtime in the stacked arrays returned by reset/step.
        self._observation_space = ref_obs_space

        # Action space is per-env (agent acts across all envs, not on batch)
        self._action_space = self._envs[0].action_space

    # ------------------------------------------------------------------
    # Properties expected by Dreamer / __main__
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space  # type: ignore[return-value]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset all envs and return stacked obs dict (N, *shape)."""
        obs_list = []
        for i, env in enumerate(self._envs):
            s = (seed + i) if seed is not None else None
            o, _ = env.reset(seed=s)
            obs_list.append(o)
        return self._stack_obs(obs_list)

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Step all envs with action repeat.

        Args:
            actions: ``(N, act_dim)`` float32 array.

        Returns:
            ``(obs, rewards, terminated, truncated, info)`` where:

            * ``obs``:        dict of ``(N, *shape)`` arrays; done envs have
                              the RESET obs (so the next act sees is_first).
            * ``rewards``:    ``(N,)`` float32, summed over repeats.
            * ``terminated``: ``(N,)`` bool.
            * ``truncated``:  ``(N,)`` bool.
            * ``info``:       ``{"final_observation": dict, "final_info": list}``
                              where ``final_*`` slots are valid only for done envs.
        """
        N = self._num_envs
        rews = np.zeros(N, dtype=np.float32)
        done = np.zeros(N, dtype=bool)
        terms = np.zeros(N, dtype=bool)
        truncs = np.zeros(N, dtype=bool)
        final_obs_per_env: list[dict[str, np.ndarray] | None] = [None] * N
        final_info_per_env: list[dict[str, Any] | None] = [None] * N
        cur_obs: list[dict[str, np.ndarray]] = [{} for _ in range(N)]

        for _ in range(self._action_repeat):
            for i in range(N):
                if done[i]:
                    continue
                o, rew, term, trunc, info = self._envs[i].step(actions[i])
                rews[i] += float(rew)
                cur_obs[i] = o
                if term or trunc:
                    final_obs_per_env[i] = {k: v.copy() for k, v in o.items()}
                    final_info_per_env[i] = {**info}
                    terms[i] = term
                    truncs[i] = trunc
                    done[i] = True

        # Auto-reset done envs — replace their obs with the reset obs
        for i in range(N):
            if done[i]:
                o_reset, _ = self._envs[i].reset()
                cur_obs[i] = o_reset

        obs = self._stack_obs(cur_obs)

        # Build final_observation (zeros for non-done envs)
        final_obs_stacked: dict[str, np.ndarray] = {}
        for k in self._obs_keys:
            parts = []
            for i in range(N):
                if final_obs_per_env[i] is not None:
                    parts.append(final_obs_per_env[i][k])  # type: ignore[index]
                else:
                    parts.append(np.zeros_like(obs[k][i]))
            final_obs_stacked[k] = np.stack(parts)

        info_out: dict[str, Any] = {
            "final_observation": final_obs_stacked,
            "final_info": final_info_per_env,
        }
        return obs, rews, terms, truncs, info_out

    def close(self) -> None:
        for env in self._envs:
            with contextlib.suppress(Exception):
                env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_obs(self, obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs_list]) for k in self._obs_keys}
