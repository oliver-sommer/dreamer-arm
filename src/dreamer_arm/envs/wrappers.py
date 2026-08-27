"""Environment wrappers for the Dreamer training loop.

``TimeLimit``       - truncates an episode after ``time_limit`` steps.
``ActionRatePenalty``- subtracts a jerk / magnitude penalty from reward.
``SyncVectorEnv``   - batches N envs, owns ``action_repeat``, auto-resets,
                      stashes ``final_info``.

``SyncVectorEnv`` is the object that ``make_vector_env`` returns; it is the
only object the trainer and agent interact with.  Its ``step`` drives *one*
agent decision (which internally repeats the wrapped env ``action_repeat``
times and sums the reward).

Auto-reset contract
-------------------
When any env reaches a done (terminated | truncated), ``SyncVectorEnv.step``
saves the terminal info in ``info["final_info"]``, calls ``env.reset()``, and
returns the RESET obs as the live obs for that env — the trainer's next act
call will see ``is_first``. The done status is latched for the remainder of
the ``action_repeat`` sequence so the env is not stepped after done
(Meta-World raises if you do).
"""

from __future__ import annotations

import contextlib
from typing import Any, SupportsFloat, cast

import gymnasium
import numpy as np
from gymnasium import spaces


class TimeLimit(gymnasium.Wrapper):  # type: ignore[misc]
    """Truncate an episode after ``time_limit`` *agent* steps.

    ``SyncVectorEnv`` calls this wrapper's ``step`` ``action_repeat`` times per
    agent step, so the truncation threshold is ``time_limit * action_repeat``
    inner ``env.step`` calls.  With ``action_repeat=1`` this reduces to
    ``time_limit`` steps.  (Previously this counted inner steps directly, which
    silently truncated episodes at ``time_limit / action_repeat`` agent steps.)
    """

    def __init__(self, env: gymnasium.Env, time_limit: int, action_repeat: int = 1) -> None:  # type: ignore[misc]
        super().__init__(env)
        self._max_inner = time_limit * action_repeat
        self._step_count = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any, *, render: bool = True) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # self.env is typed as the generic gymnasium.Env by the Wrapper base
        # class, whose step() takes no render kwarg; in this stack it is
        # always a MetaWorldEnv or another render-aware wrapper.
        obs, rew, terminated, truncated, info = self.env.step(action, render=render)  # ty: ignore[unknown-argument]
        self._step_count += 1
        if self._step_count >= self._max_inner:
            truncated = True
        return obs, rew, terminated, truncated, info


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

    def step(self, action: np.ndarray, *, render: bool = True) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        obs, rew, terminated, truncated, info = self.env.step(action, render=render)  # ty: ignore[unknown-argument]

        penalty = 0.0
        if self._rate_cost > 0.0 and self._prev_action is not None:
            jerk = float(np.sum(np.abs(action - self._prev_action)))
            penalty += self._rate_cost * jerk
        if self._mag_cost > 0.0:
            penalty += self._mag_cost * float(np.sum(action**2))

        self._prev_action = action.copy()
        return obs, float(rew) - penalty, terminated, truncated, info


class SyncVectorEnv:
    """Synchronous vectorised env with auto-reset and reward-summing action repeat."""

    def __init__(
        self,
        env_fns: list[Any],
        action_repeat: int = 1,
    ) -> None:
        self._envs: list[gymnasium.Env] = [fn() for fn in env_fns]  # type: ignore[misc]
        self._action_repeat = action_repeat
        self._num_envs = len(self._envs)

        ref_obs_space = self._envs[0].observation_space
        assert isinstance(ref_obs_space, spaces.Dict)
        self._obs_keys: list[str] = sorted(ref_obs_space.spaces.keys())

        # Expose the single-env observation space (per-env shapes, no leading N).
        # The agent/encoder reads this to determine feature dims; N is the batch dim
        # that appears at runtime in the stacked arrays returned by reset/step.
        self._observation_space = ref_obs_space

        self._action_space = self._envs[0].action_space

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        # make_vector_env always builds a Box action space for these envs.
        return cast(spaces.Box, self._action_space)

    @property
    def num_envs(self) -> int:
        return self._num_envs

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
        """Step all envs; done envs return reset observations and terminal ``final_info``."""
        N = self._num_envs
        rews = np.zeros(N, dtype=np.float32)
        done = np.zeros(N, dtype=bool)
        terms = np.zeros(N, dtype=bool)
        truncs = np.zeros(N, dtype=bool)
        final_info_per_env: list[dict[str, Any] | None] = [None] * N
        cur_obs: list[dict[str, np.ndarray]] = [{} for _ in range(N)]
        step_success = np.zeros(N, dtype=np.float32)
        ctrl_valid = np.zeros(N, dtype=bool)
        ctrl_clamp = np.zeros((N, 3), dtype=np.float32)
        ctrl_retained_xyz = np.zeros((N, 3), dtype=np.float32)
        ctrl_achieved_xyz = np.zeros((N, 3), dtype=np.float32)

        for r in range(self._action_repeat):
            # Only the last repeat's obs is ever read: cur_obs[i] is
            # overwritten every repeat, and a done env's obs is replaced by
            # the reset obs below regardless of which repeat it died on. So
            # only render the (expensive) camera image on the final repeat.
            is_last_repeat = r == self._action_repeat - 1
            for i in range(N):
                if done[i]:
                    continue
                o, rew, term, trunc, info = self._envs[i].step(
                    actions[i],
                    render=is_last_repeat,  # ty: ignore[unknown-argument]
                )
                rews[i] += float(rew)
                cur_obs[i] = o
                step_success[i] = max(step_success[i], float(info.get("step_success", 0.0)))
                ctrl_step = info.get("ctrl_step_diag")
                if ctrl_step is not None:
                    ctrl_valid[i] = True
                    ctrl_clamp[i] = np.maximum(
                        ctrl_clamp[i],
                        np.asarray(
                            [
                                ctrl_step.get("ws_clamped", 0.0),
                                ctrl_step.get("lag_clamped", 0.0),
                                ctrl_step.get("joint_limit_clamped", 0.0),
                            ],
                            dtype=np.float32,
                        ),
                    )
                    ctrl_retained_xyz[i] += np.asarray(
                        [ctrl_step.get(f"track_cmd_{axis}", 0.0) for axis in "xyz"], dtype=np.float32
                    )
                    ctrl_achieved_xyz[i] += np.asarray(
                        [ctrl_step.get(f"achieved_{axis}", 0.0) for axis in "xyz"], dtype=np.float32
                    )
                if term or trunc:
                    final_info_per_env[i] = {**info}
                    terms[i] = term
                    truncs[i] = trunc
                    done[i] = True

        for i in range(N):
            if done[i]:
                # The live observation below is a reset state rather than the
                # terminal arrival, so it cannot be paired with the preceding
                # controller outcome in a state-conditioned training loss.
                ctrl_valid[i] = False
                o_reset, _ = self._envs[i].reset()
                cur_obs[i] = o_reset

        obs = self._stack_obs(cur_obs)
        info_out: dict[str, Any] = {
            "final_info": final_info_per_env,
            "transition": {
                "success": step_success,
                "ctrl_valid": ctrl_valid,
                # Columns: workspace, following-error lag, joint limit.
                "ctrl_clamp": ctrl_clamp,
                "ctrl_retained_xyz": ctrl_retained_xyz,
                "ctrl_achieved_xyz": ctrl_achieved_xyz,
            },
        }
        return obs, rews, terms, truncs, info_out

    def close(self) -> None:
        for env in self._envs:
            with contextlib.suppress(Exception):
                env.close()

    def _stack_obs(self, obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs_list]) for k in self._obs_keys}
