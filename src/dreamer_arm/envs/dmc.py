"""dm_control Suite → gymnasium 1.x adapter for Dreamer.

The reference repo's wrapper still used the legacy ``done, info`` step
return; here we use the Gymnasium 1.x ``terminated, truncated`` split,
which lets the trainer distinguish proper terminal states (rare in DMC —
hence ``is_terminal = discount == 0``) from time-limit truncations.

Task names follow ``"<domain>_<task>"`` (e.g. ``"cartpole_swingup"``); the
two name patterns with three components (``"_sparse"`` tasks and
``"finger_turn_*"``) are handled explicitly.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

ObsDict = dict[str, np.ndarray]


def _parse_task(name: str) -> tuple[str, str]:
    """Split ``"<domain>_<task>"`` into the dm_control suite tuple."""
    if "sparse" in name or "finger_turn" in name:
        base, difficulty = name.rsplit("_", 1)
        domain, task = base.rsplit("_", 1)
        return domain, f"{task}_{difficulty}"
    domain, task = name.rsplit("_", 1)
    return domain, task


class DeepMindControl(gym.Env):  # type: ignore[type-arg]
    """Single dm_control task as a Gymnasium env with a Dict obs space.

    The obs dict always carries ``image`` (uint8 RGB at ``size``) plus the
    flattened proprioceptive entries from the task's ``observation_spec``.
    Scalars are wrapped as ``(1,)`` arrays for shape consistency with the
    multi-encoder.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    # Default camera ids that the reference repo found to give a clean
    # third-person view for these domains.
    _DEFAULT_CAMERAS: ClassVar[dict[str, int]] = {"quadruped": 2, "fish": 3}

    def __init__(
        self,
        name: str,
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        camera: int | None = None,
        seed: int = 0,
    ) -> None:
        from dm_control import suite

        self._domain, self._task = _parse_task(name)
        self._env = suite.load(self._domain, self._task, task_kwargs={"random": seed})
        self._action_repeat = int(action_repeat)
        self._size = size
        self._camera = camera if camera is not None else self._DEFAULT_CAMERAS.get(self._domain, 0)

        obs_spaces: dict[str, gym.Space] = {}  # type: ignore[type-arg]
        for key, value in self._env.observation_spec().items():
            shape = value.shape if len(value.shape) > 0 else (1,)
            obs_spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        obs_spaces["image"] = gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(obs_spaces)
        # Cached so reset(seed=...) can rebuild without re-parsing the name.
        self._camera_id = self._camera

        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            spec.minimum.astype(np.float32), spec.maximum.astype(np.float32), dtype=np.float32
        )

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsDict, dict[str, Any]]:
        del options
        if seed is not None:
            # dm_control's random state is fixed at env construction; rebuild
            # the suite task with a fresh seed if the trainer asks for one.
            from dm_control import suite

            self._env = suite.load(self._domain, self._task, task_kwargs={"random": seed})
        time_step = self._env.reset()
        return self._obs(time_step), {"discount": np.float32(time_step.discount or 1.0)}

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        if not np.isfinite(action).all():
            raise ValueError(f"non-finite action: {action}")
        total_reward = 0.0
        time_step = None
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            total_reward += float(time_step.reward or 0.0)
            if time_step.last():
                break
        assert time_step is not None  # action_repeat >= 1
        terminated = bool(time_step.discount == 0)
        truncated = bool(time_step.last() and not terminated)
        info = {"discount": np.float32(time_step.discount or 1.0)}
        return self._obs(time_step), total_reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._env.physics.render(*self._size, camera_id=self._camera)

    # ------------------------------------------------------------------ helpers

    def _obs(self, time_step: Any) -> ObsDict:
        obs: ObsDict = {}
        for key, value in time_step.observation.items():
            arr = np.asarray(value, dtype=np.float32)
            obs[key] = arr if arr.ndim > 0 else arr[None]
        obs["image"] = self.render()
        return obs
