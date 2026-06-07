"""Arm-agnostic manipulation environment.

The arm and the task are **independent axes**, wired together at construction
time.  Swapping the arm (``arm="sawyer"`` → ``arm="yam"``) requires no task
changes; adding a new task requires no arm changes.

Architecture
------------
* :class:`~dreamer_arm.envs.arms.Arm` descriptor → supplies the MjSpec base
  scene (arm + floor + lighting).
* :class:`~dreamer_arm.envs.tasks.Task` → splices task bodies into the spec,
  randomises per-episode state, and computes reward.
* :class:`~dreamer_arm.envs.control.EEController` → maps the 4-D EE action
  ``[Δx, Δy, Δz, gripper]`` to position-actuator ctrl via DLS-IK.

Action space
------------
``Box(-1, 1, (4,))`` — arm-agnostic:
* ``action[:3]`` — end-effector displacement (scaled by ``arm.ee_step_m``).
* ``action[3]``  — gripper command (-1 = closed, +1 = fully open).

Observation space
-----------------
``Dict`` containing:
* ``image``: ``(H, W, 3)`` uint8 — rendered from the fixed ``dreamer_cam``.
* ``state``: float32 ``(nq + nv,)`` — full qpos + qvel (includes task object
  free-joint DoFs when present).
* Task-specific keys (e.g. ``target`` for reach; ``object``, ``goal`` for
  pick_place) — each float32 ``(3,)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np

from dreamer_arm.envs.arms import Arm, get_arm
from dreamer_arm.envs.control import EEController
from dreamer_arm.envs.tasks import Task, get_task

if TYPE_CHECKING:
    import mujoco.viewer as _viewer_mod

ObsDict = dict[str, np.ndarray]


class Manipulation(gym.Env):  # type: ignore[type-arg]
    """Arm-swappable manipulation env driven by a 4-D end-effector action.

    Parameters
    ----------
    arm:
        Arm name (``"yam"`` or ``"sawyer"``).
    task:
        Task name (``"reach"`` or ``"pick_place"``).
    size:
        ``(height, width)`` of the rendered image in pixels.
    action_repeat:
        Number of ``mj_step`` calls per ``step()`` call; reward is summed.
    success_threshold:
        Distance threshold (metres) for success.
    seed:
        RNG seed.
    viewer:
        Open a passive MuJoCo viewer window (macOS + ``mjpython`` only).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(
        self,
        arm: str = "yam",
        task: str = "pick_place",
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 2,
        success_threshold: float = 0.05,
        seed: int = 0,
        viewer: bool = False,
    ) -> None:
        self._arm_desc: Arm = get_arm(arm)
        self._task_obj: Task = get_task(task)
        self._size = size
        self._action_repeat = int(action_repeat)
        self._success_threshold = float(success_threshold)

        # Build and compile the MuJoCo model.
        spec = self._build_spec()
        self._model: mujoco.MjModel = spec.compile()
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=size[0], width=size[1])

        # Passive viewer (macOS + mjpython).
        self._passive_viewer: _viewer_mod.Handle | None = None
        if viewer:
            import mujoco.viewer as _mv

            self._passive_viewer = _mv.launch_passive(self._model, self._data)

        # Resolve IDs.
        self._camera_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "dreamer_cam")
        )
        self._home_keyframe = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        if self._home_keyframe < 0:
            raise RuntimeError(
                "No 'home' keyframe found in the model — check the arm's scene MJCF."
            )

        # Let the task cache its own IDs.
        self._task_obj.reset_ids(self._model)

        # End-effector controller.
        self._controller = EEController(self._arm_desc, self._model)

        # RNG.
        self._rng = np.random.default_rng(seed)

        # Build observation and action spaces.
        state_dim = self._model.nq + self._model.nv
        obs_spaces: dict[str, gym.Space] = {  # type: ignore[type-arg]
            "image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        }
        for key in self._task_obj.obs_keys:
            obs_spaces[key] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_spaces)
        # 4-D EE action: [Δx, Δy, Δz, gripper] ∈ [-1, 1]^4.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    # ---------------------------------------------------------------- spec build

    def _build_spec(self) -> mujoco.MjSpec:
        """Load the arm's scene MJCF and splice in the camera + task bodies."""
        scene_path: Path = self._arm_desc.scene_path
        if not scene_path.exists():
            raise FileNotFoundError(
                f"Arm scene not found at {scene_path}; arm={self._arm_desc.name!r}."
            )
        spec = mujoco.MjSpec.from_file(str(scene_path))

        # Arm-specific patches (e.g. attach a gripper to a bare model).
        if self._arm_desc.patch_spec is not None:
            self._arm_desc.patch_spec(spec)

        # Fixed third-person camera — deterministic across resets.
        spec.worldbody.add_camera(
            name="dreamer_cam",
            pos=[0.6, -0.6, 0.6],
            xyaxes=[0.7, 0.7, 0.0, -0.4, 0.4, 0.8],
        )

        # Let the task splice its own bodies (objects, goals, walls, …).
        self._task_obj.build(spec)

        return spec

    # ---------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsDict, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_keyframe)

        # Task-specific randomisation (object spawn, goal position, etc.).
        self._task_obj.reset(self._model, self._data, self._rng)

        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()
        return self._obs(), {"success": False}

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        total_reward = 0.0
        terminated = False
        for _ in range(self._action_repeat):
            self._controller.apply(action, self._model, self._data)
            mujoco.mj_step(self._model, self._data)
            self._sync_viewer()
            r, success = self._task_obj.reward(
                self._model, self._data, self._controller, self._success_threshold
            )
            total_reward += float(r)
            if success:
                terminated = True
                break

        info: dict[str, Any] = {"success": terminated}
        return self._obs(), total_reward, terminated, False, info

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self._camera_id)
        return self._renderer.render()

    def close(self) -> None:
        if self._passive_viewer is not None:
            self._passive_viewer.close()
        self._renderer.close()

    # ---------------------------------------------------------------- viewer

    def _sync_viewer(self) -> None:
        if self._passive_viewer is not None and self._passive_viewer.is_running():
            self._passive_viewer.sync()

    # ---------------------------------------------------------------- helpers

    def _obs(self) -> ObsDict:
        state = np.concatenate([self._data.qpos, self._data.qvel], dtype=np.float32)
        obs: ObsDict = {"image": self.render(), "state": state}
        obs.update(self._task_obj.observe(self._model, self._data))
        return obs
