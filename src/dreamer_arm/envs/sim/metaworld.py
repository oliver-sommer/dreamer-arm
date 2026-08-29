"""Gymnasium Dict-obs adapter wrapping a single Meta-World task env.

Converts Meta-World's 39-dim flat ``state`` observation into a Dict of
non-privileged modalities that the Dreamer encoder can consume:

  ``scene``      - uint8 RGB (H, W, 3) from our own ``mujoco.Renderer``
  ``wrist_image``- uint8 RGB (H, W, 3) from the wrist camera (optional)
  ``proprio``    - float32 (10,): arm joint angles (6), gripper opening (1),
                   controlled-tool xyz (3). For Sawyer the joint-angle slots
                   are zeroed.
  ``task_id``    - float32 one-hot (num_tasks,) for multi-task runs (optional)

The privileged ``state`` returned by ``env._get_obs()`` is **never** fed to
the agent; it bundles object/goal coordinates unavailable on a real robot.

Domain randomisation (all toggleable per episode via ``reset()``):
  * Camera pose on a hemisphere behind the arm
  * Small camera-position jitter
  * Scene-lighting colour / intensity variation
  * Wrist barrel distortion (via scipy; silently skipped if unavailable)

Sticky success:  Meta-World never terminates on success, but signals
``info["success"]`` (float 0/1) every step.  This adapter ORs the flag across
the episode so the vector env's ``final_info["success"]`` reflects the whole
episode.
"""

from __future__ import annotations

import contextlib
import numbers
from typing import Any, ClassVar

import gymnasium
import mujoco
import numpy as np

from dreamer_arm.envs.action import ACTION_SPEC
from dreamer_arm.envs.control.metrics import ControllerMetrics
from dreamer_arm.envs.observation import ObservationSpec
from dreamer_arm.envs.sim.arms.base import Arm
from dreamer_arm.envs.sim.rendering import SceneRenderer

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


class MetaWorldEnv(gymnasium.Env):  # type: ignore[misc]
    """Dict-obs Gymnasium wrapper around a single Meta-World task env."""

    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        env: Any,
        task: Any,
        arm: str,
        size: tuple[int, int],
        camera: str,
        *,
        arm_plugin: Arm | None = None,
        wrist_camera: str | None = None,
        scene_randomize: bool = False,
        camera_pose_randomize: bool = False,
        camera_jitter: float = 0.0,
        wrist_fisheye: float = 0.0,
        task_idx: int | None = None,
        num_tasks: int | None = None,
        success_threshold: float = 1.0,
        viewer: bool = False,
    ) -> None:
        super().__init__()
        self._arm_plugin = arm_plugin
        self._wrist_camera = wrist_camera
        self._task_idx = task_idx
        self._success_threshold = success_threshold
        if (task_idx is None) != (num_tasks is None):
            raise ValueError("task_idx and num_tasks must be supplied together")
        self._action_spec = ACTION_SPEC
        self._observation_spec = ObservationSpec(
            image_size=size,
            wrist_image=wrist_camera is not None,
            task_count=num_tasks,
        )

        self._env = env
        env.set_task(task)
        self._task_name: str = str(task.env_name)

        self._rendering = SceneRenderer(
            env,
            size,
            camera,
            wrist_camera,
            scene_randomize=scene_randomize,
            camera_pose_randomize=camera_pose_randomize,
            camera_jitter=camera_jitter,
            wrist_fisheye=wrist_fisheye,
        )

        # Last rendered frame(s), returned by step(render=False) instead of a
        # fresh render -- see step()'s docstring.  Always set on reset()
        # (which always renders), so step() never sees them as None.
        self._last_scene: np.ndarray | None = None
        self._last_wrist: np.ndarray | None = None

        # No post-render flip: mujoco.Renderer.render() normalises the GL buffer
        # itself and returns upright pixels on every backend (verified on CGL and
        # EGL with a known-geometry probe).  Orientation is now purely a property
        # of the camera declarations in yam_xyz_base.xml, which are rolled so that
        # every up-vector points up.  See the camera comment there for why an
        # np.flipud "fix" was actively harmful.

        # Resolve joint addresses for proprio
        m = env.model
        if arm == "yam":
            self._arm_qadr: np.ndarray | None = np.array(
                [int(m.jnt_qposadr[m.joint(n).id]) for n in _ARM_JOINT_NAMES],
                dtype=np.int32,
            )
        else:
            self._arm_qadr = None  # Sawyer: zeros

        # Controller diagnostics (YAM only): grasp_site is the IK-controlled
        # point, used to measure achieved-vs-commanded TCP motion (the stuck
        # signal).  Aggregates are reset each episode in reset().
        self._is_yam = arm == "yam"
        self._grasp_site_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")) if self._is_yam else -1
        self._controller_metrics = ControllerMetrics(self._is_yam, self._grasp_site_id)

        self._episode_success: bool = False
        self._reward_component_sums: dict[str, float] = {}
        self._reward_component_counts: dict[str, int] = {}
        self._rng = np.random.default_rng()

        # Passive viewer (mjpython only; None when viewer=False)
        self._mj_viewer: Any = None
        if viewer:
            import mujoco.viewer as _mjv

            self._mj_viewer = _mjv.launch_passive(env.model, env.data)

        self.observation_space = self._observation_spec.make_space()
        self.action_space = self._action_spec.make_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Inner env reset (calls _reset_hand → YamArm.reset_hand → sets init_tcp)
        self._env.reset()
        self._episode_success = False
        self._reward_component_sums.clear()
        self._reward_component_counts.clear()
        self._controller_metrics.reset(self._env.data)

        self._rendering.reset(self._rng)

        obs = self._get_obs_dict()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray, *, render: bool = True
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step the inner env.

        ``render=False`` skips the camera render(s) in the returned obs
        (``proprio``/``task_id`` are always computed -- they're cheap and
        ``SyncVectorEnv`` needs correct non-image obs every inner step).  The
        ``scene``/``wrist_image`` keys still come back correctly shaped: they
        hold the *last actually rendered* frame rather than a fresh one. This
        is for ``SyncVectorEnv``'s ``action_repeat`` loop, where every repeat
        but the last is immediately overwritten and its rendered pixels are
        never read -- see ``wrappers.py``.
        """
        try:
            action = self._action_spec.coerce(action)
            _, reward, terminated, truncated, inner_info = self._env.step(action)
        except (ValueError, RuntimeError):
            # Physics instability (NaN positions) propagated into reward computation.
            # This is not an MDP terminal state, so flag it as a *truncation* (not
            # terminated) — otherwise the critic learns these glitch states have
            # zero future value. Zero reward, and the episode resets.
            obs = self._get_obs_dict(render=render)
            return obs, 0.0, False, True, self._build_info()

        if self._is_yam:
            diagnostics = (
                self._arm_plugin.last_diagnostics
                if self._arm_plugin is not None
                else getattr(self._env, "_ctrl_diag", None)
            )
            self._controller_metrics.accumulate(diagnostics, self._env.data)

        if float(inner_info.get("success", 0.0)) >= self._success_threshold:
            self._episode_success = True
        self._accumulate_reward_info(inner_info)

        if self._mj_viewer is not None:
            self._mj_viewer.sync()

        obs = self._get_obs_dict(render=render)
        info = self._build_info()
        # Episode summaries alone cannot supervise action feasibility: by the
        # time they are logged, the individual command that caused a clamp is
        # lost. Expose the current controller step separately; SyncVectorEnv
        # aggregates it across action repeats before replay storage.
        info["step_success"] = float(inner_info.get("success", 0.0))
        if self._is_yam and diagnostics:
            info["ctrl_step_diag"] = {
                key: float(diagnostics.get(key, 0.0))
                for key in (
                    "ws_clamped",
                    "lag_clamped",
                    "joint_limit_clamped",
                    "track_cmd_x",
                    "track_cmd_y",
                    "track_cmd_z",
                    "achieved_x",
                    "achieved_y",
                    "achieved_z",
                )
            }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._mj_viewer is not None:
            with contextlib.suppress(Exception):
                self._mj_viewer.close()
            self._mj_viewer = None
        with contextlib.suppress(Exception):
            self._rendering.close()
        with contextlib.suppress(Exception):
            self._env.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._rendering.close()

    def _build_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"task_name": self._task_name, "success": self._episode_success}
        if self._reward_component_sums:
            info["reward_diag"] = {
                key: total / self._reward_component_counts[key] for key, total in self._reward_component_sums.items()
            }
        summary = self._controller_metrics.summary()
        if summary is not None:
            info["ctrl_diag"] = summary
        return info

    def _accumulate_reward_info(self, inner_info: dict[str, Any]) -> None:
        """Retain Meta-World's task reward terms for episode diagnostics.

        The scalar reward remains completely unchanged.  Exposing the numeric
        components makes it possible to see whether the model is exploiting a
        dense shaping term while task success remains zero.
        """
        for key, value in inner_info.items():
            if key == "success" or not isinstance(value, numbers.Real):
                continue
            numeric = float(value)
            if not np.isfinite(numeric):
                continue
            self._reward_component_sums[key] = self._reward_component_sums.get(key, 0.0) + numeric
            self._reward_component_counts[key] = self._reward_component_counts.get(key, 0) + 1

    def _get_obs_dict(self, render: bool = True) -> dict[str, np.ndarray]:
        if render or self._last_scene is None:
            self._last_scene = self._render_scene()
        if self._wrist_camera is not None and (render or self._last_wrist is None):
            self._last_wrist = self._render_wrist()

        d = self._env.data
        joints = (
            d.qpos[self._arm_qadr].astype(np.float32) if self._arm_qadr is not None else np.zeros(6, dtype=np.float32)
        )
        gripper_open = float(self._env.get_gripper_open())
        # The shared contract names the Cartesian point controlled by each
        # backend. YAM controls grasp_site; Sawyer's upstream mocap path and
        # rewards use the mean of its two finger sites (tcp_center).
        tool_position = (
            np.asarray(d.site_xpos[self._grasp_site_id], dtype=np.float32)
            if self._is_yam
            else np.asarray(self._env.tcp_center, dtype=np.float32)
        )
        assert self._last_scene is not None
        return self._observation_spec.make(
            scene=self._last_scene,
            joint_positions=joints,
            gripper_open=gripper_open,
            tool_position=tool_position,
            wrist_image=self._last_wrist,
            task_index=self._task_idx,
        )

    def _render_scene(self) -> np.ndarray:
        return self._rendering.render_scene()

    def _render_wrist(self) -> np.ndarray:
        return self._rendering.render_wrist()
