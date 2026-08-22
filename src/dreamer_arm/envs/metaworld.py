"""Gymnasium Dict-obs adapter wrapping a single Meta-World task env.

Converts Meta-World's 39-dim flat ``state`` observation into a Dict of
non-privileged modalities that the Dreamer encoder can consume:

  ``scene``      - uint8 RGB (H, W, 3) from our own ``mujoco.Renderer``
  ``wrist_image``- uint8 RGB (H, W, 3) from the wrist camera (optional)
  ``proprio``    - float32 (10,): arm joint angles (6), gripper opening (1),
                   TCP xyz (3).  For Sawyer the joint-angle slots are zeroed.
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
from typing import Any, ClassVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_PROPRIO_DIM = 10  # 6 joint angles + 1 gripper open + 3 TCP xyz

# Hemisphere sampling for camera DR (world frame, z=up)
_CAM_LOOKAT = np.array([0.0, 0.55, 0.15])
_CAM_AZIMUTH_RANGE = (100.0, 200.0)  # degrees (behind arm)
_CAM_ELEVATION_RANGE = (-50.0, -15.0)  # degrees
_CAM_DISTANCE_RANGE = (0.85, 1.3)  # metres


# ---------------------------------------------------------------------------
# Barrel distortion helper
# ---------------------------------------------------------------------------


def _apply_barrel_distortion(image: np.ndarray, k1: float) -> np.ndarray:
    """Barrel (k1>0) / pincushion (k1<0) distortion for wrist fisheye."""
    if abs(k1) < 1e-6:
        return image
    try:
        from scipy.ndimage import map_coordinates  # type: ignore[import-not-found]
    except ImportError:
        return image

    h, w, c = image.shape
    cx, cy = w / 2.0, h / 2.0
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    xn = (x_idx - cx) / cx
    yn = (y_idx - cy) / cy
    r2 = xn**2 + yn**2
    factor = 1.0 + k1 * r2
    xs = xn * factor * cx + cx
    ys = yn * factor * cy + cy
    out = np.empty_like(image)
    for ch in range(c):
        out[:, :, ch] = map_coordinates(
            image[:, :, ch].astype(np.float32),
            [ys, xs],
            order=1,
            mode="nearest",
        )
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


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
        self._arm = arm
        self._size = size
        self._camera = camera
        self._wrist_camera = wrist_camera
        self._scene_randomize = scene_randomize
        self._camera_pose_randomize = camera_pose_randomize
        self._camera_jitter = camera_jitter
        self._wrist_fisheye = wrist_fisheye
        self._task_idx = task_idx
        self._num_tasks = num_tasks
        self._success_threshold = success_threshold

        self._env = env
        env.set_task(task)
        self._task_name: str = str(task.env_name)

        # Single renderer — one EGL context for both scene and wrist
        self._renderer = mujoco.Renderer(env.model, height=size[0], width=size[1])
        self._scene_cam: mujoco.MjvCamera | None = None  # set each reset

        # mujoco.Renderer.render() flips the GL buffer assuming the EGL / GLFW /
        # OSMesa convention (bottom-to-top), which is correct on the Linux/EGL
        # training box.  The macOS CGL backend already returns pixels top-to-
        # bottom, so that flip leaves CGL renders upside down.  Undo it *only*
        # on CGL so output is upright on every backend.
        gl_ctx = getattr(self._renderer, "_gl_context", None)
        self._flip_vertical = type(gl_ctx).__module__.endswith(".cgl") if gl_ctx else False

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
        self._reset_ctrl_diag()

        self._episode_success: bool = False
        self._rng = np.random.default_rng()

        self._light_diffuse0 = m.light_diffuse.copy()
        self._light_ambient0 = m.light_ambient.copy()

        # Passive viewer (mjpython only; None when viewer=False)
        self._mj_viewer: Any = None
        if viewer:
            import mujoco.viewer as _mjv

            self._mj_viewer = _mjv.launch_passive(env.model, env.data)

        # Observation & action spaces
        H, W = size
        obs_dict: dict[str, spaces.Space[Any]] = {
            "scene": spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (_PROPRIO_DIM,), dtype=np.float32),
        }
        if wrist_camera is not None:
            obs_dict["wrist_image"] = spaces.Box(0, 255, (H, W, 3), dtype=np.uint8)
        if task_idx is not None and num_tasks is not None:
            obs_dict["task_id"] = spaces.Box(0.0, 1.0, (num_tasks,), dtype=np.float32)
        self.observation_space: spaces.Dict = spaces.Dict(obs_dict)
        self.action_space: spaces.Box = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

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
        self._reset_ctrl_diag()
        if self._is_yam:
            self._prev_tcp = np.asarray(self._env.data.site_xpos[self._grasp_site_id], dtype=np.float64).copy()

        # Camera setup for this episode
        if self._camera_pose_randomize or self._camera_jitter > 0.0:
            self._scene_cam = self._make_free_camera()
        else:
            self._scene_cam = None  # use named camera string

        if self._scene_randomize:
            self._randomise_lighting()

        obs = self._get_obs_dict()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        try:
            _, reward, terminated, truncated, inner_info = self._env.step(action)
        except (ValueError, RuntimeError):
            # Physics instability (NaN positions) propagated into reward computation.
            # This is not an MDP terminal state, so flag it as a *truncation* (not
            # terminated) — otherwise the critic learns these glitch states have
            # zero future value. Zero reward, and the episode resets.
            obs = self._get_obs_dict()
            return obs, 0.0, False, True, self._build_info()

        if self._is_yam:
            self._accumulate_ctrl_diag()

        if float(inner_info.get("success", 0.0)) >= self._success_threshold:
            self._episode_success = True

        if self._mj_viewer is not None:
            self._mj_viewer.sync()

        obs = self._get_obs_dict()
        return obs, float(reward), bool(terminated), bool(truncated), self._build_info()

    def close(self) -> None:
        if self._mj_viewer is not None:
            with contextlib.suppress(Exception):
                self._mj_viewer.close()
            self._mj_viewer = None
        with contextlib.suppress(Exception):
            self._renderer.close()
        with contextlib.suppress(Exception):
            self._env.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._renderer.close()

    # ------------------------------------------------------------------
    # Controller diagnostics (YAM)
    # ------------------------------------------------------------------

    def _reset_ctrl_diag(self) -> None:
        """Zero the per-episode EEController diagnostic aggregates."""
        self._cd_n = 0
        self._cd_sigma_sum = 0.0
        self._cd_sigma_min = float("inf")
        self._cd_ori_capped = 0
        self._cd_dq_clamped = 0
        self._cd_track_sum = 0.0
        self._cd_track_n = 0
        self._cd_stuck = 0
        self._prev_tcp: np.ndarray | None = None

    def _accumulate_ctrl_diag(self) -> None:
        """Fold the controller's last per-step diagnostics into episode aggregates.

        ``YamArm.actuate`` stashes its per-step diag on ``env._ctrl_diag``; here
        we read it after the inner step and also measure the *achieved* TCP
        displacement (at the IK-controlled ``grasp_site``) against the commanded
        step.  A persistently low achieved/commanded ratio is the stuck signal.
        """
        diag = getattr(self._env, "_ctrl_diag", None)
        if not diag:
            return
        self._cd_n += 1
        sigma = float(diag.get("sigma_min", 0.0))
        self._cd_sigma_sum += sigma
        self._cd_sigma_min = min(self._cd_sigma_min, sigma)
        self._cd_ori_capped += int(diag.get("ori_capped", 0.0) > 0.0)
        self._cd_dq_clamped += int(diag.get("dq_clamped", 0.0) > 0.0)

        tcp = np.asarray(self._env.data.site_xpos[self._grasp_site_id], dtype=np.float64)
        cmd = float(diag.get("cmd_norm", 0.0))
        # Only count steps that actually commanded motion (cmd ~ 0 → ratio is
        # meaningless and would falsely read as "stuck").
        if self._prev_tcp is not None and cmd > 1e-4:
            ratio = float(np.linalg.norm(tcp - self._prev_tcp)) / cmd
            self._cd_track_sum += ratio
            self._cd_track_n += 1
            if ratio < 0.25:
                self._cd_stuck += 1
        self._prev_tcp = tcp

    def _ctrl_diag_summary(self) -> dict[str, float] | None:
        """Episode-aggregate controller diagnostics, or ``None`` if unavailable."""
        if not self._is_yam or self._cd_n == 0:
            return None
        return {
            "sigma_min_mean": self._cd_sigma_sum / self._cd_n,
            "sigma_min_min": self._cd_sigma_min,
            "frac_ori_capped": self._cd_ori_capped / self._cd_n,
            "frac_dq_clamped": self._cd_dq_clamped / self._cd_n,
            "track_ratio_mean": (self._cd_track_sum / self._cd_track_n if self._cd_track_n else 1.0),
            "frac_stuck": (self._cd_stuck / self._cd_track_n if self._cd_track_n else 0.0),
        }

    def _build_info(self) -> dict[str, Any]:
        """Step/reset info dict, tagged with task name, success, and ctrl_diag."""
        info: dict[str, Any] = {"task_name": self._task_name, "success": self._episode_success}
        summary = self._ctrl_diag_summary()
        if summary is not None:
            info["ctrl_diag"] = summary
        return info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {
            "scene": self._render_scene(),
            "proprio": self._get_proprio(),
        }
        if self._wrist_camera is not None:
            obs["wrist_image"] = self._render_wrist()
        if self._task_idx is not None and self._num_tasks is not None:
            one_hot = np.zeros(self._num_tasks, dtype=np.float32)
            one_hot[self._task_idx] = 1.0
            obs["task_id"] = one_hot
        return obs

    def _get_proprio(self) -> np.ndarray:
        d = self._env.data
        joints = (
            d.qpos[self._arm_qadr].astype(np.float32) if self._arm_qadr is not None else np.zeros(6, dtype=np.float32)
        )
        lc = np.asarray(d.body("leftclaw").xpos)
        rc = np.asarray(d.body("rightclaw").xpos)
        gripper_open = np.array(
            [float(np.clip(np.linalg.norm(rc - lc) / 0.1, 0.0, 1.0))],
            dtype=np.float32,
        )
        tcp = np.asarray(d.body("hand").xpos, dtype=np.float32)
        return np.concatenate([joints, gripper_open, tcp])

    def _render_scene(self) -> np.ndarray:
        if self._scene_cam is not None:
            self._renderer.update_scene(self._env.data, camera=self._scene_cam)
        else:
            self._renderer.update_scene(self._env.data, camera=self._camera)
        return self._grab_frame()

    def _render_wrist(self) -> np.ndarray:
        self._renderer.update_scene(self._env.data, camera=self._wrist_camera)
        frame = self._grab_frame()
        if self._wrist_fisheye != 0.0:
            frame = _apply_barrel_distortion(frame, self._wrist_fisheye)
        return frame

    def _grab_frame(self) -> np.ndarray:
        """Render the current scene, correcting the CGL vertical flip (see __init__)."""
        frame = self._renderer.render()
        return np.flipud(frame).copy() if self._flip_vertical else frame.copy()

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def _make_free_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if self._camera_pose_randomize:
            cam.azimuth = float(self._rng.uniform(*_CAM_AZIMUTH_RANGE))
            cam.elevation = float(self._rng.uniform(*_CAM_ELEVATION_RANGE))
            cam.distance = float(self._rng.uniform(*_CAM_DISTANCE_RANGE))
        else:
            cam.azimuth = 150.0
            cam.elevation = -35.0
            cam.distance = 1.0
        lookat = _CAM_LOOKAT.copy()
        if self._camera_jitter > 0.0:
            lookat += self._rng.normal(0.0, self._camera_jitter, 3)
        cam.lookat[:] = lookat
        return cam

    def _randomise_lighting(self) -> None:
        # Scale the captured baseline, never the current value: MjModel persists
        # across resets, so compounding would fade the scene to black (both
        # factors have mean < 1, ~0.85x per episode).
        m = self._env.model
        for li in range(m.nlight):
            tint = self._rng.uniform(0.7, 1.0, 3).astype(np.float32)
            m.light_diffuse[li] = np.clip(self._light_diffuse0[li] * tint, 0.0, 1.0)
            m.light_ambient[li] = np.clip(self._light_ambient0[li] * float(self._rng.uniform(0.5, 1.2)), 0.0, 1.0)
