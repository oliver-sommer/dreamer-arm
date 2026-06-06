"""i2rt YAM arm in MuJoCo as a Gymnasium env.

The vendored MJCF (``assets/i2rt_yam/scene.xml``) ships the arm alone; the
task-specific bits (a mocap target body for "reach") are spliced in at load
time via :class:`mujoco.MjSpec` so the on-disk asset stays pristine and
reusable for future tasks.

The first task is **reach**: the arm has to bring its ``grasp_site`` close to
a randomly-placed target mocap. Reward is a shaped distance term plus a
success bonus when the TCP is within ``success_threshold`` metres.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import gymnasium as gym
import mujoco
import numpy as np

ObsDict = dict[str, np.ndarray]

# Joints + actuators expected by the vendored YAM model. We hard-code the
# count instead of doing dynamic discovery so a mismatch (e.g. wrong asset)
# fails fast at construction time rather than silently sampling bad shapes.
NUM_ARM_JOINTS = 6
NUM_GRIPPER_DOFS = 2  # left + right finger (coupled by an equality)
NUM_ACTUATORS = 7  # 6 joints + 1 gripper position


def _asset_dir() -> Path:
    """Resolve the absolute path to the vendored i2rt_yam asset dir."""
    # Asset dir lives at <repo>/assets/i2rt_yam; this file is at
    # <repo>/src/dreamer_arm/envs/yam.py.
    return Path(__file__).resolve().parents[3] / "assets" / "i2rt_yam"


def _build_spec(task: str, render_size: tuple[int, int]) -> mujoco.MjSpec:
    """Load ``scene.xml`` via MjSpec and inject task-specific bodies/cameras."""
    asset_dir = _asset_dir()
    scene_path = asset_dir / "scene.xml"
    if not scene_path.exists():
        raise FileNotFoundError(
            f"YAM asset not found at {scene_path}; expected vendored "
            "mujoco_menagerie/i2rt_yam under assets/."
        )
    spec = mujoco.MjSpec.from_file(str(scene_path))

    # Fixed third-person camera so renders are deterministic across resets.
    spec.worldbody.add_camera(
        name="dreamer_cam",
        pos=[0.6, -0.6, 0.6],
        xyaxes=[0.7, 0.7, 0.0, -0.4, 0.4, 0.8],
    )

    if task == "reach":
        target = spec.worldbody.add_body(
            name="target",
            mocap=True,
            pos=[0.4, 0.0, 0.2],
        )
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            rgba=[0.1, 0.4, 1.0, 0.6],
            contype=0,
            conaffinity=0,
        )
    else:
        raise ValueError(f"unknown YAM task: {task!r} (only 'reach' is implemented)")

    return spec


class YAM(gym.Env):  # type: ignore[type-arg]
    """YAM arm + reach task as a Dreamer-shaped Gymnasium env.

    Observation
    -----------
    - ``image``: ``(H, W, 3)`` uint8 — render of ``dreamer_cam``.
    - ``state``: float32 — concatenation of ``qpos`` and ``qvel``.
    - ``target``: float32 ``(3,)`` — target position in world frame.

    Action
    ------
    Continuous ``Box`` in ``[-1, 1]^7`` mapped to the model's actuator
    ``ctrlrange``. The 7 actuators are the 6 arm joints plus the gripper.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(
        self,
        task: Literal["reach"] = "reach",
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 2,
        success_threshold: float = 0.05,
        target_range: tuple[tuple[float, float], ...] = (
            (0.25, 0.55),  # x
            (-0.25, 0.25),  # y
            (0.10, 0.40),  # z
        ),
        seed: int = 0,
    ) -> None:
        self._task = task
        self._size = size
        self._action_repeat = int(action_repeat)
        self._success_threshold = float(success_threshold)
        self._target_range = np.asarray(target_range, dtype=np.float32)

        spec = _build_spec(task, size)
        self._model: mujoco.MjModel = spec.compile()
        self._data: mujoco.MjData = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=size[0], width=size[1])
        self._camera_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "dreamer_cam")
        )
        self._tcp_site_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        )
        self._target_body_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "target")
        )
        self._home_keyframe = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        )

        # Sanity checks — fail fast if the asset has drifted.
        if self._model.nu != NUM_ACTUATORS:
            raise RuntimeError(
                f"expected {NUM_ACTUATORS} actuators on YAM, got {self._model.nu}"
            )

        self._rng = np.random.default_rng(seed)
        self._ctrl_low = self._model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_high = self._model.actuator_ctrlrange[:, 1].astype(np.float32)

        state_dim = self._model.nq + self._model.nv
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
                "target": gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (NUM_ACTUATORS,), dtype=np.float32)

    # ---------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObsDict, dict[str, object]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_keyframe)
        # Randomise target position within the configured workspace.
        target_pos = self._rng.uniform(self._target_range[:, 0], self._target_range[:, 1])
        self._data.mocap_pos[0] = target_pos
        mujoco.mj_forward(self._model, self._data)
        return self._obs(), {"success": False}

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, object]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self._ctrl_low + 0.5 * (action + 1.0) * (self._ctrl_high - self._ctrl_low)

        total_reward = 0.0
        terminated = False
        for _ in range(self._action_repeat):
            self._data.ctrl[:] = ctrl
            mujoco.mj_step(self._model, self._data)
            r, success = self._reward()
            total_reward += float(r)
            if success:
                terminated = True
                break

        info: dict[str, object] = {"success": terminated}
        return self._obs(), total_reward, terminated, False, info

    def render(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self._camera_id)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    # ---------------------------------------------------------------- helpers

    def _tcp_pos(self) -> np.ndarray:
        return self._data.site_xpos[self._tcp_site_id].astype(np.float32, copy=True)

    def _target_pos(self) -> np.ndarray:
        return self._data.mocap_pos[0].astype(np.float32, copy=True)

    def _reward(self) -> tuple[float, bool]:
        dist = float(np.linalg.norm(self._tcp_pos() - self._target_pos()))
        # Bounded shaped reward in (0, 1]: 1 at the target, ~0 far away.
        # The 10x scale gives meaningful gradient inside ~20 cm.
        shaped = float(np.exp(-10.0 * dist))
        success = dist < self._success_threshold
        bonus = 1.0 if success else 0.0
        return shaped + bonus, success

    def _obs(self) -> ObsDict:
        state = np.concatenate([self._data.qpos, self._data.qvel], dtype=np.float32)
        return {
            "image": self.render(),
            "state": state,
            "target": self._target_pos(),
        }
