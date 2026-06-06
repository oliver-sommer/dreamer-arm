"""i2rt YAM arm in MuJoCo as a Gymnasium env.

The vendored MJCF (``assets/i2rt_yam/scene.xml``) ships the arm alone; the
task-specific bits are spliced in at load time via :class:`mujoco.MjSpec` so the
on-disk asset stays pristine and reusable for future tasks.

Tasks
-----
**reach** (``task="reach"``)
    Bring the ``grasp_site`` within ``success_threshold`` of a randomly-placed
    target mocap.  Reward: ``exp(-10·dist)`` shaped + 1.0 success bonus.

**pick_place** (``task="pick_place"``, *default*)
    Grasp a 4 cm cube and carry it to a randomly-placed 3-D goal.
    Reward is dense-staged: reach-to-object → grasp/lift → transport to goal →
    success bonus.  Both the cube spawn and the goal are randomised every reset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import gymnasium as gym
import mujoco
import numpy as np

if TYPE_CHECKING:
    import mujoco.viewer as _viewer_mod

ObsDict = dict[str, np.ndarray]

NUM_ARM_JOINTS = 6
NUM_GRIPPER_DOFS = 2
NUM_ACTUATORS = 7

# pick_place object geometry — 4 cm cube, rests on the floor plane (z = FLOOR_Z).
_FLOOR_Z: float = -0.01
_OBJ_HALF: float = 0.02  # half-extent of the cube
_OBJ_REST_Z: float = _FLOOR_Z + _OBJ_HALF  # object-centre height when resting


def _asset_dir() -> Path:
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

    elif task == "pick_place":
        # Graspable cube — free-jointed so it obeys gravity and can be picked.
        obj = spec.worldbody.add_body(
            name="object",
            pos=[0.4, 0.0, _OBJ_REST_Z],
        )
        obj.add_freejoint(name="object")
        obj.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[_OBJ_HALF, _OBJ_HALF, _OBJ_HALF],
            rgba=[0.9, 0.3, 0.1, 1.0],
            mass=0.05,
            contype=1,
            conaffinity=1,
            friction=[1.0, 0.01, 0.001],
        )

        # Visual goal marker — mocap so it can be repositioned every reset.
        goal = spec.worldbody.add_body(
            name="goal",
            mocap=True,
            pos=[0.4, 0.0, 0.2],
        )
        goal.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0],
            rgba=[0.1, 0.9, 0.3, 0.4],
            contype=0,
            conaffinity=0,
        )

        # The free joint adds 7 to nq; extend the home keyframe qpos so
        # mj_resetDataKeyframe doesn't choke on a length mismatch.
        key = spec.keys[0]  # "home"
        key.qpos = [*key.qpos, 0.4, 0.0, _OBJ_REST_Z, 1.0, 0.0, 0.0, 0.0]

    else:
        raise ValueError(f"unknown YAM task: {task!r} (supported: 'reach', 'pick_place')")

    return spec


class YAM(gym.Env):  # type: ignore[type-arg]
    """YAM arm in MuJoCo with reach and pick_place tasks.

    Observations (reach)
    --------------------
    - ``image``: ``(H, W, 3)`` uint8
    - ``state``: float32 ``(nq+nv,)`` — qpos + qvel
    - ``target``: float32 ``(3,)`` — target position in world frame

    Observations (pick_place)
    -------------------------
    - ``image``: ``(H, W, 3)`` uint8
    - ``state``: float32 ``(nq+nv,)`` — includes object free-joint qpos/qvel
    - ``object``: float32 ``(3,)`` — object centre in world frame
    - ``goal``: float32 ``(3,)`` — goal position in world frame

    Action
    ------
    Continuous ``Box`` in ``[-1, 1]^7`` mapped to the model's actuator
    ``ctrlrange``.  The 7 actuators are the 6 arm joints plus the gripper
    (0 = closed, 0.041 = fully open).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(
        self,
        task: Literal["reach", "pick_place"] = "pick_place",
        size: tuple[int, int] = (64, 64),
        action_repeat: int = 2,
        success_threshold: float = 0.05,
        target_range: tuple[tuple[float, float], ...] = (
            (0.25, 0.55),  # x
            (-0.25, 0.25),  # y
            (0.10, 0.40),  # z
        ),
        seed: int = 0,
        viewer: bool = False,
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

        self._passive_viewer: _viewer_mod.Handle | None = None
        if viewer:
            import mujoco.viewer as _mv

            self._passive_viewer = _mv.launch_passive(self._model, self._data)

        self._camera_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "dreamer_cam")
        )
        self._tcp_site_id = int(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        )
        self._home_keyframe = int(mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home"))

        if task == "reach":
            self._target_body_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "target")
            )
        elif task == "pick_place":
            self._object_body_id = int(
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "object")
            )
            # Resolve qpos/dof addresses by finding the joint that belongs to
            # the object body — avoids relying on the free-joint name.
            _obj_jnt_id = int(
                next(
                    j
                    for j in range(self._model.njnt)
                    if self._model.jnt_bodyid[j] == self._object_body_id
                )
            )
            self._obj_qpos_adr: int = int(self._model.jnt_qposadr[_obj_jnt_id])
            self._obj_dof_adr: int = int(self._model.jnt_dofadr[_obj_jnt_id])

        if self._model.nu != NUM_ACTUATORS:
            raise RuntimeError(f"expected {NUM_ACTUATORS} actuators on YAM, got {self._model.nu}")

        self._rng = np.random.default_rng(seed)
        self._ctrl_low = self._model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_high = self._model.actuator_ctrlrange[:, 1].astype(np.float32)

        state_dim = self._model.nq + self._model.nv
        base_spaces: dict[str, gym.Space] = {  # type: ignore[type-arg]
            "image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32),
        }
        if task == "reach":
            base_spaces["target"] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        else:
            base_spaces["object"] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
            base_spaces["goal"] = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(base_spaces)
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

        if self._task == "reach":
            target_pos = self._rng.uniform(self._target_range[:, 0], self._target_range[:, 1])
            self._data.mocap_pos[0] = target_pos

        else:  # pick_place
            # Sample object x/y within workspace; object rests on the floor.
            xy_lo = self._target_range[:2, 0]
            xy_hi = self._target_range[:2, 1]
            _MIN_SEP = 0.10  # minimum x/y distance between object and goal
            for _ in range(100):
                obj_xy = self._rng.uniform(xy_lo, xy_hi)
                goal = self._rng.uniform(self._target_range[:, 0], self._target_range[:, 1])
                if np.linalg.norm(obj_xy - goal[:2]) >= _MIN_SEP:
                    break

            # Write object free-joint qpos (3 pos + 4 quat) and zero velocity.
            adr = self._obj_qpos_adr
            self._data.qpos[adr : adr + 3] = [obj_xy[0], obj_xy[1], _OBJ_REST_Z + 1e-3]
            self._data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
            dadr = self._obj_dof_adr
            self._data.qvel[dadr : dadr + 6] = 0.0

            self._data.mocap_pos[0] = goal

        mujoco.mj_forward(self._model, self._data)
        self._sync_viewer()
        return self._obs(), {"success": False}

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, object]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self._ctrl_low + 0.5 * (action + 1.0) * (self._ctrl_high - self._ctrl_low)

        total_reward = 0.0
        terminated = False
        for _ in range(self._action_repeat):
            self._data.ctrl[:] = ctrl
            mujoco.mj_step(self._model, self._data)
            self._sync_viewer()
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
        if self._passive_viewer is not None:
            self._passive_viewer.close()
        self._renderer.close()

    # ----------------------------------------------------------------- viewer

    def _sync_viewer(self) -> None:
        if self._passive_viewer is not None and self._passive_viewer.is_running():
            self._passive_viewer.sync()

    # ---------------------------------------------------------------- helpers

    def _tcp_pos(self) -> np.ndarray:
        return self._data.site_xpos[self._tcp_site_id].astype(np.float32, copy=True)

    def _target_pos(self) -> np.ndarray:
        return self._data.mocap_pos[0].astype(np.float32, copy=True)

    def _object_pos(self) -> np.ndarray:
        return self._data.xpos[self._object_body_id].astype(np.float32, copy=True)

    def _goal_pos(self) -> np.ndarray:
        return self._data.mocap_pos[0].astype(np.float32, copy=True)

    def _reward(self) -> tuple[float, bool]:
        if self._task == "reach":
            dist = float(np.linalg.norm(self._tcp_pos() - self._target_pos()))
            shaped = float(np.exp(-10.0 * dist))
            success = dist < self._success_threshold
            bonus = 1.0 if success else 0.0
            return shaped + bonus, success

        # pick_place staged reward
        tcp = self._tcp_pos()
        obj = self._object_pos()
        goal = self._goal_pos()

        d_reach = float(np.linalg.norm(tcp - obj))
        d_place = float(np.linalg.norm(obj - goal))
        obj_z = float(obj[2])

        # Grasped: object lifted off the resting surface AND gripper near object.
        grasped = (obj_z - _OBJ_REST_Z > 0.03) and (d_reach < 0.05)

        reach = float(np.exp(-10.0 * d_reach))
        lift = 1.0 if grasped else 0.0
        place = float(np.exp(-10.0 * d_place)) if grasped else 0.0
        success = grasped and (d_place < self._success_threshold)
        bonus = 5.0 if success else 0.0

        return reach + 2.0 * lift + 4.0 * place + bonus, success

    def _obs(self) -> ObsDict:
        state = np.concatenate([self._data.qpos, self._data.qvel], dtype=np.float32)
        obs: ObsDict = {"image": self.render(), "state": state}
        if self._task == "reach":
            obs["target"] = self._target_pos()
        else:
            obs["object"] = self._object_pos()
            obs["goal"] = self._goal_pos()
        return obs
