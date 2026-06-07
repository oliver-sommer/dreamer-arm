"""Pick-and-place task: grasp a cube and move it to a 3-D goal.

Goal positions are sampled from ``[rest_z, z_max]`` (inclusive of table level)
so the agent can succeed by pushing as well as lifting.  Success is defined as
``‖object - goal‖ < success_threshold`` regardless of whether the object is
grasped — matching Meta-World's convention.

Reward (dense staged)
---------------------
``reach = exp(-10·‖tcp - obj‖)``            always
``lift  = 1.0 if grasped else 0.0``         grasped = lifted + hand near obj
``place = exp(-10·‖obj - goal‖)``           always (ungated — push is valid)
``bonus = 5.0 if ‖obj-goal‖ < threshold``
``total = reach + 2·lift + 4·place + bonus``

Bug-fixes vs the old yam.py
----------------------------
- Goal z was previously ``[0.10, 0.40]`` (always floating); now includes
  ``rest_z`` so on-table goals are possible.
- Success previously required ``grasped``; now it is pure distance.
- ``place`` reward was gated on ``grasped``; now it is always shaped.
"""

from __future__ import annotations

import mujoco
import numpy as np

from dreamer_arm.envs.control import EEController

# Object geometry (4 cm cube).
_FLOOR_Z: float = -0.01  # floor plane z
_OBJ_HALF: float = 0.02  # half-extent
_OBJ_REST_Z: float = _FLOOR_Z + _OBJ_HALF  # = 0.01 m — object centre at rest

# Default workspace limits.
_DEFAULT_XY_RANGE: tuple[tuple[float, float], tuple[float, float]] = (
    (0.25, 0.55),  # x
    (-0.25, 0.25),  # y
)
_DEFAULT_Z_MAX: float = 0.40

_LIFT_EPS: float = 0.03  # object must be lifted this much to count as grasped
_NEAR_EPS: float = 0.05  # gripper must be this close to object
_MIN_SEP: float = 0.10  # minimum xy distance between object spawn and goal


class PickPlaceTask:
    """Pick-and-place task — grasp a cube and carry it to a 3-D goal."""

    name = "pick_place"
    obs_keys: tuple[str, ...] = ("object", "goal")

    def __init__(
        self,
        xy_range: tuple[tuple[float, float], tuple[float, float]] = _DEFAULT_XY_RANGE,
        z_max: float = _DEFAULT_Z_MAX,
    ) -> None:
        self._xy_range = np.asarray(xy_range, dtype=np.float32)  # (2, 2) [lo, hi]
        self._z_max = float(z_max)

        # Cached IDs — filled by reset_ids().
        self._obj_body_id: int = -1
        self._obj_qpos_adr: int = -1
        self._obj_dof_adr: int = -1
        self._goal_mocap_id: int = 0

    # -------------------------------------------------------------- protocol

    def build(self, spec: mujoco.MjSpec) -> None:
        """Splice the free-jointed cube and the goal marker into *spec*."""
        # ---- Graspable cube ----
        obj = spec.worldbody.add_body(
            name="object",
            pos=[0.4, 0.0, _OBJ_REST_Z],
        )
        obj.add_freejoint(name="object_freejoint")
        obj.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[_OBJ_HALF, _OBJ_HALF, _OBJ_HALF],
            rgba=[0.9, 0.3, 0.1, 1.0],
            mass=0.05,
            contype=1,
            conaffinity=1,
            friction=[1.0, 0.01, 0.001],
        )

        # ---- Visual goal marker (non-collidable mocap) ----
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

        # The free joint adds 7 to nq; extend the home keyframe so
        # mj_resetDataKeyframe doesn't choke on a length mismatch.
        if spec.keys:
            key = spec.keys[0]
            key.qpos = [*key.qpos, 0.4, 0.0, _OBJ_REST_Z, 1.0, 0.0, 0.0, 0.0]

    def reset_ids(self, model: mujoco.MjModel) -> None:
        """Cache body and joint addresses after model compilation."""
        self._obj_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object"))
        if self._obj_body_id < 0:
            raise RuntimeError("'object' body not found after model compile")

        # Find the free joint belonging to the object body.
        obj_jnt_id = int(
            next(j for j in range(model.njnt) if model.jnt_bodyid[j] == self._obj_body_id)
        )
        self._obj_qpos_adr = int(model.jnt_qposadr[obj_jnt_id])
        self._obj_dof_adr = int(model.jnt_dofadr[obj_jnt_id])

        # Find mocap index for the goal body.
        goal_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal"))
        if goal_body_id < 0:
            raise RuntimeError("'goal' body not found after model compile")
        self._goal_mocap_id = int(model.body_mocapid[goal_body_id])

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
    ) -> None:
        """Randomise object spawn and goal with a minimum separation."""
        xy_lo = self._xy_range[:, 0]
        xy_hi = self._xy_range[:, 1]

        for _ in range(100):
            obj_xy = rng.uniform(xy_lo, xy_hi)
            # Goal z spans from resting-on-table up to z_max.
            goal_xyz = np.array(
                [
                    rng.uniform(xy_lo[0], xy_hi[0]),
                    rng.uniform(xy_lo[1], xy_hi[1]),
                    rng.uniform(_OBJ_REST_Z, self._z_max),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(obj_xy - goal_xyz[:2]) >= _MIN_SEP:
                break

        # Write object free-joint pose (position + identity quaternion).
        adr = self._obj_qpos_adr
        data.qpos[adr : adr + 3] = [obj_xy[0], obj_xy[1], _OBJ_REST_Z + 1e-3]
        data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        # Zero object velocity.
        dadr = self._obj_dof_adr
        data.qvel[dadr : dadr + 6] = 0.0

        # Place goal marker.
        data.mocap_pos[self._goal_mocap_id] = goal_xyz

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            "object": np.array(data.xpos[self._obj_body_id], dtype=np.float32),
            "goal": np.array(data.mocap_pos[self._goal_mocap_id], dtype=np.float32),
        }

    def reward(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: EEController,
        success_threshold: float,
    ) -> tuple[float, bool]:
        tcp = controller.tcp_pos(data)
        obj = np.array(data.xpos[self._obj_body_id], dtype=np.float32)
        goal = np.array(data.mocap_pos[self._goal_mocap_id], dtype=np.float32)

        d_reach = float(np.linalg.norm(tcp - obj))
        d_place = float(np.linalg.norm(obj - goal))
        obj_z = float(obj[2])

        # Grasped heuristic: object lifted off the table AND gripper near it.
        grasped = (obj_z - _OBJ_REST_Z > _LIFT_EPS) and (d_reach < _NEAR_EPS)

        reach = float(np.exp(-10.0 * d_reach))
        lift = 1.0 if grasped else 0.0
        place = float(np.exp(-10.0 * d_place))  # ungated — pushing counts
        success = d_place < success_threshold
        bonus = 5.0 if success else 0.0

        return reach + 2.0 * lift + 4.0 * place + bonus, success
