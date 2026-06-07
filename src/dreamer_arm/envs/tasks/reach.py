"""Reach task: bring the TCP within ``success_threshold`` of a random 3-D target.

This is a direct port of the ``reach`` branch in the old ``yam.py``, now
factored into the arm-agnostic task protocol.

Reward
------
``shaped = exp(-10·dist)``,  ``bonus = 1.0`` on success.
``total = shaped + bonus``
"""

from __future__ import annotations

import mujoco
import numpy as np

from dreamer_arm.envs.control import EEController

# Default randomisation range (x, y, z) in metres.
_DEFAULT_RANGE: tuple[tuple[float, float], ...] = (
    (0.25, 0.55),  # x
    (-0.25, 0.25),  # y
    (0.10, 0.40),  # z
)


class ReachTask:
    """Reach task — move the TCP to a randomly placed 3-D target."""

    name = "reach"
    obs_keys: tuple[str, ...] = ("target",)

    def __init__(
        self,
        target_range: tuple[tuple[float, float], ...] = _DEFAULT_RANGE,
    ) -> None:
        self._target_range = np.asarray(target_range, dtype=np.float32)
        self._mocap_id: int = 0  # index into data.mocap_pos; set in reset_ids

    # -------------------------------------------------------------- protocol

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add a non-collidable translucent sphere as the reach target."""
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

    def reset_ids(self, model: mujoco.MjModel) -> None:
        """Find the mocap index for the target body."""
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if body_id < 0:
            raise RuntimeError("'target' body not found after model compile")
        # model.body_mocapid[body_id] is the mocap index (-1 if not a mocap body).
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError("'target' body is not a mocap body")
        self._mocap_id = mocap_id

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
    ) -> None:
        target_pos = rng.uniform(self._target_range[:, 0], self._target_range[:, 1])
        data.mocap_pos[self._mocap_id] = target_pos

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {"target": np.array(data.mocap_pos[self._mocap_id], dtype=np.float32)}

    def reward(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: EEController,
        success_threshold: float,
    ) -> tuple[float, bool]:
        tcp = controller.tcp_pos(data)
        target = np.array(data.mocap_pos[self._mocap_id], dtype=np.float32)
        dist = float(np.linalg.norm(tcp - target))
        shaped = float(np.exp(-10.0 * dist))
        success = dist < success_threshold
        bonus = 1.0 if success else 0.0
        return shaped + bonus, success
