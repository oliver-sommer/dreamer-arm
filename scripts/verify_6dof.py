"""Verify 6-DOF orientation lock in EEController.

Run with:  python scripts/verify_6dof.py
"""

import mujoco
import numpy as np

from dreamer_arm.envs.arms import get_arm
from dreamer_arm.envs.control import EEController


def quat_from_xmat(xmat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, xmat)
    return q


def angle_between_quats(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angular distance in degrees between two unit quaternions."""
    e = np.zeros(3)
    mujoco.mju_subQuat(e, qa, qb)
    return float(np.degrees(np.linalg.norm(e)))


def run(arm_name: str = "yam", n_steps: int = 50) -> None:
    arm = get_arm(arm_name)
    spec = mujoco.MjSpec.from_file(str(arm.scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)

    ctrl = EEController(arm, model)
    print(f"[{arm_name}] quat_target = {ctrl._quat_target}")

    # Reset to home.
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    q0 = quat_from_xmat(data.site_xmat[ctrl._tcp_id])
    print(
        f"[{arm_name}] orientation at home:   {q0}  (angle to target: {angle_between_quats(ctrl._quat_target, q0):.2f}°)"
    )

    # Sequence of pure-translation actions.
    actions = [
        [1, 0, 0, -1],  # +x
        [0, 1, 0, -1],  # +y
        [-1, 0, 0, -1],  # -x
        [0, -1, 0, -1],  # -y
        [0, 0, 1, -1],  # +z
    ] * (n_steps // 5)

    max_angle = 0.0
    for _i, a in enumerate(actions):
        ctrl.apply(np.array(a, dtype=np.float32), model, data)
        mujoco.mj_step(model, data)
        q = quat_from_xmat(data.site_xmat[ctrl._tcp_id])
        ang = angle_between_quats(ctrl._quat_target, q)
        max_angle = max(max_angle, ang)

    q_final = quat_from_xmat(data.site_xmat[ctrl._tcp_id])
    print(
        f"[{arm_name}] orientation at end:    {q_final}  (angle to target: {angle_between_quats(ctrl._quat_target, q_final):.2f}°)"
    )
    print(f"[{arm_name}] max orientation drift over {n_steps} steps: {max_angle:.2f}°")
    if max_angle < 15.0:
        print(f"[{arm_name}] PASS: orientation held within 15°")
    else:
        print(
            f"[{arm_name}] WARN: orientation drifted {max_angle:.1f}° — check sign/frame of mju_subQuat"
        )


if __name__ == "__main__":
    run("yam")
