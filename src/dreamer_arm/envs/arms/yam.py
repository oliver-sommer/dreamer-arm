"""YAM arm descriptor for the manipulation framework.

Values derived from ``assets/i2rt_yam/yam.xml``:
- 6 revolute joints (joint1-joint6), all position-actuated.
- 1 gripper actuator on ``left_finger`` (right mirrors via equality),
  ctrl range ``[0.0, 0.041]``.
- TCP site ``grasp_site`` at link_6 local ``[0, 0, 0.1347]``.
- Home keyframe: ``qpos = [0, 1.047, 1.047, 0, 0, 0, 0, 0]`` (6 arm + 2 finger
  DOFs), ``ctrl = [0, 1.047, 1.047, 0, 0, 0, 0]`` (6 arm + 1 gripper actuator).
"""

from __future__ import annotations

from pathlib import Path

from dreamer_arm.envs.arms import Arm

_ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "i2rt_yam"

YAM_ARM = Arm(
    name="yam",
    scene_path=_ASSET_DIR / "scene.xml",
    # 6 arm joints + 2 finger DOFs (left_finger, right_finger).  The right
    # finger is constrained by an equality joint and is NOT an actuator, but
    # it still has its own qpos slot that the keyframe must cover.
    home_qpos=(0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0, 0.0),
    # 6 arm actuators + 1 gripper actuator (no right_finger actuator).
    home_ctrl=(0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0),
    tcp_site="grasp_site",
    arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    arm_actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    gripper_actuator="gripper",
    gripper_closed=0.0,
    gripper_open=0.041,
    ee_step_m=0.02,
    max_joint_step=0.3,
)
