"""Sawyer arm descriptor for the manipulation framework.

Assets are vendored from the DeepMind MuJoCo Menagerie (Apache-2.0):
  https://github.com/google-deepmind/mujoco_menagerie/tree/main/rethink_robotics_sawyer

The menagerie model is a bare 7-DOF arm with no gripper.  A minimal parallel
gripper is attached to ``right_l6`` at load time via :func:`_patch_sawyer_spec`,
which is called by :class:`~dreamer_arm.envs.manip.Manipulation` before
compiling the scene.

Values derived from the menagerie XML
--------------------------------------
- 7 revolute arm joints ``right_j0``-``right_j6``.
- 7 actuators named ``a0``-``a6`` (``general`` with affine bias = PD ctrl).
- ``right_l6`` is the distal wrist link; ``attachment_site`` at ``pos="0 0 0.0245"``.
- ``grasp_site`` added by :func:`_patch_sawyer_spec` at the fingertip level.
- Home keyframe: ``qpos = [0, -1.18, 0, 2.18, 0, 0.57, 3.3161]`` (arm only in XML;
  MuJoCo pads the two gripper DOFs with zeros at compile time).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from dreamer_arm.envs.arms import Arm

_ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "rethink_robotics_sawyer"
_SCENE_PATH = _ASSET_DIR / "scene.xml"


def _patch_sawyer_spec(spec: mujoco.MjSpec) -> None:
    """Attach grasp_site + parallel gripper to the bare menagerie Sawyer."""
    # Find the distal wrist body.
    ee_body = next((b for b in spec.bodies if b.name == "right_l6"), None)
    if ee_body is None:
        raise RuntimeError(
            f"Sawyer scene is missing body 'right_l6' — check assets at {_ASSET_DIR}"
        )

    # TCP site between the finger tips (~0.08 m along right_l6 local z).
    ee_body.add_site(
        name="grasp_site",
        pos=[0.0, 0.0, 0.08],
        size=[0.005, 0, 0],
        rgba=[0.0, 1.0, 0.0, 1.0],
        group=4,
    )

    REACH = 0.04  # finger origin offset from wrist along z

    lf = ee_body.add_body(name="left_finger", pos=[-0.02, 0.0, REACH])
    lf.add_joint(
        name="left_finger",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-0.001, 0.02],
        armature=0.1,
        frictionloss=0.1,
    )
    lf.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.005, 0.010, 0.020],
        rgba=[0.4, 0.4, 0.7, 1.0],
        contype=1,
        conaffinity=1,
    )

    rf = ee_body.add_body(name="right_finger", pos=[0.02, 0.0, REACH])
    rf.add_joint(
        name="right_finger",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[-1, 0, 0],
        range=[-0.001, 0.02],
        armature=0.1,
        frictionloss=0.1,
    )
    rf.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.005, 0.010, 0.020],
        rgba=[0.4, 0.4, 0.7, 1.0],
        contype=1,
        conaffinity=1,
    )

    # Right finger mirrors left (data = polycoef in MjSpec API).
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_JOINT,
        name1="left_finger",
        name2="right_finger",
        data=[0, -1, 0, 0, 0],
    )

    # PD position controller: force = 100*(ctrl - q) - 10*qvel.
    spec.add_actuator(
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="left_finger",
        name="gripper",
        ctrlrange=[0.0, 0.041],
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[100, 0, 0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0, -100, -10],
    )
    # MuJoCo pads the existing home keyframe's qpos/ctrl with zeros for the
    # new gripper DOFs at compile time — no explicit keyframe update needed.


def _check_assets() -> None:
    if not _SCENE_PATH.exists():
        raise FileNotFoundError(
            f"Sawyer assets not found at {_ASSET_DIR}.\n"
            "The assets/ directory should be committed alongside the source.\n"
            "See README for provenance."
        )


SAWYER_ARM = Arm(
    name="sawyer",
    scene_path=_SCENE_PATH,
    # 7 arm DOFs + 2 gripper finger DOFs added by _patch_sawyer_spec.
    home_qpos=(0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.3161, 0.0, 0.0),
    home_ctrl=(0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.3161, 0.0),
    tcp_site="grasp_site",
    arm_joint_names=(
        "right_j0",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_j6",
    ),
    arm_actuator_names=("a0", "a1", "a2", "a3", "a4", "a5", "a6"),
    gripper_actuator="gripper",
    gripper_closed=0.0,
    gripper_open=0.041,
    ee_step_m=0.02,
    ik_damping=5e-3,
    patch_spec=_patch_sawyer_spec,
)

_check_assets()
