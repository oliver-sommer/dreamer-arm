"""Arm descriptors and registry for the arm-swappable manipulation framework.

Each arm is a frozen :class:`Arm` dataclass that supplies the :class:`EEController`
and :class:`Manipulation` env with everything they need to be arm-agnostic:
which joints drive the IK, where the TCP site is, how the gripper maps, etc.

Usage::

    from dreamer_arm.envs.arms import get_arm

    arm = get_arm("yam")  # or "sawyer"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Arm:
    """Everything the Manipulation env + IK controller need from one arm.

    Parameters
    ----------
    name:
        Unique string identifier matching the ``arm=...`` config flag.
    scene_path:
        Path to the arm's scene MJCF (includes the robot model, floor, and
        lighting).  Task bodies are spliced in on top via MjSpec.
    home_qpos:
        Home joint positions in model order (arm joints + gripper DOFs).
        Must exactly match the ``qpos`` length once the arm is compiled *without*
        any task bodies.
    home_ctrl:
        Home actuator ctrl values (position actuator targets).
    tcp_site:
        Name of the TCP (tool-centre-point) site used by the IK Jacobian and
        task rewards.
    arm_joint_names:
        Ordered names of the **arm** joints fed into the DLS-IK.  Does NOT
        include the gripper joint.
    arm_actuator_names:
        Actuator names in the same order as ``arm_joint_names``.  For
        position-actuated arms these usually match the joint names.
    gripper_actuator:
        Name of the gripper position actuator.
    gripper_closed:
        Ctrl value when the gripper is fully closed.
    gripper_open:
        Ctrl value when the gripper is fully open.
    ee_step_m:
        Maximum end-effector displacement per *controlled* step (metres).
        The raw action ``∈ [-1,1]`` is scaled by this factor before the IK step.
    max_joint_step:
        Hard ceiling on |dq_i| per control step (rad).  The raw DLS solution
        is scaled uniformly so the largest component never exceeds this value.
        Set to 0.0 (default) to disable the clamp.
    ik_damping:
        Base damped-least-squares damping λ (rad·s).  Larger → more stable near
        singularities but looser tracking.  Always applied.
    ik_damping_max:
        Extra damping λ added on top of ``ik_damping`` when the arm approaches a
        singularity (Nakamura adaptive damping; ramps in as ``sigma_min`` falls
        below ``ik_damping_sigma0``).
    ik_damping_sigma0:
        Singular-value threshold below which the adaptive damping ramps in.  Set
        to 0.0 to disable adaptive damping (constant ``ik_damping`` only).
    tcp_approach_axis:
        Local axis index (0=x, 1=y, 2=z) of the TCP site that points along the
        gripper's approach direction.  The controller regulates this axis to
        ``ori_target_axis`` in world frame, leaving roll about it free — this
        avoids pinning the wrist at its limits.  ``None`` disables down-axis
        regulation (falls back to full-orientation tracking of ``tcp_target_quat``).
    ori_target_axis:
        World-frame unit vector the approach axis is driven toward (default
        straight down, ``(0, 0, -1)`` = gripper-down).
    """

    name: str
    scene_path: Path
    home_qpos: tuple[float, ...]
    home_ctrl: tuple[float, ...]
    tcp_site: str
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    gripper_actuator: str
    gripper_closed: float
    gripper_open: float
    ee_step_m: float = 0.02
    max_joint_step: float = 0.0
    # Damped-least-squares IK damping (see field docstrings above).
    ik_damping: float = 5e-3
    ik_damping_max: float = 0.0
    ik_damping_sigma0: float = 0.0
    # Down-axis orientation regulation (see field docstrings above).
    tcp_approach_axis: int | None = None
    ori_target_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Optional world-frame TCP workspace box ``((xlo, ylo, zlo), (xhi, yhi, zhi))``.
    # When set, the controller clamps the commanded EE *target* into this box each
    # step (outward motion at a face is zeroed, inward motion is free).  This keeps
    # the policy from driving the TCP below the table, behind the base, or out
    # past the reachable workspace into the retract-fold singularity — without a
    # base re-mount (an empirical base sweep showed moving the mount only shrinks
    # forward reach).  ``None`` disables the clamp.
    workspace_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    # Optional hook called by Manipulation._build_spec after loading the arm
    # scene but before task bodies are spliced in.  Use this to patch the raw
    # vendor assets at load time (e.g. attach a gripper to a bare arm model).
    patch_spec: Callable[[Any], None] | None = field(default=None, compare=False, hash=False)
    # Optional fixed TCP target orientation (w, x, y, z).  None → capture from
    # the arm's 'home' keyframe at controller construction time.  Only used when
    # tcp_approach_axis is None (full-orientation fallback).
    tcp_target_quat: tuple[float, float, float, float] | None = None


def get_arm(name: str) -> Arm:
    """Return the :class:`Arm` descriptor for *name*.

    Supported names: ``"yam"``, ``"sawyer"``.
    """
    if name == "yam":
        from dreamer_arm.envs.arms.yam import YAM_ARM

        return YAM_ARM
    if name == "sawyer":
        from dreamer_arm.envs.arms.sawyer import SAWYER_ARM

        return SAWYER_ARM
    raise ValueError(f"Unknown arm: {name!r}.  Supported: 'yam', 'sawyer'.")
