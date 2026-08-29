"""Sawyer arm — no-op control seam.

Sawyer uses Meta-World's default mocap (kinematic) actuation.
Leaving the external hooks unset lets the upstream code run:
``set_xyz_action + do_simulation`` for actuation, the default ``_reset_hand``
mocap-servo for resets, and native claw-body aperture sensing.

This class exists only to satisfy the ``Arm`` protocol so the factory code can
treat both arms uniformly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dreamer_arm.envs.control.metrics import ServoState
from dreamer_arm.envs.sim.arms.base import ArmConfig


class SawyerArm:
    """No-op arm that defers to Meta-World's default Sawyer mocap path."""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "sawyer"

    @property
    def last_diagnostics(self) -> Mapping[str, float] | None:
        return None

    @property
    def servo_state(self) -> ServoState | None:
        return None

    def attach(self, env: Any) -> None:
        """Do nothing — leave hooks unset so the default mocap path runs."""

    def actuate(self, env: Any, action: Any) -> None:
        """Never called (hook not installed)."""

    def reset_hand(self, env: Any, steps: int) -> None:
        """Never called (hook not installed)."""

    def gripper_open(self, env: Any) -> float:
        """Return Meta-World's native normalized claw separation."""
        left = env.data.body("leftclaw").xpos
        right = env.data.body("rightclaw").xpos
        return float(np.clip(np.linalg.norm(right - left) / 0.1, 0.0, 1.0))
