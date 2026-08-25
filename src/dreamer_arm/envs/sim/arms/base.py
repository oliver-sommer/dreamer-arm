"""Arm protocol and factory.

Each ``Arm`` implementation installs Meta-World's two injectable hooks
(``_external_actuation`` / ``_external_reset_hand``) on a task env instance.
The hook API is a fixed contract defined by the fork:

* ``_external_actuation(env, action) -> None``: fully advances physics.
* ``_external_reset_hand(env, steps) -> None``: must set ``env.init_tcp``.

``SawyerArm`` leaves both hooks unset so the default mocap path runs.
``YamArm`` installs full DLS-IK actuation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from dreamer_arm.envs.control.metrics import ServoState


@dataclass(frozen=True)
class ArmConfig:
    """Tuning knobs forwarded to the arm implementation."""

    name: str
    """Arm identifier: ``"yam"`` or ``"sawyer"``."""

    max_ee_speed_m_s: float = 0.35
    """Maximum Cartesian speed per axis at action magnitude 1.0 (m/s)."""

    max_lag_m: float = 0.035
    """Maximum retained-target tracking error per Cartesian axis (m).

    This anti-windup leash preserves contact authority without allowing an
    obstructed or unreachable target to drift arbitrarily far from the arm.
    The default matches ``max_ee_speed_m_s * joint_target_horizon_s``.
    """

    workspace_low: tuple[float, float, float] | None = None
    """Fallback Cartesian workspace lower bound ``(x, y, z)`` in metres.

    YAM prefers the attached Meta-World environment's ``mocap_low`` bound;
    this value is used only when the environment exposes no bound.
    ``None`` together with ``workspace_high=None`` leaves it unbounded.
    """

    workspace_high: tuple[float, float, float] | None = None
    """Fallback Cartesian workspace upper bound ``(x, y, z)`` in metres.

    Must be supplied together with ``workspace_low``.  See that field for
    environment-first resolution semantics.
    """

    damping: float = 0.10
    """DLS regularisation λ on the length-scaled task Jacobian (dimensionless
    singular values, see length_scale), where the YAM home spectrum's
    smallest value is ≈0.16 — chosen via the controller bench to damp only
    near-degenerate directions while smoothing the
    overshoot/reversal transients a reversing policy induces."""

    nullspace_gain: float = 1.0
    """Posture-bias rate (1/s); pulls joints toward home configuration."""

    ori_gain: float = 1.0
    """Orientation feedback gain (1/s)."""

    max_ori_speed_rad_s: float = 1.0
    """Angular-speed cap for orientation feedback (rad/s)."""

    ori_weight: float = 0.0
    """Relative priority of orientation versus unit-scaled translation in
    the DLS objective.  Unlike ori_gain, zero truly disables the rotational
    constraint.  The YAM default is position-only because its restricted
    wrist cannot preserve the home orientation across the task workspace."""

    joint_margin: float = 0.05
    """Soft joint-limit margin (rad)."""

    max_joint_speed_rad_s: float = 3.0
    """Maximum joint speed used to bound the IK lookahead (rad/s).

    The task and posture components are capped separately before summing, so
    their combined implied speed can reach twice this value in the worst case.
    """

    joint_target_horizon_s: float = 0.10
    """Time scale used for nullspace gain and joint-target step limits.

    Cartesian translation uses the retained target's tracking error instead;
    ``max_lag_m`` now bounds that servo lead directly.
    """

    length_scale: float = 0.25
    """Characteristic length (metres) used to make the DLS Jacobian's
    position and orientation rows unit-consistent; see IKConfig."""

    settle_steps: int = 200
    """Physics sub-steps used to settle the arm during ``reset_hand``."""


class Arm(Protocol):
    """Protocol for arm control seam implementations."""

    @property
    def name(self) -> str: ...

    @property
    def last_diagnostics(self) -> Mapping[str, float] | None: ...

    @property
    def servo_state(self) -> ServoState | None: ...

    def attach(self, env: Any) -> None: ...

    def actuate(self, env: Any, action: Any) -> None: ...

    def reset_hand(self, env: Any, steps: int) -> None: ...


def make_arm(name: str, cfg: ArmConfig | None = None) -> Arm:
    from dreamer_arm.envs.sim.arms.sawyer import SawyerArm
    from dreamer_arm.envs.sim.arms.yam import YamArm

    if cfg is None:
        cfg = ArmConfig(name=name)
    if name == "yam":
        return YamArm(cfg)
    if name == "sawyer":
        return SawyerArm(cfg)
    raise ValueError(f"Unknown arm: {name!r}. Expected 'yam' or 'sawyer'.")
