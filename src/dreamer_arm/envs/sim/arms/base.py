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

    ee_step_m: float = 0.05
    """Per-action increment (m) to the accumulated TCP setpoint when action
    magnitude is 1.0 — not a per-step displacement bound; see max_lead_m."""

    damping: float = 0.10
    """DLS regularisation λ on the length-scaled task Jacobian (dimensionless
    singular values, see length_scale), where the YAM home spectrum's
    smallest value is ≈0.16 — chosen via the controller bench to damp only
    near-degenerate directions while smoothing the
    overshoot/reversal transients a reversing policy induces."""

    nullspace_gain: float = 1.0
    """Posture-bias gain; pulls joints toward home configuration."""

    ori_gain: float = 1.0
    """Scale on orientation error in the DLS task vector."""

    joint_margin: float = 0.05
    """Soft joint-limit margin (rad)."""

    max_joint_step: float = 1.0
    """Per-step cap on max |dq_i| (rad), applied separately to the task and
    posture-bias components (so the combined dq can reach up to 2x this in
    the worst case); preserves IK direction, prevents near-singular or
    orientation-dominated steps from slamming joints to limits.  Chosen via
    sweep: the MuJoCo position servo only realises a few percent of a
    commanded joint increment per 12.5ms control step, so a setpoint cap
    anywhere near typical steady-state joint velocities starves the servo of
    the lead it needs — this is a setpoint-target bound, not a realised
    velocity bound."""

    max_lead_m: float = 0.25
    """Leash (metres) bounding how far the accumulated TCP setpoint may lead
    the actual TCP — the real per-step task-error bound (ee_step_m is only
    the per-action increment to that setpoint).  Chosen via sweep."""

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
