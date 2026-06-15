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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass  # avoid circular; actual impls import control from envs/


@dataclass(frozen=True)
class ArmConfig:
    """Tuning knobs forwarded to the arm implementation."""

    name: str
    """Arm identifier: ``"yam"`` or ``"sawyer"``."""

    ee_step_m: float = 0.05
    """Full TCP delta (m) per agent-step when action magnitude is 1.0."""

    damping: float = 0.05
    """DLS regularisation λ (constant, bounds joint velocity near singularities)."""

    nullspace_gain: float = 1.0
    """Posture-bias gain; pulls joints toward home configuration."""

    ori_gain: float = 1.0
    """Scale on orientation error in the DLS task vector."""

    joint_margin: float = 0.05
    """Soft joint-limit margin (rad)."""

    settle_steps: int = 200
    """Physics sub-steps used to settle the arm during ``reset_hand``."""


class Arm(Protocol):
    """Protocol for arm control seam implementations."""

    @property
    def name(self) -> str: ...

    def attach(self, env: Any) -> None:
        """Resolve model ids, capture home pose, install hooks on *env*."""
        ...

    def actuate(self, env: Any, action: Any) -> None:
        """Signature matching ``_external_actuation(env, action) -> None``."""
        ...

    def reset_hand(self, env: Any, steps: int) -> None:
        """Signature matching ``_external_reset_hand(env, steps) -> None``."""
        ...


def make_arm(name: str, cfg: ArmConfig | None = None) -> Arm:
    """Instantiate the arm implementation for *name*.

    Args:
        name: ``"yam"`` or ``"sawyer"``.
        cfg:  Optional config; defaults are used if *None*.
    """
    from dreamer_arm.envs.arms.sawyer import SawyerArm
    from dreamer_arm.envs.arms.yam import YamArm

    if cfg is None:
        cfg = ArmConfig(name=name)
    if name == "yam":
        return YamArm(cfg)
    if name == "sawyer":
        return SawyerArm(cfg)
    raise ValueError(f"Unknown arm: {name!r}. Expected 'yam' or 'sawyer'.")
