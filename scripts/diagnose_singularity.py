"""Diagnose YAM arm singularity / DLS damping in the EEController.

Three probes:

  (a) Home-pose probe   — sigma_min / dq_max for unit translations & small
                          reorientation commands at the home keyframe pose.
                          Reveals whether the home pose itself sits near a
                          wrist singularity.

  (b) Random walk       — 300 random-action steps; tracks sigma_min trajectory,
                          dq_max blowup count, clip_active rate, backoff rate.

  (c) Damping sweep     — repeat (b) for ik_damping ∈ {5e-3, 1e-2, 5e-2, 1e-1};
                          tabulates max(dq_max) and clip-rate to find the right
                          operating point.

All probes use the MW-equivalent EEController configuration:
  ee_step_m = 0.01   (matches _YAM_MW_EE_STEP_M in metaworld.py)
  ori_gain  = 0.1    (matches metaworld.py:501)
  tcp_target_quat = gripper-down orientation from the home keyframe

Run with:  python scripts/diagnose_singularity.py
"""

from __future__ import annotations

import dataclasses
import textwrap

import mujoco
import numpy as np

from dreamer_arm.envs.arms import get_arm
from dreamer_arm.envs.control import EEController

# Match the Meta-World EEController configuration (metaworld.py:495, 501).
_MW_EE_STEP_M: float = 0.01
_MW_ORI_GAIN: float = 0.1

_WALK_STEPS: int = 300
_WALK_SEED: int = 42
_DAMPING_VALUES: tuple[float, ...] = (5e-3, 1e-2, 5e-2, 1e-1)

# dq_max threshold for a "blowup" step.  One radian per step is already large
# for a single control frame; 2 rad is clearly problematic.
_BLOWUP_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_controller(damping: float) -> tuple[mujoco.MjModel, EEController]:
    """Build a YAM EEController with the given damping in MW configuration."""
    arm_base = get_arm("yam")
    spec = mujoco.MjSpec.from_file(str(arm_base.scene_path))
    model = spec.compile()

    # Capture gripper-down TCP orientation from the home keyframe.
    scratch = mujoco.MjData(model)
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if key_id < 0:
        raise RuntimeError("No 'home' keyframe in YAM scene.")
    mujoco.mj_resetDataKeyframe(model, scratch, key_id)
    mujoco.mj_forward(model, scratch)
    tcp_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, arm_base.tcp_site))
    quat_down = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_down, scratch.site_xmat[tcp_id])

    arm_mw = dataclasses.replace(
        arm_base,
        ik_damping=damping,
        ee_step_m=_MW_EE_STEP_M,
        tcp_target_quat=(
            float(quat_down[0]),
            float(quat_down[1]),
            float(quat_down[2]),
            float(quat_down[3]),
        ),
    )
    ctrl = EEController(arm_mw, model)
    ctrl._ori_gain = _MW_ORI_GAIN
    return model, ctrl


def _reset_to_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: EEController,
) -> None:
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# (a) Home-pose probe
# ---------------------------------------------------------------------------

_HOME_ACTIONS: list[tuple[str, list[float]]] = [
    ("+x", [1, 0, 0, -1]),
    ("-x", [-1, 0, 0, -1]),
    ("+y", [0, 1, 0, -1]),
    ("-y", [0, -1, 0, -1]),
    ("+z", [0, 0, 1, -1]),
    ("-z", [0, 0, -1, -1]),
    ("ori", [0, 0, 0, -1]),  # zero-translation: only orientation error drives IK
]


def probe_home(damping: float = 5e-3) -> None:
    """Report IK stats at the home pose for each unit action direction."""
    model, ctrl = _build_controller(damping)
    data = mujoco.MjData(model)

    print(f"\n{'─' * 70}")
    print(f"(a) Home-pose probe  [ik_damping={damping:.0e}]")
    print(f"{'─' * 70}")
    print(f"{'action':>6}  {'sigma_min':>10}  {'dq_max':>9}  {'dq_norm':>9}  {'clip':>5}")
    print(f"{'─' * 70}")

    for label, a in _HOME_ACTIONS:
        _reset_to_home(model, data, ctrl)
        # Apply without stepping physics — we just want the IK response.
        ctrl.apply(np.array(a, dtype=np.float64), model, data)
        d = ctrl.last_diag
        clip = "YES" if d["clip_active"] else "no"
        print(
            f"{label:>6}  {d['sigma_min']:>10.4f}  {d['dq_max']:>9.4f}  "
            f"{d['dq_norm']:>9.4f}  {clip:>5}"
        )

    # Also check the home-pose Jacobian rank via sigma_min before any action.
    _reset_to_home(model, data, ctrl)
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, ctrl._tcp_id)
    jacp_arm = jac_p[:, ctrl._arm_dof_adrs]
    jacr_arm = jac_r[:, ctrl._arm_dof_adrs]
    J = np.vstack([jacp_arm, jacr_arm])
    svs = np.linalg.svd(J, compute_uv=False)
    print("\n  Jacobian singular values at home: " + "  ".join(f"{s:.4f}" for s in svs))
    print(f"  Condition number (smax/smin): {svs.max() / max(svs.min(), 1e-12):.1f}")


# ---------------------------------------------------------------------------
# (b) Random walk
# ---------------------------------------------------------------------------


def random_walk(damping: float = 5e-3, seed: int = _WALK_SEED) -> dict[str, float]:
    """Run a random-action walk; return summary stats."""
    model, ctrl = _build_controller(damping)
    data = mujoco.MjData(model)
    _reset_to_home(model, data, ctrl)

    rng = np.random.default_rng(seed)
    sigma_mins: list[float] = []
    dq_maxs: list[float] = []
    clips: int = 0
    backoffs: int = 0  # backoff_alpha < 1.0

    for _ in range(_WALK_STEPS):
        a = rng.uniform(-1.0, 1.0, size=4)
        ctrl.apply(a, model, data)
        mujoco.mj_step(model, data)
        d = ctrl.last_diag
        sigma_mins.append(d["sigma_min"])
        dq_maxs.append(d["dq_max"])
        if d["clip_active"]:
            clips += 1
        if d["backoff_alpha"] < 1.0:
            backoffs += 1

    dq_arr = np.array(dq_maxs)
    return {
        "damping": damping,
        "sigma_min_mean": float(np.mean(sigma_mins)),
        "sigma_min_min": float(np.min(sigma_mins)),
        "dq_max_max": float(dq_arr.max()),
        "dq_max_mean": float(dq_arr.mean()),
        "blowup_frac": float((dq_arr > _BLOWUP_THRESHOLD).mean()),
        "clip_frac": float(clips / _WALK_STEPS),
        "backoff_frac": float(backoffs / _WALK_STEPS),
    }


def probe_walk(damping: float = 5e-3) -> None:
    stats = random_walk(damping)
    print(f"\n{'─' * 70}")
    print(f"(b) Random walk [{_WALK_STEPS} steps, ik_damping={damping:.0e}]")
    print(f"{'─' * 70}")
    print(
        textwrap.dedent(f"""\
      sigma_min    mean={stats["sigma_min_mean"]:.4f}  min={stats["sigma_min_min"]:.4f}
      dq_max       max={stats["dq_max_max"]:.4f}  mean={stats["dq_max_mean"]:.4f}
      blowup frac  (dq_max>{_BLOWUP_THRESHOLD:.0f} rad): {stats["blowup_frac"]:.1%}
      clip frac    (joint limit bound): {stats["clip_frac"]:.1%}
      backoff frac (self-coll gate <1): {stats["backoff_frac"]:.1%}""")
    )


# ---------------------------------------------------------------------------
# (c) Damping sweep
# ---------------------------------------------------------------------------


def probe_sweep() -> None:
    print(f"\n{'─' * 70}")
    print(f"(c) Damping sweep  [{_WALK_STEPS} steps each, same random seed]")
    print(f"{'─' * 70}")
    print(
        f"{'damping':>10}  {'lam2':>10}  {'smin_min':>10}  "
        f"{'dq_max max':>11}  {'blowup%':>9}  {'clip%':>7}"
    )
    print(f"{'─' * 70}")
    for d in _DAMPING_VALUES:
        s = random_walk(d)
        blowup_pct = s["blowup_frac"] * 100
        clip_pct = s["clip_frac"] * 100
        lam2 = d**2
        print(
            f"{d:>10.0e}  {lam2:>10.2e}  {s['sigma_min_min']:>10.4f}  "
            f"{s['dq_max_max']:>11.4f}  {blowup_pct:>8.1f}%  {clip_pct:>6.1f}%"
        )
    print(
        "\n  Interpretation: lower blowup% + clip% with reasonable dq_max_max"
        " indicates the right damping.\n"
        f"  Current default: ik_damping=5e-3 (λ²={5e-3**2:.2e})."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("YAM arm singularity / EEController diagnostic")
    print(f"  MW config: ee_step_m={_MW_EE_STEP_M}, ori_gain={_MW_ORI_GAIN}")
    print("=" * 70)
    probe_home()
    probe_walk()
    probe_sweep()
    print("\nDone.")


if __name__ == "__main__":
    main()
