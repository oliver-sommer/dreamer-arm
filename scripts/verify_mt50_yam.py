"""Coverage test: all 50 MT50 tasks build, step, and pass reachability on arm=yam.

Run from the repo root:
    pixi run python scripts/verify_mt50_yam.py
    pixi run python scripts/verify_mt50_yam.py --diagnose   # extra residual info on failures
    pixi run python scripts/verify_mt50_yam.py --no-ori     # debug: zero ori_gain (dishonest)

Outputs a PASS / REACH-FAIL / ERROR table.

REACH-FAIL means the task loads/steps cleanly but the YAM arm can't get within
REACH_THRESHOLD_M of the primary object using a scripted top-down approach.
"""

from __future__ import annotations

import sys

import numpy as np

from dreamer_arm.envs.factory import _mt_task_names

REACH_THRESHOLD_M = 0.08  # metres; within this = reachable
SETTLE_STEPS = 10  # no-op steps to let objects come to rest before capturing obj_pos
MAX_REACH_STEPS = 200  # total scripted-approach budget (3 phases)
WAYPOINT_RADIUS = 0.02  # m; waypoint considered "reached" at this distance
RISE_Z = 0.40  # m; Phase 0 — rise to this height to escape home singularity
ABOVE_OFFSET_M = 0.15  # m; Phase 1 — how far above obj_pos the lateral waypoint is placed

DIAGNOSE = "--diagnose" in sys.argv
# --no-ori: debug flag — disables orientation gain so the IK only tracks position.
# Makes PASS counts look better but is dishonest about training behaviour.
# Default OFF so PASS means genuinely reachable with ori_gain=0.1 (training default).
NO_ORI = "--no-ori" in sys.argv


# ---------------------------------------------------------------------------


def _scripted_reach(env: object, obj_pos: np.ndarray) -> tuple[float, np.ndarray]:
    """Servo TCP toward obj_pos using a 3-phase top-down approach.

    Phase 0: rise to RISE_Z at current xy — escapes the near-singular home config and
             allows free lateral movement at height.
    Phase 1: traverse xy to (obj_x, obj_y) at RISE_Z — approach from above without
             knocking the object sideways.
    Phase 2: descend directly onto obj_pos.

    Returns (best_dist_m, best_tcp_pos) — minimum TCP-to-obj_pos distance seen.
    """
    init_tcp = np.asarray(env._env.tcp_center, dtype=np.float64)
    above_z = float(max(obj_pos[2] + ABOVE_OFFSET_M, RISE_Z))

    # Waypoints in order
    waypoints = [
        np.array([init_tcp[0], init_tcp[1], RISE_Z], dtype=np.float64),  # Phase 0: rise
        np.array([obj_pos[0], obj_pos[1], above_z], dtype=np.float64),  # Phase 1: lateral
        obj_pos.copy(),  # Phase 2: descend
    ]
    wp_idx = 0

    best_dist = np.inf
    best_tcp = init_tcp.copy()

    for _ in range(MAX_REACH_STEPS):
        tcp = np.asarray(env._env.tcp_center, dtype=np.float64)

        # Advance to next waypoint when close enough
        while wp_idx < len(waypoints) - 1:
            if float(np.linalg.norm(waypoints[wp_idx] - tcp)) < WAYPOINT_RADIUS:
                wp_idx += 1
            else:
                break

        target = waypoints[wp_idx]
        delta = target - tcp

        dist_to_obj = float(np.linalg.norm(obj_pos - tcp))
        if dist_to_obj < best_dist:
            best_dist = dist_to_obj
            best_tcp = tcp.copy()
        if best_dist < REACH_THRESHOLD_M:
            break

        direction = np.clip(delta / (np.linalg.norm(delta) + 1e-8) * 5, -1, 1)
        env.step(np.array([*direction, -1.0], dtype=np.float32))

    return best_dist, best_tcp


def test_task(task: str) -> tuple[str, str]:
    """Returns (status, detail) where status ∈ PASS | REACH-FAIL | ERROR."""
    try:
        from dreamer_arm.envs.metaworld import MetaWorld

        env = MetaWorld(name=task, arm="yam", seed=0)
        env.reset()

        # Settle: a few no-op open-gripper steps so free-floating objects come to rest
        settle = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(SETTLE_STEPS):
            env.step(settle)

        # Capture object position AFTER settle (stable; not disturbed by random actions)
        obj_pos = np.array(env._env._get_pos_objects()[:3], dtype=np.float64)

        # Optionally disable orientation gain (--no-ori flag) for debugging only.
        # Default: leave ori_gain at the training value (0.1) so PASS is honest.
        if NO_ORI and hasattr(env._env, "_yam_controller"):
            env._env._yam_controller._ori_gain = 0.0

        best_dist, best_tcp = _scripted_reach(env, obj_pos)

        detail = ""
        if DIAGNOSE and best_dist >= REACH_THRESHOLD_M:
            residual = obj_pos - best_tcp
            axis = ["x", "y", "z"][int(np.argmax(np.abs(residual)))]
            # Check joint saturation
            mw_env = env._env
            ctrl = np.asarray(mw_env.data.ctrl)
            lo = mw_env.model.actuator_ctrlrange[:, 0]
            hi = mw_env.model.actuator_ctrlrange[:, 1]
            saturated = int(np.sum((ctrl <= lo + 1e-3) | (ctrl >= hi - 1e-3)))
            detail = (
                f"dist={best_dist:.3f}m  obj={np.round(obj_pos, 3)}  "
                f"tcp={np.round(best_tcp, 3)}  residual={np.round(residual, 3)}  "
                f"dom_axis={axis}  sat_joints={saturated}"
            )
        elif best_dist >= REACH_THRESHOLD_M:
            detail = f"dist={best_dist:.3f}m"

        env.close()

        return ("PASS", "") if best_dist < REACH_THRESHOLD_M else ("REACH-FAIL", detail)

    except Exception as e:
        return "ERROR", str(e)[:120]


def main() -> None:
    tasks = _mt_task_names("MT50")
    mode = " [DIAGNOSE]" if DIAGNOSE else ""
    print(f"Testing {len(tasks)} MT50 tasks with arm=yam  (threshold={REACH_THRESHOLD_M}m){mode}\n")

    results: list[tuple[str, str, str]] = []
    for task in tasks:
        status, detail = test_task(task)
        results.append((status, task, detail))
        marker = "✓" if status == "PASS" else ("~" if status == "REACH-FAIL" else "✗")
        print(f"  {marker} {status:12s}  {task:<35s}  {detail}")

    print(f"\n{'=' * 70}")
    for s in ("PASS", "REACH-FAIL", "ERROR"):
        items = [r for r in results if r[0] == s]
        if items:
            print(f"{s} ({len(items)}): {', '.join(r[1] for r in items)}")

    errors = [r for r in results if r[0] == "ERROR"]
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
