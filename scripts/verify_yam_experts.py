"""Scripted-expert acceptance suite for the YAM arm in Meta-World.

Rolls Meta-World's scripted Sawyer experts (same 4-D EE-delta action
interface) on a representative task set with ``arm="yam"`` and reports, per
task: success rate, sigma_min statistics, wrist-limit saturation, stuck steps
(commanded motion with <1 mm TCP displacement), and kinematic-sanity
violations (TCP below the table or behind the arm base).

Acceptance bar (see tasks/todo.md):
- reach and drawer-close: 3/3 success
- pick-place: object lifted above z=0.10 in >= 2/3 episodes
- push / pick-place: stuck steps ~0 (pre-fix baselines: 64 and 118 per ep)
- no episode with TCP below z=0 or behind the base

Run:  pixi run python scripts/verify_yam_experts.py [--episodes N]
"""

from __future__ import annotations

import argparse

import numpy as np

TASKS = {
    "reach": "SawyerReachV3Policy",
    "push": "SawyerPushV3Policy",
    "pick-place": "SawyerPickPlaceV3Policy",
    "door-open": "SawyerDoorOpenV3Policy",
    "drawer-open": "SawyerDrawerOpenV3Policy",
    "drawer-close": "SawyerDrawerCloseV3Policy",
    "button-press-topdown": "SawyerButtonPressTopdownV3Policy",
    "window-open": "SawyerWindowOpenV3Policy",
}

STEPS_PER_EPISODE = 200


def run_task(task: str, polname: str, episodes: int, base_y: float) -> str:
    import metaworld.policies as policies
    import mujoco

    from dreamer_arm.envs.metaworld import MetaWorld

    pol = getattr(policies, polname)()
    env = MetaWorld(task, arm="yam", seed=0)
    ctrl = env._env._yam_controller
    mw = env._env
    qadr = ctrl._arm_qpos_adrs
    names = list(ctrl._arm.arm_joint_names)
    j4, j5 = names.index("joint4"), names.index("joint5")
    gid_obj = mujoco.mj_name2id(mw.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")

    succs: list[bool] = []
    lifted: list[bool] = []
    stuck_eps: list[int] = []
    sig_all: list[float] = []
    wrist_sat: list[float] = []
    sanity_bad = 0
    for _ep in range(episodes):
        obs, _ = env.reset()
        state = obs["state"]
        prev = ctrl.tcp_pos(mw.data).astype(np.float64)
        stuck = 0
        success = False
        max_obj_z = -np.inf
        for _ in range(STEPS_PER_EPISODE):
            a = np.clip(pol.get_action(state.astype(np.float64)), -1, 1)
            obs, _r, _term, _trunc, info = env.step(a.astype(np.float32))
            state = obs["state"]
            tcp = ctrl.tcp_pos(mw.data).astype(np.float64)
            if np.linalg.norm(a[:3]) > 0.3 and np.linalg.norm(tcp - prev) < 1e-3:
                stuck += 1
            prev = tcp
            sig_all.append(ctrl.last_diag["sigma_min"])
            q4, q5 = mw.data.qpos[qadr[j4]], mw.data.qpos[qadr[j5]]
            wrist_sat.append(float(abs(q4) > np.pi / 2 - 0.05 or abs(q5) > np.pi / 2 - 0.05))
            if tcp[2] < 0.0 or tcp[1] < base_y:
                sanity_bad += 1
            if gid_obj >= 0:
                max_obj_z = max(max_obj_z, float(mw.data.geom_xpos[gid_obj][2]))
            if info.get("success"):
                success = True
                break
        succs.append(success)
        lifted.append(max_obj_z > 0.10)
        stuck_eps.append(stuck)
    env.close()

    sig = np.array(sig_all)
    return (
        f"{task:22s} success={sum(succs)}/{episodes}  lifted={sum(lifted)}/{episodes}  "
        f"sigma mean={sig.mean():.3f} min={sig.min():.3f} frac<0.03={np.mean(sig < 0.03):.0%}  "
        f"wrist_sat={np.mean(wrist_sat):.0%}  stuck/ep={np.mean(stuck_eps):.0f}  "
        f"sanity_violations={sanity_bad}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--base-y", type=float, default=0.0, help="arm-base y for the behind-base sanity check")
    args = parser.parse_args()
    for task, polname in TASKS.items():
        try:
            print(run_task(task, polname, args.episodes, args.base_y))
        except Exception as exc:  # noqa: BLE001 -- one task's failure must not abort the run
            print(f"{task:22s} ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
