from __future__ import annotations

from typing import Any

import numpy as np

_SIGMA_MIN_HOME_FLOOR = 0.12


def _make_yam_inner(task_name: str = "reach") -> tuple[Any, Any, int]:
    import metaworld
    import mujoco

    from dreamer_arm.envs.sim.arms import make_arm

    metaworld.set_active_arm("yam")
    benchmark = metaworld.MT1(task_name + "-v3", seed=0)
    env_cls = next(iter(benchmark.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm("yam")
    arm.attach(inner)
    inner.set_task(benchmark.train_tasks[0])
    site_id = int(mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    return inner, arm, site_id


def test_yam_home_pose_well_conditioned() -> None:
    import mujoco

    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        model, data = inner.model, inner.data
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jacobian = np.vstack([jacp[:, arm._arm_dadr], jacr[:, arm._arm_dadr]])
        weights = np.ones(6)
        weights[:3] = 1.0 / arm._cfg.length_scale
        sigma_min = float(np.linalg.svd(jacobian * weights[:, None], compute_uv=False).min())
    finally:
        inner.close()
    assert sigma_min >= _SIGMA_MIN_HOME_FLOOR


def test_yam_tracks_commands_without_sticking() -> None:
    import mujoco

    inner, arm, site_id = _make_yam_inner()
    rng = np.random.default_rng(0)
    per_direction_track: list[float] = []
    per_direction_stuck: list[float] = []
    try:
        for _ in range(6):
            inner.reset()
            mujoco.mj_forward(inner.model, inner.data)
            previous_tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
            direction = rng.uniform(-1.0, 1.0, 3)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            action = np.array([*(0.7 * direction), -1.0], dtype=np.float32)
            ratios: list[float] = []
            for _ in range(20):
                inner.step(action)
                diagnostics = arm.last_diagnostics
                tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
                command = float(diagnostics.get("cmd_norm", 0.0)) if diagnostics else 0.0
                if command > 1e-4:
                    ratios.append(float(np.linalg.norm(tcp - previous_tcp)) / command)
                previous_tcp = tcp
            if ratios:
                per_direction_track.append(float(np.mean(ratios)))
                per_direction_stuck.append(float(np.mean([ratio < 0.25 for ratio in ratios])))
    finally:
        inner.close()

    assert per_direction_track
    assert float(np.mean(per_direction_track)) >= 0.3
    assert float(np.mean(per_direction_stuck)) <= 0.3


def test_yam_translation_priority_reaches_task_workspace() -> None:
    """A task target must not become unreachable just to preserve wrist pose."""
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        target = (np.asarray(inner.goal_space.low) + np.asarray(inner.goal_space.high)) / 2.0
        for _ in range(300):
            # Exercise the controller/physics solve without adding error from
            # the benchmark's separate velocity-to-setpoint feedback loop.
            arm._x_des = target.copy()
            inner.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        error = float(np.linalg.norm(np.asarray(inner.data.site_xpos[site_id]) - target))
    finally:
        inner.close()

    assert error < 0.02


def test_yam_exposes_servo_state_after_attach() -> None:
    inner, arm, _ = _make_yam_inner()
    try:
        state = arm.servo_state
        assert state is not None
        assert state.qpos_indices.shape == (6,)
        assert state.actuator_ids.shape == (6,)
        assert state.home_qpos.shape == (6,)
    finally:
        inner.close()
