from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_SIGMA_MIN_HOME_FLOOR = 0.12


def _make_yam_inner(task_name: str = "reach", arm_cfg: Any | None = None, seed: int = 0) -> tuple[Any, Any, int]:
    import metaworld
    import mujoco

    from dreamer_arm.envs.sim.arms import make_arm

    metaworld.set_active_arm("yam")
    benchmark = metaworld.MT1(task_name + "-v3", seed=seed)
    env_cls = next(iter(benchmark.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm("yam", arm_cfg)
    arm.attach(inner)
    inner.set_task(benchmark.train_tasks[0])
    site_id = int(mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    return inner, arm, site_id


def _move_yam_to(
    inner: Any,
    arm: Any,
    site_id: int,
    target: np.ndarray,
    grip: float,
    budget: int,
    max_force_ratio: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    stable = 0
    obs = inner._get_obs()
    info: dict[str, Any] = {}
    for _ in range(budget):
        delta = target - arm._p_target
        distance = float(np.linalg.norm(delta))
        requested = delta * min(1.0, 0.002 / max(distance, 1e-12))
        dt = float(inner.model.opt.timestep) * int(inner.frame_skip)
        xyz = np.clip(requested / (dt * arm._cfg.max_ee_speed_m_s), -1.0, 1.0)
        obs, _, _, _, info = inner.step(np.array([*xyz, grip], dtype=np.float32))
        forces = np.abs(inner.data.actuator_force[arm._arm_act_ids])
        limits = inner.model.actuator_forcerange[arm._arm_act_ids, 1]
        max_force_ratio[0] = max(max_force_ratio[0], float(np.max(forces / limits)))
        error = float(np.linalg.norm(np.asarray(inner.data.site_xpos[site_id]) - target))
        stable = stable + 1 if error < 0.006 else 0
        if stable >= 8:
            break
    return obs, info


def _gripper_geom_ids(inner: Any) -> set[int]:
    roots = {int(inner.model.body("leftclaw").id), int(inner.model.body("rightclaw").id)}
    bodies: set[int] = set()
    for body_id in range(inner.model.nbody):
        cursor = body_id
        while cursor > 0:
            if cursor in roots:
                bodies.add(body_id)
                break
            cursor = int(inner.model.body_parentid[cursor])
    return {geom_id for geom_id in range(inner.model.ngeom) if int(inner.model.geom_bodyid[geom_id]) in bodies}


def test_yam_gripper_opens_closes_and_has_no_internal_contacts() -> None:
    inner, arm, _ = _make_yam_inner()
    try:
        obs, _ = inner.reset()
        for _ in range(50):
            obs, *_ = inner.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        left = np.asarray(inner.data.body("leftpad").xpos)
        right = np.asarray(inner.data.body("rightpad").xpos)
        open_aperture = float(np.linalg.norm(left - right))
        open_observation = float(obs[3])

        gripper_geoms = _gripper_geom_ids(inner)
        internal_contacts = [
            contact
            for contact in inner.data.contact
            if int(contact.geom1) in gripper_geoms and int(contact.geom2) in gripper_geoms
        ]

        for _ in range(80):
            obs, *_ = inner.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        left = np.asarray(inner.data.body("leftpad").xpos)
        right = np.asarray(inner.data.body("rightpad").xpos)
        closed_aperture = float(np.linalg.norm(left - right))
        closed_observation = float(obs[3])
    finally:
        inner.close()

    assert not internal_contacts
    assert open_aperture > 0.070
    assert closed_aperture < 0.005
    assert open_observation > 0.9
    assert closed_observation < 0.1
    assert closed_observation < open_observation


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_yam_pick_place_scripted_grasp_lifts_original_object(seed: int) -> None:
    from dreamer_arm.envs.sim.arms import ArmConfig

    cfg = ArmConfig(
        name="yam",
        workspace_low=(-0.5, 0.4, 0.01),
        workspace_high=(0.5, 1.0, 0.5),
    )
    inner, arm, site_id = _make_yam_inner("pick-place", cfg, seed=seed)
    max_force_ratio = [0.0]
    try:
        obs, _ = inner.reset()
        obj_start = np.asarray(obs[4:7], dtype=np.float64).copy()
        obs, _ = _move_yam_to(
            inner,
            arm,
            site_id,
            obj_start + np.array([0.0, 0.0, 0.08]),
            -1.0,
            160,
            max_force_ratio,
        )
        obs, info = _move_yam_to(
            inner,
            arm,
            site_id,
            obj_start + np.array([0.0, 0.0, 0.003]),
            -1.0,
            100,
            max_force_ratio,
        )
        assert float(np.linalg.norm(np.asarray(inner.data.site_xpos[site_id]) - obs[4:7])) < 0.006

        for _ in range(60):
            obs, _, _, _, info = inner.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert inner.touching_main_object
        assert float(info["grasp_reward"]) > 0.97
        assert float(info["mw_v2_grasp_reward"]) < 0.5

        lift_target = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
        lift_target[2] += 0.10
        obs, info = _move_yam_to(inner, arm, site_id, lift_target, 1.0, 120, max_force_ratio)
        object_lift = float(obs[6] - obj_start[2])
    finally:
        inner.close()

    assert object_lift > 0.05
    assert float(info["grasp_success"]) == 1.0
    assert np.isfinite(float(info["mw_v2_reward"]))
    assert max_force_ratio[0] < 0.8


@pytest.mark.parametrize("task_name", ["push-back", "soccer", "stick-push", "sweep", "sweep-into"])
def test_yam_custom_caging_rewards_expose_finite_original_shadow(task_name: str) -> None:
    inner, _, _ = _make_yam_inner(task_name)
    try:
        inner.reset()
        _, reward, _, _, info = inner.step(np.zeros(4, dtype=np.float32))
    finally:
        inner.close()

    assert np.isfinite(float(reward))
    assert np.isfinite(float(info["mw_v2_reward"]))
    assert np.isfinite(float(info["mw_v2_grasp_reward"]))


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
            tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64)
            xyz = np.clip((target - tcp) / arm._cfg.max_ee_speed_m_s, -1.0, 1.0)
            inner.step(np.array([*xyz, -1.0], dtype=np.float32))
        error = float(np.linalg.norm(np.asarray(inner.data.site_xpos[site_id]) - target))
    finally:
        inner.close()

    assert error < 0.025


def test_yam_resets_stalled_setpoint_and_reverses_without_windup() -> None:
    """An unreachable command must not leave latent error after reversal."""
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        unreachable = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(40):
            inner.step(unreachable)

        before_reverse = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
        reverse = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(5):
            inner.step(reverse)
        after_reverse = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
    finally:
        inner.close()

    assert not hasattr(arm, "_x_des")
    assert after_reverse[1] - before_reverse[1] > 0.005


def test_yam_default_is_position_only() -> None:
    inner, arm, _ = _make_yam_inner()
    try:
        inner.reset()
        inner.step(np.array([0.5, 0.0, 0.0, -1.0], dtype=np.float32))
        diagnostics = arm.last_diagnostics or {}
    finally:
        inner.close()

    assert diagnostics["ori_task_norm"] == 0.0
    assert diagnostics["ori_capped"] == 0.0


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


def test_yam_action_dimension_signs_units_and_gripper_mapping() -> None:
    """XYZ is ordered velocity in m/s; positive gripper closes."""
    inner, arm, _ = _make_yam_inner()
    try:
        inner.reset()
        action = np.array([0.25, -0.5, 0.75, 1.0], dtype=np.float32)
        control_dt = float(inner.model.opt.timestep) * int(inner.frame_skip)
        inner.step(action)
        diagnostics = arm.last_diagnostics or {}

        expected_delta = action[:3] * arm._cfg.max_ee_speed_m_s * control_dt
        assert diagnostics["cmd_x"] == pytest.approx(expected_delta[0])
        assert diagnostics["cmd_y"] == pytest.approx(expected_delta[1])
        assert diagnostics["cmd_z"] == pytest.approx(expected_delta[2])
        np.testing.assert_allclose(
            [diagnostics[f"track_cmd_{axis}"] for axis in "xyz"],
            expected_delta,
            atol=1e-12,
        )
        assert diagnostics["cmd_speed_m_s"] == pytest.approx(np.linalg.norm(action[:3] * arm._cfg.max_ee_speed_m_s))
        assert inner.data.ctrl[arm._grip_act_id] == pytest.approx(0.0)
    finally:
        inner.close()


def test_yam_zero_action_holds_commanded_pose_while_loaded_on_table() -> None:
    """A retained downward target must keep the grasp site loaded and still."""
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        down = np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        for _ in range(80):
            inner.step(down)

        diagnostics = arm.last_diagnostics or {}
        assert diagnostics["ws_clamped"] == 1.0
        assert diagnostics["cmd_z"] < 0.0
        assert diagnostics["track_cmd_z"] == pytest.approx(0.0, abs=1e-12)
        target_at_contact = arm._p_target.copy()

        positions = []
        for _ in range(40):
            inner.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
            positions.append(np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy())

        np.testing.assert_allclose(arm._p_target, target_at_contact, atol=1e-12)
        # The fingertip/table contact remains loaded while the controlled grasp
        # site moves by less than half a millimetre over the hold interval.
        assert float(np.ptp(np.asarray(positions)[:, 2])) < 5e-4
    finally:
        inner.close()


def test_yam_blocked_arm_clamps_each_target_lag_axis() -> None:
    """A physically blocked arm cannot accumulate an unreachable reference."""
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        original_do_simulation = inner.do_simulation
        inner.do_simulation = lambda _ctrl, _frames: None
        action = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32)
        for _ in range(100):
            arm.actuate(inner, action)

        tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64)
        lag = arm._p_target - tcp
        assert float(np.max(np.abs(lag))) <= arm._cfg.max_lag_m + 1e-12
        diagnostics = arm.last_diagnostics or {}
        assert diagnostics["lag_clamped"] == 1.0
        assert diagnostics["target_lag_norm"] == pytest.approx(np.linalg.norm(lag))
        inner.do_simulation = original_do_simulation
    finally:
        inner.close()


def test_yam_reversal_changes_target_immediately_and_motion_within_leash_transient() -> None:
    """Reference reversal is immediate; physical reversal is bounded by the leash."""
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        original_do_simulation = inner.do_simulation
        inner.do_simulation = lambda _ctrl, _frames: None
        forward = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(100):
            arm.actuate(inner, forward)
        inner.do_simulation = original_do_simulation

        previous_target_x = float(arm._p_target[0])
        previous_tcp_x = float(inner.data.site_xpos[site_id, 0])
        reverse = np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        dt = float(inner.model.opt.timestep) * int(inner.frame_skip)
        max_unwind_steps = int(np.ceil(arm._cfg.max_lag_m / (arm._cfg.max_ee_speed_m_s * dt))) + 2

        reversed_at: int | None = None
        for step in range(1, max_unwind_steps + 1):
            inner.step(reverse)
            tcp_x = float(inner.data.site_xpos[site_id, 0])
            if step == 1:
                assert float(arm._p_target[0]) < previous_target_x
            if tcp_x < previous_tcp_x - 1e-5:
                reversed_at = step
                break
            previous_tcp_x = tcp_x

        assert reversed_at is not None
        assert reversed_at <= max_unwind_steps
    finally:
        inner.close()


def test_yam_target_resets_to_grasp_site() -> None:
    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        initial_target = arm._p_target.copy()
        for _ in range(10):
            inner.step(np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))
        assert not np.allclose(arm._p_target, initial_target)

        inner.reset()
        np.testing.assert_allclose(arm._p_target, inner.data.site_xpos[site_id], atol=1e-12)
    finally:
        inner.close()


def test_yam_targets_reset_independently_in_sync_vector_env() -> None:
    from gymnasium import spaces

    from dreamer_arm.envs.wrappers import SyncVectorEnv

    class _RawYamAdapter:
        """Render-free vector adapter around a real MuJoCo YAM environment."""

        def __init__(self) -> None:
            self.inner, self.arm, self.site_id = _make_yam_inner()
            self.observation_space = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)})
            self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

        def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            self.inner.reset(seed=seed)
            return {"proprio": np.zeros(1, dtype=np.float32)}, {}

        def step(
            self, action: np.ndarray, *, render: bool = True
        ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
            _obs, reward, terminated, truncated, info = self.inner.step(action)
            return {"proprio": np.zeros(1, dtype=np.float32)}, float(reward), terminated, truncated, info

        def close(self) -> None:
            self.inner.close()

    vec = SyncVectorEnv([_RawYamAdapter, _RawYamAdapter], action_repeat=1)
    try:
        vec.reset()
        vec.step(
            np.array(
                [
                    [1.0, 0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            )
        )
        targets_before_reset = [env.arm._p_target.copy() for env in vec._envs]

        vec.reset()
        for index, env in enumerate(vec._envs):
            assert not np.allclose(env.arm._p_target, targets_before_reset[index])
            np.testing.assert_allclose(env.arm._p_target, env.inner.data.site_xpos[env.site_id], atol=1e-12)
    finally:
        vec.close()


def test_yam_uses_metaworld_workspace_bounds_as_fallback() -> None:
    inner, arm, _ = _make_yam_inner()
    try:
        np.testing.assert_allclose(arm._ws_low, inner.mocap_low)
        np.testing.assert_allclose(arm._ws_high, inner.mocap_high)
    finally:
        inner.close()


def test_yam_workspace_config_overrides_environment() -> None:
    from dreamer_arm.envs.sim.arms import ArmConfig, make_arm

    arm = make_arm(
        "yam",
        ArmConfig(
            name="yam",
            workspace_low=(-0.1, 0.2, 0.3),
            workspace_high=(0.4, 0.5, 0.6),
        ),
    )
    env = SimpleNamespace(mocap_low=(-9.0, -9.0, -9.0), mocap_high=(9.0, 9.0, 9.0))
    low, high, source = arm._resolve_workspace_bounds(env)
    np.testing.assert_allclose(low, [-0.1, 0.2, 0.3])
    np.testing.assert_allclose(high, [0.4, 0.5, 0.6])
    assert source == "config"


def test_yam_nonfinite_joint_guard_resyncs_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamer_arm.envs.sim.arms import yam as yam_module

    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
        arm._p_target = tcp + 0.01
        qadr = int(arm._arm_qadr[0])
        saved_q = float(inner.data.qpos[qadr])
        inner.data.qpos[qadr] = np.nan
        monkeypatch.setattr(yam_module.mujoco, "mj_forward", lambda _model, _data: None)

        arm.actuate(inner, np.zeros(4, dtype=np.float32))

        np.testing.assert_allclose(arm._p_target, tcp)
        inner.data.qpos[qadr] = saved_q
    finally:
        inner.close()


def test_yam_nonfinite_tcp_guard_preserves_last_finite_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamer_arm.envs.sim.arms import yam as yam_module

    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        target = arm._p_target.copy()
        inner.data.site_xpos[site_id, 0] = np.nan
        monkeypatch.setattr(yam_module.mujoco, "mj_forward", lambda _model, _data: None)

        arm.actuate(inner, np.zeros(4, dtype=np.float32))

        np.testing.assert_allclose(arm._p_target, target)
    finally:
        inner.close()


def test_yam_nonfinite_jacobian_guard_resyncs_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamer_arm.envs.sim.arms import yam as yam_module

    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()

        def _nonfinite_jacobian(_model: Any, _data: Any, jacp: np.ndarray, jacr: np.ndarray, _site: int) -> None:
            jacp.fill(np.nan)
            jacr.fill(np.nan)

        monkeypatch.setattr(yam_module.mujoco, "mj_jacSite", _nonfinite_jacobian)
        arm.actuate(inner, np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))

        np.testing.assert_allclose(arm._p_target, tcp)
    finally:
        inner.close()


def test_yam_nonfinite_ik_target_guard_resyncs_target(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamer_arm.envs.sim.arms import yam as yam_module

    inner, arm, site_id = _make_yam_inner()
    try:
        inner.reset()
        tcp = np.asarray(inner.data.site_xpos[site_id], dtype=np.float64).copy()
        monkeypatch.setattr(
            yam_module,
            "solve_dls",
            lambda _J, _e, q, _home, _ranges, _cfg, **_kwargs: np.full_like(q, np.nan),
        )

        arm.actuate(inner, np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))

        np.testing.assert_allclose(arm._p_target, tcp)
    finally:
        inner.close()
