"""Smoke tests for the Meta-World env factory.

These are marked ``slow`` because MuJoCo + Meta-World take noticeable time to
import and compile models on first use. They run only with ``pixi run pytest
-m slow``; the default ``test`` task skips them.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Single-task Meta-World (MT1)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metaworld_single_task_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:door-open", seed=0, time_limit=10)
    obs, info = env.reset()
    assert set(obs.keys()) >= {"scene", "state", "is_first", "is_last", "is_terminal"}
    assert "task_id" not in obs  # single-task: no one-hot
    assert obs["scene"].dtype == np.uint8
    assert obs["scene"].shape == (64, 64, 3)
    assert obs["state"].shape == (39,)  # Meta-World's fixed-size obs
    assert obs["is_first"]
    assert info.get("task") == "door-open"
    # Sawyer EE action: (x, y, z, gripper)
    assert env.action_space.shape == (4,)

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


# ---------------------------------------------------------------------------
# Multi-task one-hot conditioning
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_metaworld_task_id_one_hot() -> None:
    """A pinned task index emits a one-hot ``task_id`` of length num_tasks."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:door-open", seed=0, time_limit=5, task_idx=3, num_tasks=10)
    obs, _ = env.reset()
    assert "task_id" in obs
    assert obs["task_id"].shape == (10,)
    assert int(np.argmax(obs["task_id"])) == 3
    assert obs["task_id"].sum() == pytest.approx(1.0)
    # task_id is carried through step unchanged (task is pinned per env).
    obs, *_ = env.step(env.action_space.sample())
    assert int(np.argmax(obs["task_id"])) == 3
    env.close()


@pytest.mark.slow
def test_metaworld_mt_env_num_guard() -> None:
    """env_num not a multiple of the task count raises a clear error."""
    from dreamer_arm.envs.factory import make_vector_env

    with pytest.raises(ValueError, match="multiple"):
        make_vector_env("metaworld:MT10", num_envs=7)


# ---------------------------------------------------------------------------
# ActionRatePenalty wrapper
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_action_rate_penalty_zero_cost_is_noop() -> None:
    """rate_cost=0 produces identical rewards to the unwrapped env."""
    from dreamer_arm.envs.factory import make_env

    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, size=(4,)).astype(np.float32) for _ in range(5)]

    env_base = make_env("metaworld:reach", seed=0, time_limit=10)
    env_pen = make_env("metaworld:reach", seed=0, time_limit=10, action_rate_cost=0.0)
    env_base.reset(seed=0)
    env_pen.reset(seed=0)

    for act in actions:
        _, r_base, term_b, trunc_b, _ = env_base.step(act)
        _, r_pen, _, _, _ = env_pen.step(act)
        assert r_base == pytest.approx(r_pen), "zero cost must produce identical rewards"
        if term_b or trunc_b:
            break

    env_base.close()
    env_pen.close()


@pytest.mark.slow
def test_action_rate_penalty_no_penalty_on_first_step() -> None:
    """No jerk penalty is applied on the very first step of an episode."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:reach", seed=0, time_limit=10, action_rate_cost=1.0)
    env.reset(seed=0)
    act = np.ones(4, dtype=np.float32)
    _, reward, _, _, info = env.step(act)

    assert info["action_rate_cost"] == pytest.approx(0.0), "first step must incur no rate penalty"
    assert reward == pytest.approx(info["task_reward"])
    env.close()


@pytest.mark.slow
def test_action_rate_penalty_constant_action_zero_rate_cost() -> None:
    """Repeating the same action incurs no rate penalty after the first step."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:reach", seed=0, time_limit=10, action_rate_cost=1.0)
    env.reset(seed=0)
    act = np.full(4, 0.5, dtype=np.float32)

    env.step(act)  # first step: no prev action, skip
    _, reward, _, _, info = env.step(act)  # same action again → Δa = 0

    assert info["action_rate_cost"] == pytest.approx(0.0), (
        "constant action must have zero rate cost"
    )
    assert reward == pytest.approx(info["task_reward"])
    env.close()


@pytest.mark.slow
def test_action_rate_penalty_alternating_action_incurs_cost() -> None:
    """Alternating between opposite actions produces a positive jerk penalty."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("metaworld:reach", seed=0, time_limit=10, action_rate_cost=1.0)
    env.reset(seed=0)
    pos = np.ones(4, dtype=np.float32)
    neg = -np.ones(4, dtype=np.float32)

    env.step(pos)  # first step: baseline prev_action = pos
    _, reward, _, _, info = env.step(neg)  # Δa = neg - pos = -2 each → sum((Δa)²) = 4*4 = 16

    expected_cost = 1.0 * float(np.sum((neg - pos) ** 2))
    assert info["action_rate_cost"] == pytest.approx(expected_cost)
    assert reward == pytest.approx(info["task_reward"] - expected_cost)
    env.close()


# ---------------------------------------------------------------------------
# Visual domain randomization
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scene_randomize_changes_frames_and_model_arrays() -> None:
    """scene_randomize=True must change mat_rgba and the rendered image each reset."""
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld("reach", scene_randomize=True, seed=1)
    model = env._env.model

    env.reset()
    rgba_a = model.mat_rgba.copy()
    frame_a = env.render().copy()

    env.reset()
    rgba_b = model.mat_rgba.copy()
    frame_b = env.render().copy()

    # At least one material RGBA must differ between resets.
    assert not np.allclose(rgba_a, rgba_b), "mat_rgba should differ across DR resets"
    # Rendered images must differ (RGBA tint + possible texture swap).
    assert np.mean(np.abs(frame_a.astype(float) - frame_b.astype(float))) > 1.0, (
        "rendered scene frames should differ when scene_randomize=True"
    )
    env.close()


@pytest.mark.slow
def test_camera_pose_randomize_changes_pose_and_frames() -> None:
    """camera_pose_randomize=True must change cam_pos/cam_quat and the rendered image."""
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld("reach", camera_pose_randomize=True, camera_jitter=0.0, seed=2)
    model = env._env.model
    cam_id = env._camera_id

    env.reset()
    pos_a = model.cam_pos[cam_id].copy()
    quat_a = model.cam_quat[cam_id].copy()
    frame_a = env.render().copy()

    env.reset()
    pos_b = model.cam_pos[cam_id].copy()
    quat_b = model.cam_quat[cam_id].copy()
    frame_b = env.render().copy()

    # Camera position must differ between resets.
    assert not np.allclose(pos_a, pos_b), "cam_pos should differ across pose-randomize resets"
    # Camera orientation must also differ.
    assert not np.allclose(quat_a, quat_b), "cam_quat should differ across pose-randomize resets"

    # Each sampled position must be on the near/behind-arm side (y < arm base + margin).
    _ARM_BASE_Y = 0.23
    _MARGIN = 0.5  # allow up to 0.5 m in front of arm base (generous for wide azimuth)
    for pos in (pos_a, pos_b):
        assert pos[1] < _ARM_BASE_Y + _MARGIN, (
            f"camera y={pos[1]:.3f} should be on the near side (< {_ARM_BASE_Y + _MARGIN})"
        )

    # Rendered images must differ.
    assert np.mean(np.abs(frame_a.astype(float) - frame_b.astype(float))) > 1.0, (
        "rendered scene frames should differ when camera_pose_randomize=True"
    )
    env.close()


@pytest.mark.slow
def test_dr_flags_off_leave_model_arrays_unchanged() -> None:
    """With both DR flags False, cam_pos and mat_rgba must not change across resets."""
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld(
        "reach", scene_randomize=False, camera_pose_randomize=False, camera_jitter=0.0, seed=3
    )
    model = env._env.model
    cam_id = env._camera_id

    env.reset()
    pos_a = model.cam_pos[cam_id].copy()
    rgba_a = model.mat_rgba.copy()

    env.reset()
    pos_b = model.cam_pos[cam_id].copy()
    rgba_b = model.mat_rgba.copy()

    assert np.allclose(pos_a, pos_b), "cam_pos must not change when DR is disabled"
    assert np.allclose(rgba_a, rgba_b), "mat_rgba must not change when DR is disabled"
    env.close()


# ---------------------------------------------------------------------------
# YAM singularity robustness
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_yam_ee_controller_bounded_and_progressing_near_singularity() -> None:
    """DLS IK stays bounded *and* keeps moving at a near-singular wrist pose.

    Drives joint5 to within 0.01 rad of its +π/2 wrist-singularity limit, then
    applies 20 max-amplitude random actions.  Asserts:

    1. Damping keeps the solve well-posed: ``sigma_min`` of the DLS-solvable
       system stays finite and ``dq`` never contains NaNs/Infs.
    2. The step clamp holds: every step satisfies ``dq_max ≤ max_joint_step``.
    3. No lock: the committed joint target changes across the rollout (the arm
       does not freeze in place), which was the failure mode of the undamped
       pseudo-inverse near this singularity.
    """
    import mujoco

    from dreamer_arm.envs.arms import get_arm
    from dreamer_arm.envs.control import EEController

    arm = get_arm("yam")
    spec = mujoco.MjSpec.from_file(str(arm.scene_path))
    model = spec.compile()

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1.0, 1.0, size=(20, 4))
    j5_idx = list(arm.arm_joint_names).index("joint5")

    ctrl = EEController(arm, model)
    data = mujoco.MjData(model)
    key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # Place joint5 near its +π/2 upper limit — classic YAM wrist singularity.
    data.qpos[ctrl._arm_qpos_adrs[j5_idx]] = np.pi / 2 - 0.01
    mujoco.mj_forward(model, data)

    q_targets = []
    for i, a in enumerate(actions):
        ctrl.apply(a, model, data)
        dq_max = ctrl.last_diag["dq_max"]
        assert np.isfinite(dq_max), f"step {i}: non-finite dq near singularity"
        assert dq_max <= arm.max_joint_step + 1e-9, (
            f"step {i}: dq_max={dq_max:.4f} exceeded max_joint_step={arm.max_joint_step} "
            f"(sigma_min={ctrl.last_diag['sigma_min']:.4f})"
        )
        q_targets.append(np.array([data.ctrl[aid] for aid in ctrl._arm_act_ids]))
        mujoco.mj_step(model, data)

    # No lock: the commanded joint target must actually change over the rollout.
    q_targets = np.array(q_targets)
    assert np.ptp(q_targets, axis=0).max() > 1e-3, (
        "arm appears locked near the wrist singularity (joint targets never moved)"
    )


@pytest.mark.slow
def test_yam_mw_no_singularity_lock() -> None:
    """A random rollout in the real MW env does not lock or saturate the wrist.

    Reproduces the original failure (near-singular home + wrist pinned at ±π/2 →
    self-collision back-off freezes the arm).  After the down-axis + DLS +
    re-posed-home fix the arm should stay well-conditioned: very few stuck steps,
    the wrist off its limits, and ``sigma_min`` above the near-singular alarm.
    """
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld("reach", arm="yam", seed=0)
    ctrl = env._env._yam_controller
    qadr = ctrl._arm_qpos_adrs
    j4 = list(ctrl._arm.arm_joint_names).index("joint4")
    j5 = list(ctrl._arm.arm_joint_names).index("joint5")

    env.reset()
    rng = np.random.default_rng(0)
    prev = ctrl.tcp_pos(env._env.data).astype(np.float64)
    stuck = 0
    wrist_sat = 0
    sigma_mins = []
    n = 200
    for _ in range(n):
        env.step(rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32))
        cur = ctrl.tcp_pos(env._env.data).astype(np.float64)
        if np.linalg.norm(cur - prev) < 1e-4:
            stuck += 1
        prev = cur
        sigma_mins.append(ctrl.last_diag["sigma_min"])
        q4 = env._env.data.qpos[qadr[j4]]
        q5 = env._env.data.qpos[qadr[j5]]
        if abs(q4) > np.pi / 2 - 0.05 or abs(q5) > np.pi / 2 - 0.05:
            wrist_sat += 1
    env.close()

    sigma_mins = np.array(sigma_mins)
    # Baselines before the fix were ~41/200 stuck and ~106/300 wrist-saturated.
    assert stuck <= 0.05 * n, f"arm locked on {stuck}/{n} steps (expected ~0)"
    assert wrist_sat <= 0.10 * n, f"wrist saturated on {wrist_sat}/{n} steps"
    assert np.mean(sigma_mins < 0.03) <= 0.10, (
        f"near-singular on {np.mean(sigma_mins < 0.03):.0%} of steps"
    )


# ---------------------------------------------------------------------------
# YAM self-collision gate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_yam_self_collision_gate_home_pose_safe() -> None:
    """EEController._self_collides returns False for the YAM home pose."""
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld("reach", arm="yam")
    controller = env._env._yam_controller
    env.reset()

    # After reset the arm is at home pose; qpos[arm_qpos_adrs] = home config.
    q_home = np.array([env._env.data.qpos[a] for a in controller._arm_qpos_adrs])
    assert not controller._self_collides(q_home), (
        "YAM home pose must not register as arm self-collision"
    )
    env.close()


@pytest.mark.slow
def test_yam_self_collision_gate_rollout_invariant() -> None:
    """After each step the arm qpos never triggers self-collision (gate invariant).

    The self-collision gate inside ``EEController.apply`` must ensure that the
    committed joint target is always collision-free.  We verify this by running
    50 random-action steps and calling ``_self_collides`` on the resulting arm
    qpos after each step.
    """
    from dreamer_arm.envs.metaworld import MetaWorld

    rng = np.random.default_rng(42)
    env = MetaWorld("reach", arm="yam")
    controller = env._env._yam_controller

    env.reset()
    for step in range(50):
        action = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
        env.step(action)
        q_arm = np.array([env._env.data.qpos[a] for a in controller._arm_qpos_adrs])
        assert not controller._self_collides(q_arm), (
            f"arm link self-penetration detected at step {step}: q={q_arm}"
        )
    env.close()
