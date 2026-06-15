"""Tests for dreamer_arm.envs.

Fast tests (pure-numpy / mock):
  * DLS-IK boundedness near a singularity
  * DLS-IK posture bias pulls toward home
  * quat_log_error returns zero for identity error
  * Action clamp (env-side guard)
  * Gripper sign convention
  * SyncVectorEnv auto-reset / final_info / action_repeat (mock)

Slow tests (require MuJoCo — skipped by default):
  * Factory env_num % task_count guard
  * Single-task smoke (reset → obs dict shape, step returns correct schema)
  * Sticky success (success ORed across episode)
  * Multi-task one-hot task_id
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DLS-IK (pure numpy — no MuJoCo import)
# ---------------------------------------------------------------------------
from dreamer_arm.envs.control import IKConfig, quat_log_error, solve_dls


def _make_cfg(**kwargs: Any) -> IKConfig:
    defaults: dict[str, Any] = dict(
        ee_step_m=0.05, damping=0.05, nullspace_gain=1.0, ori_gain=1.0, joint_margin=0.05
    )
    defaults.update(kwargs)
    return IKConfig(**defaults)


def _random_J(rng: np.random.Generator, n: int = 6, perturb: float = 0.0) -> np.ndarray:
    """Random (6, n) Jacobian; perturb→0 makes it near-singular."""
    J = rng.normal(0, 1, (6, n))
    if perturb > 0.0:
        J[:, 1:] *= perturb  # columns 1..n-1 nearly zero → near-singular
    return J


def test_dls_bounded_near_singularity() -> None:
    """‖dq‖ must be finite and bounded even for an ill-conditioned Jacobian."""
    rng = np.random.default_rng(0)
    q = np.zeros(6)
    q_home = np.zeros(6)
    jnt_range = np.tile([-np.pi, np.pi], (6, 1))
    e = np.ones(6) * 0.05

    # Make J effectively rank-1 by zeroing out most columns
    J = np.zeros((6, 6))
    J[:, 0] = rng.normal(0, 1, 6)  # only one non-zero column → rank 1

    cfg = _make_cfg(damping=0.05, nullspace_gain=0.0)
    q_target = solve_dls(J, e, q, q_home, jnt_range, cfg)

    dq = q_target - q
    assert np.all(np.isfinite(dq)), "dq contains non-finite values at singularity"
    # With λ=0.05, ‖e‖≈√6*0.05≈0.122, DLS bound: ‖dq‖ ≤ ‖e‖/λ = ~2.45
    assert float(np.linalg.norm(dq)) < 10.0, f"‖dq‖ = {np.linalg.norm(dq):.3f} is too large"


def test_dls_posture_bias_toward_home() -> None:
    """With zero task error, dq must point toward q_home - q."""
    q = np.array([0.5, -0.3, 0.2, -0.1, 0.4, -0.2])
    q_home = np.zeros(6)
    jnt_range = np.tile([-np.pi, np.pi], (6, 1))
    e = np.zeros(6)  # no task error

    rng = np.random.default_rng(1)
    J = rng.normal(0, 1, (6, 6))
    cfg = _make_cfg(nullspace_gain=1.0)
    q_target = solve_dls(J, e, q, q_home, jnt_range, cfg)

    # With e=0, the DLS term vanishes and dq = N · (q_home - q)
    # → q_target should move toward q_home
    dist_before = float(np.linalg.norm(q_home - q))
    dist_after = float(np.linalg.norm(q_home - q_target))
    assert dist_after < dist_before, (
        f"Posture bias did not reduce distance to home: {dist_before:.4f} → {dist_after:.4f}"
    )


def test_quat_log_error_identity() -> None:
    """Orientation error between identical rotations is (near) zero."""
    # Identity rotation matrix
    mat = np.eye(3)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
    err = quat_log_error(mat.ravel(), q_target)
    assert err.shape == (3,)
    assert float(np.linalg.norm(err)) < 1e-6, f"Identity error not zero: {err}"


def test_quat_log_error_nonzero() -> None:
    """Rotation by 90° around Z gives a non-zero log error."""
    from dreamer_arm.envs.control import _mat2quat

    # 90° around Z
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    _mat2quat(mat)  # smoke: _mat2quat handles the rotated site without error
    # Target = identity
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    err = quat_log_error(mat.ravel(), q_target)
    assert float(np.linalg.norm(err)) > 0.5, f"Expected large error, got {err}"


def test_solve_dls_joint_clamp() -> None:
    """q_target must stay within joint limits including margin."""
    cfg = _make_cfg(joint_margin=0.1)
    q = np.array([1.4, -1.4, 1.4, -1.4, 1.4, -1.4])
    q_home = np.zeros(6)
    jnt_range = np.tile([-1.5, 1.5], (6, 1))
    J = np.eye(6)
    e = np.ones(6) * 0.5  # push toward limits

    q_target = solve_dls(J, e, q, q_home, jnt_range, cfg)
    lo = jnt_range[:, 0] + 0.1
    hi = jnt_range[:, 1] - 0.1
    assert np.all(q_target >= lo - 1e-9), f"Below lo+margin: {q_target}"
    assert np.all(q_target <= hi + 1e-9), f"Above hi-margin: {q_target}"


# ---------------------------------------------------------------------------
# SyncVectorEnv (mock — no MuJoCo)
# ---------------------------------------------------------------------------


def _make_mock_env(obs_keys: list[str], act_dim: int = 4) -> Any:
    """Create a minimal mock gymnasium.Env with a Dict observation space."""
    import gymnasium
    from gymnasium import spaces

    obs_space = spaces.Dict(
        {
            "scene": spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)

    env = MagicMock(spec=gymnasium.Env)
    env.observation_space = obs_space
    env.action_space = act_space

    def _reset(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs = {
            "scene": np.zeros((8, 8, 3), dtype=np.uint8),
            "proprio": np.zeros(10, dtype=np.float32),
        }
        return obs, {}

    def _step(action: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs = {
            "scene": np.ones((8, 8, 3), dtype=np.uint8),
            "proprio": np.ones(10, dtype=np.float32),
        }
        return obs, 1.0, False, False, {}

    env.reset = MagicMock(side_effect=_reset)
    env.step = MagicMock(side_effect=_step)
    env.close = MagicMock()
    return env


def _make_done_env(done_on_step: int = 1) -> Any:
    """Mock env that terminates after ``done_on_step`` steps."""
    import gymnasium
    from gymnasium import spaces

    obs_space = spaces.Dict(
        {
            "scene": spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    env = MagicMock(spec=gymnasium.Env)
    env.observation_space = obs_space
    env.action_space = act_space
    step_count = [0]

    def _reset(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        step_count[0] = 0
        return {"scene": np.zeros((8, 8, 3), np.uint8), "proprio": np.zeros(10, np.float32)}, {
            "success": False
        }

    def _step(action: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        step_count[0] += 1
        done = step_count[0] >= done_on_step
        obs = {"scene": np.zeros((8, 8, 3), np.uint8), "proprio": np.zeros(10, np.float32)}
        return obs, 1.0, done, False, {"success": done}

    env.reset = MagicMock(side_effect=_reset)
    env.step = MagicMock(side_effect=_step)
    env.close = MagicMock()
    return env


def test_sync_vec_env_reset_shape() -> None:
    """SyncVectorEnv.reset returns stacked obs with leading N axis."""
    from dreamer_arm.envs.wrappers import SyncVectorEnv

    N = 3
    env_fns = [lambda: _make_mock_env([]) for _ in range(N)]
    vec = SyncVectorEnv(env_fns, action_repeat=1)
    obs = vec.reset()

    assert "scene" in obs
    assert "proprio" in obs
    # observation_space reports per-env shape; reset/step return (N, *shape)
    assert obs["scene"].shape == (N, 8, 8, 3)
    assert obs["proprio"].shape == (N, 10)
    # observation_space itself is per-env (no leading N)
    assert vec.observation_space["scene"].shape == (8, 8, 3)
    assert vec.observation_space["proprio"].shape == (10,)
    vec.close()


def test_sync_vec_env_auto_reset() -> None:
    """Done envs are auto-reset; obs for done env is the RESET obs."""
    from dreamer_arm.envs.wrappers import SyncVectorEnv

    N = 2
    # env0: terminates on step 1; env1: never terminates in this test
    env0 = _make_done_env(done_on_step=1)
    env1 = _make_mock_env([])
    vec = SyncVectorEnv([lambda e=env0: e, lambda e=env1: e], action_repeat=1)
    vec.reset()

    actions = np.zeros((N, 4), dtype=np.float32)
    obs, _rewards, terms, truncs, info = vec.step(actions)

    # env0 should have been done and auto-reset
    assert terms[0] or truncs[0], "env0 should have done"
    assert not (terms[1] or truncs[1]), "env1 should not be done"
    # After auto-reset, env0 obs is zeros (from _reset); env1 obs is ones (from _step)
    assert obs["proprio"][0].sum() == 0.0, "env0 obs should be from reset (zeros)"
    assert obs["proprio"][1].sum() == 10.0, "env1 obs should be from step (ones)"
    # final_observation[0] should be the LAST obs before reset (also zeros from step)
    assert "final_observation" in info
    vec.close()


def test_sync_vec_env_final_info() -> None:
    """final_info is populated for done envs, None for live envs."""
    from dreamer_arm.envs.wrappers import SyncVectorEnv

    N = 2
    env0 = _make_done_env(done_on_step=1)
    env1 = _make_mock_env([])
    vec = SyncVectorEnv([lambda e=env0: e, lambda e=env1: e], action_repeat=1)
    vec.reset()

    actions = np.zeros((N, 4), dtype=np.float32)
    _, _, _, _, info = vec.step(actions)

    fin = info["final_info"]
    assert fin[0] is not None, "env0 final_info should be set"
    assert fin[1] is None, "env1 final_info should be None"
    assert "success" in fin[0]


def test_sync_vec_env_action_repeat() -> None:
    """action_repeat=3 should call inner step 3x and sum reward."""
    from dreamer_arm.envs.wrappers import SyncVectorEnv

    # env that always returns reward=1.0 and never terminates
    env = _make_mock_env([])
    vec = SyncVectorEnv([lambda e=env: e], action_repeat=3)
    vec.reset()

    actions = np.zeros((1, 4), dtype=np.float32)
    _, rewards, _, _, _ = vec.step(actions)

    assert float(rewards[0]) == pytest.approx(3.0), f"Expected summed reward 3.0, got {rewards[0]}"
    vec.close()


def test_sync_vec_env_done_skips_remaining_repeats() -> None:
    """Env done on first repeat — remaining repeats must NOT be called."""
    from dreamer_arm.envs.wrappers import SyncVectorEnv

    env = _make_done_env(done_on_step=1)
    vec = SyncVectorEnv([lambda e=env: e], action_repeat=5)
    vec.reset()

    actions = np.zeros((1, 4), dtype=np.float32)
    _, rewards, terms, _, _ = vec.step(actions)

    assert terms[0], "Should be done"
    # Only 1 step called (done on first); reward should be 1.0, not 5.0
    assert float(rewards[0]) == pytest.approx(1.0)
    vec.close()


# ---------------------------------------------------------------------------
# TimeLimit wrapper
# ---------------------------------------------------------------------------


def test_time_limit_truncates() -> None:
    """TimeLimit truncates the episode when step_count >= time_limit."""
    from dreamer_arm.envs.wrappers import TimeLimit

    env = _make_mock_env([])
    env = TimeLimit(env, time_limit=3)
    env.reset()

    for i in range(2):
        _, _, _term, trunc, _ = env.step(np.zeros(4))
        assert not trunc, f"Should not truncate at step {i + 1}"

    _, _, _, trunc, _ = env.step(np.zeros(4))
    assert trunc, "Should truncate at step 3"


# ---------------------------------------------------------------------------
# ActionRatePenalty wrapper
# ---------------------------------------------------------------------------


def test_action_rate_penalty_zero_cost() -> None:
    """Zero cost: reward is unmodified."""
    from dreamer_arm.envs.wrappers import ActionRatePenalty

    env = _make_mock_env([])
    env = ActionRatePenalty(env, action_rate_cost=0.0, action_mag_cost=0.0)
    env.reset()
    a = np.ones(4, dtype=np.float32)
    _, rew, _, _, _ = env.step(a)
    assert float(rew) == pytest.approx(1.0)


def test_action_rate_penalty_nonzero() -> None:
    """Non-zero cost subtracts a positive penalty."""
    from dreamer_arm.envs.wrappers import ActionRatePenalty

    env = _make_mock_env([])
    env = ActionRatePenalty(env, action_rate_cost=1.0, action_mag_cost=0.0)
    env.reset()
    # First step: no previous action → no jerk penalty
    a0 = np.zeros(4, dtype=np.float32)
    _, rew0, _, _, _ = env.step(a0)
    assert float(rew0) == pytest.approx(1.0), "First step should have no jerk penalty"

    # Second step: change action → jerk = sum(|Δa|)
    a1 = np.ones(4, dtype=np.float32)
    _, rew1, _, _, _ = env.step(a1)
    assert float(rew1) < 1.0, "Second step should have a jerk penalty"


# ---------------------------------------------------------------------------
# Slow (MuJoCo-heavy) tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_factory_env_num_guard() -> None:
    """make_vector_env raises ValueError when env_num % task_count != 0."""
    from dreamer_arm.envs.factory import _resolve_task_assignments

    with pytest.raises(ValueError, match="divisible"):
        _resolve_task_assignments("MT10", num_envs=7, seed=0)


@pytest.mark.slow
def test_factory_single_task_obs_keys() -> None:
    """Single-task factory produces obs dict with scene + proprio keys."""
    from dreamer_arm.envs.factory import make_vector_env

    vec = make_vector_env(
        "metaworld:door-open",
        num_envs=1,
        seed=0,
        size=(64, 64),
        action_repeat=1,
        time_limit=5,
        arm="yam",
        camera="corner",
    )
    obs = vec.reset()
    assert "scene" in obs
    assert "proprio" in obs
    assert obs["scene"].shape == (1, 64, 64, 3)
    assert obs["proprio"].shape == (1, 10)
    assert obs["scene"].dtype == np.uint8
    assert obs["proprio"].dtype == np.float32
    vec.close()


@pytest.mark.slow
def test_factory_multitask_task_id() -> None:
    """MT10 with 10 envs: each env emits a one-hot task_id of length 10."""
    from dreamer_arm.envs.factory import make_vector_env

    N = 10
    vec = make_vector_env(
        "metaworld:MT10",
        num_envs=N,
        seed=0,
        size=(64, 64),
        action_repeat=1,
        time_limit=5,
        arm="yam",
        camera="corner",
    )
    obs = vec.reset()
    assert "task_id" in obs, "Multi-task obs should have task_id key"
    assert obs["task_id"].shape == (N, N), f"task_id shape: {obs['task_id'].shape}"

    # Each env's task_id should be a valid one-hot
    for i in range(N):
        row = obs["task_id"][i]
        assert float(row.sum()) == pytest.approx(1.0), f"Env {i}: task_id not one-hot"
    vec.close()


@pytest.mark.slow
def test_sticky_success() -> None:
    """MetaWorldEnv success is sticky (OR across episode)."""
    import metaworld

    from dreamer_arm.envs.arms import make_arm
    from dreamer_arm.envs.metaworld import MetaWorldEnv

    mt1 = metaworld.MT1("door-open", seed=0)
    env_cls = next(iter(mt1.train_classes.values()))
    task = mt1.train_tasks[0]

    arm = make_arm("yam")
    inner = env_cls(render_mode=None)
    arm.attach(inner)

    env = MetaWorldEnv(inner, task, arm="yam", size=(64, 64), camera="corner")
    _obs, info = env.reset()
    assert not info["success"]

    # Manually set episode_success to simulate a step that reported success
    env._episode_success = True

    # Even if next inner step returns success=False, sticky flag must persist
    _obs2, _rew, _term, _trunc, info2 = env.step(np.zeros(4, dtype=np.float32))
    assert info2["success"], "Sticky success flag should persist across steps"
    env.close()
