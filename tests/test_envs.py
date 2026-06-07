"""Smoke tests for env factories.

These are marked ``slow`` because MuJoCo takes noticeable time to
import and compile models on first use. They run only with ``pixi run pytest
-m slow``; the default ``test`` task skips them.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Manipulation env (arm-agnostic framework, YAM arm)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_manip_yam_reach_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("manip:reach", arm="yam", seed=0, time_limit=10)
    obs, info = env.reset()
    assert set(obs.keys()) >= {"image", "state", "target", "is_first", "is_last", "is_terminal"}
    assert "object" not in obs
    assert obs["image"].dtype == np.uint8
    assert obs["image"].shape == (64, 64, 3)
    assert obs["is_first"]
    assert info["success"] is False
    # 4-D arm-agnostic EE action
    assert env.action_space.shape == (4,)

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(act)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.slow
def test_manip_yam_pick_place_obs_action_spaces() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("manip:pick_place", arm="yam", seed=0, time_limit=10)
    obs, info = env.reset()
    assert set(obs.keys()) >= {
        "image",
        "state",
        "object",
        "goal",
        "is_first",
        "is_last",
        "is_terminal",
    }
    assert "target" not in obs
    assert obs["object"].shape == (3,)
    assert obs["goal"].shape == (3,)
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["is_first"]
    assert info["success"] is False
    assert env.action_space.shape == (4,)

    for _ in range(5):
        act = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(act)
        assert np.isfinite(reward) and reward >= 0.0
        if terminated or truncated:
            break
    env.close()


@pytest.mark.slow
def test_manip_yam_pick_place_object_rests_on_floor() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("manip:pick_place", arm="yam", seed=1, time_limit=60)
    env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(40):
        obs, *_ = env.step(zero)
    z = float(obs["object"][2])
    # Object should rest near floor_top (~0.01 m), not tunnel or fly away.
    assert -0.02 < z < 0.06, f"unexpected object z={z}"
    env.close()


@pytest.mark.slow
def test_manip_yam_pick_place_spawn_separation() -> None:
    from dreamer_arm.envs.factory import make_env

    env = make_env("manip:pick_place", arm="yam", seed=2, time_limit=10)
    obs, _ = env.reset()
    sep = float(np.linalg.norm(obs["object"][:2] - obs["goal"][:2]))
    assert sep > 0.08, f"object and goal too close at reset: {sep:.3f} m"
    env.close()


@pytest.mark.slow
def test_manip_yam_pick_place_goal_includes_floor_level() -> None:
    """Goal z should sometimes be near floor level (bug-fix: was always mid-air)."""
    from dreamer_arm.envs.factory import make_env

    env = make_env("manip:pick_place", arm="yam", seed=99, time_limit=10)
    goal_zs = []
    for _ in range(20):
        obs, _ = env.reset()
        goal_zs.append(float(obs["goal"][2]))
    env.close()
    # With 20 resets from [0.01, 0.40], at least some should be below 0.15.
    assert min(goal_zs) < 0.15, (
        f"All goal z values are mid-air (min={min(goal_zs):.3f}). "
        "Floor-level goal bug may not be fixed."
    )


@pytest.mark.slow
def test_manip_yam_pick_place_success_no_grasp_required() -> None:
    """Success should fire when object is at goal even when not grasped."""
    from dreamer_arm.envs.manip import Manipulation

    env = Manipulation(arm="yam", task="pick_place", seed=0)
    env.reset()

    # Teleport the object directly onto the goal.
    task = env._task_obj  # type: ignore[attr-defined]
    goal = np.array(env._data.mocap_pos[task._goal_mocap_id], dtype=np.float32)
    adr = task._obj_qpos_adr
    env._data.qpos[adr : adr + 3] = goal
    env._data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env._model, env._data)

    _, success = task.reward(env._model, env._data, env._controller, env._success_threshold)
    assert success, "Success should fire when object is at goal (no grasp required)"
    env.close()


@pytest.mark.slow
def test_manip_yam_ik_moves_tcp() -> None:
    """Commanding +x EE action should move the TCP in the +x direction."""
    from dreamer_arm.envs.manip import Manipulation

    env = Manipulation(arm="yam", task="reach", seed=0, action_repeat=4)
    env.reset()
    tcp_before = env._controller.tcp_pos(env._data).copy()  # type: ignore[attr-defined]

    # Command +x for 10 controlled steps.
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(10):
        env.step(action)

    tcp_after = env._controller.tcp_pos(env._data).copy()  # type: ignore[attr-defined]
    dx = float(tcp_after[0] - tcp_before[0])
    assert dx > 0.005, f"TCP x should increase with +x action; got Δx={dx:.4f} m"
    env.close()
