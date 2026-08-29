import numpy as np
import pytest

from dreamer_arm.envs.observation import ObservationSpec
from dreamer_arm.envs.sim import metaworld as metaworld_env
from dreamer_arm.envs.sim.arms import make_arm


class _Renderer:
    def __init__(self, _env: object, size: tuple[int, int], *_args: object, **_kwargs: object) -> None:
        self._size = size

    def reset(self, _rng: np.random.Generator) -> None:
        pass

    def render_scene(self) -> np.ndarray:
        return np.zeros((*self._size, 3), dtype=np.uint8)

    def render_wrist(self) -> np.ndarray:
        return np.zeros((*self._size, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


def test_sticky_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import metaworld

    monkeypatch.setattr(metaworld_env, "SceneRenderer", _Renderer)
    metaworld.set_active_arm("yam")
    benchmark = metaworld.MT1("door-open-v3", seed=0)
    env_cls = next(iter(benchmark.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm("yam")
    arm.attach(inner)
    env = metaworld_env.MetaWorldEnv(inner, benchmark.train_tasks[0], arm="yam", size=(64, 64), camera="corner")
    try:
        _, info = env.reset()
        assert not info["success"]
        env._episode_success = True
        *_, next_info = env.step(np.zeros(4, dtype=np.float32))
        assert next_info["success"]
        assert next_info["reward_diag"]
        assert "mw_v2_reward" in next_info["reward_diag"]
        assert all(np.isfinite(value) for value in next_info["reward_diag"].values())
    finally:
        env.close()


@pytest.mark.parametrize("arm_name", ["yam", "sawyer"])
def test_proprio_reports_the_backend_controlled_tool_point(monkeypatch: pytest.MonkeyPatch, arm_name: str) -> None:
    import metaworld

    monkeypatch.setattr(metaworld_env, "SceneRenderer", _Renderer)
    metaworld.set_active_arm(arm_name)
    benchmark = metaworld.MT1("reach-v3", seed=0)
    env_cls = next(iter(benchmark.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm(arm_name)
    arm.attach(inner)
    env = metaworld_env.MetaWorldEnv(
        inner,
        benchmark.train_tasks[0],
        arm=arm_name,
        arm_plugin=arm,
        size=(64, 64),
        camera="corner",
    )
    try:
        observation, _ = env.reset()
        expected = (
            np.asarray(inner.data.site_xpos[env._grasp_site_id]) if arm_name == "yam" else np.asarray(inner.tcp_center)
        )
        np.testing.assert_allclose(observation["proprio"][ObservationSpec.TOOL_POSITION], expected, atol=1e-7)
        assert observation["proprio"][ObservationSpec.GRIPPER_OPEN] == pytest.approx(inner.get_gripper_open())
        assert env.observation_space.contains(observation)
    finally:
        env.close()
