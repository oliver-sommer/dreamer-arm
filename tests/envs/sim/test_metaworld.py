import numpy as np
import pytest

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
    finally:
        env.close()
