from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from dreamer_arm.envs.sim import factory
from dreamer_arm.envs.sim.arms import ArmConfig
from dreamer_arm.envs.sim.factory import _resolve_task_assignments, make_vector_env
from dreamer_arm.utils.config import get_config_root


def test_build_from_config_forwards_arm_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_make_vector_env(*_args: Any, **kwargs: Any) -> Any:
        captured["arm_cfg"] = kwargs.get("arm_cfg")
        return MagicMock()

    monkeypatch.setattr(factory, "make_vector_env", _fake_make_vector_env)
    with initialize_config_dir(config_dir=str(get_config_root()), version_base=None):
        cfg = compose(
            config_name="training/dreamer",
            overrides=["envs/sim=metaworld", "envs.sim.task=door-open", "envs.sim.arms.damping=0.3"],
        )
    factory.build_from_config(cfg)

    arm_cfg = captured["arm_cfg"]
    assert isinstance(arm_cfg, ArmConfig)
    assert arm_cfg.damping == pytest.approx(0.3)
    assert arm_cfg.name == "yam"


def test_factory_env_num_guard() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _resolve_task_assignments("MT10", num_envs=7, seed=0)


@pytest.mark.slow
def test_factory_single_task_obs_keys() -> None:
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
    try:
        obs = vec.reset()
        assert obs["scene"].shape == (1, 64, 64, 3)
        assert obs["proprio"].shape == (1, 10)
        assert obs["scene"].dtype == np.uint8
        assert obs["proprio"].dtype == np.float32
    finally:
        vec.close()


@pytest.mark.slow
def test_factory_multitask_task_id() -> None:
    count = 10
    vec = make_vector_env(
        "metaworld:MT10",
        num_envs=count,
        seed=0,
        size=(64, 64),
        action_repeat=1,
        time_limit=5,
        arm="yam",
        camera="corner",
    )
    try:
        task_ids = vec.reset()["task_id"]
        assert task_ids.shape == (count, count)
        np.testing.assert_allclose(task_ids.sum(axis=1), np.ones(count))
    finally:
        vec.close()


@pytest.mark.slow
def test_action_repeat_renders_scene_only_once_per_agent_step() -> None:
    vec = make_vector_env(
        "metaworld:reach",
        num_envs=1,
        seed=0,
        size=(64, 64),
        action_repeat=3,
        time_limit=50,
        arm="yam",
        camera="corner",
    )
    try:
        vec.reset()
        inner = vec._envs[0].env
        original_render_scene = inner._render_scene
        render_calls = 0

        def _counting_render_scene() -> np.ndarray:
            nonlocal render_calls
            render_calls += 1
            return original_render_scene()

        inner._render_scene = _counting_render_scene
        vec.step(np.zeros((1, 4), dtype=np.float32))
        assert render_calls == 1
    finally:
        vec.close()
