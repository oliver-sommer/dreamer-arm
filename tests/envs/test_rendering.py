from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from dreamer_arm.envs.rendering import SceneRenderer, apply_barrel_distortion


def _make_yam_inner() -> tuple[Any, Any, int]:
    import metaworld
    import mujoco

    from dreamer_arm.envs.arms import make_arm

    metaworld.set_active_arm("yam")
    benchmark = metaworld.MT1("reach-v3", seed=0)
    env_cls = next(iter(benchmark.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm("yam")
    arm.attach(inner)
    inner.set_task(benchmark.train_tasks[0])
    site_id = int(mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    return inner, arm, site_id


def test_cameras_are_upright() -> None:
    import mujoco

    inner, _, _ = _make_yam_inner()
    try:
        data = mujoco.MjData(inner.model)
        mujoco.mj_forward(inner.model, data)
        inverted = []
        for camera_id in range(inner.model.ncam):
            name = mujoco.mj_id2name(inner.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            up_z = float(np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)[2, 1])
            if up_z < -1e-6:
                inverted.append((name, up_z))
        assert not inverted
    finally:
        inner.close()


def test_render_path_applies_no_flip() -> None:
    source = inspect.getsource(SceneRenderer.grab_frame)
    assert "flipud" not in source
    assert "fliplr" not in source


def test_zero_barrel_distortion_preserves_image() -> None:
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    np.testing.assert_array_equal(apply_barrel_distortion(image, k1=0.0), image)
