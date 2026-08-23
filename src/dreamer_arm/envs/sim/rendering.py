"""MuJoCo camera rendering and visual randomisation for simulated environments."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

_CAM_LOOKAT = np.array([0.0, 0.55, 0.15])
_CAM_AZIMUTH_RANGE = (100.0, 200.0)
_CAM_ELEVATION_RANGE = (-50.0, -15.0)
_CAM_DISTANCE_RANGE = (0.85, 1.3)


def apply_barrel_distortion(image: np.ndarray, k1: float) -> np.ndarray:
    """Apply radial barrel or pincushion distortion to an RGB frame."""
    if abs(k1) < 1e-6:
        return image
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        return image

    height, width, channels = image.shape
    cx, cy = width / 2.0, height / 2.0
    y_idx, x_idx = np.mgrid[0:height, 0:width]
    xn = (x_idx - cx) / cx
    yn = (y_idx - cy) / cy
    factor = 1.0 + k1 * (xn**2 + yn**2)
    xs = xn * factor * cx + cx
    ys = yn * factor * cy + cy
    out = np.empty_like(image)
    for channel in range(channels):
        out[:, :, channel] = map_coordinates(
            image[:, :, channel].astype(np.float32),
            [ys, xs],
            order=1,
            mode="nearest",
        )
    return out.astype(np.uint8)


class SceneRenderer:
    """Own one MuJoCo renderer for scene and wrist cameras."""

    def __init__(
        self,
        env: Any,
        size: tuple[int, int],
        camera: str,
        wrist_camera: str | None,
        *,
        scene_randomize: bool,
        camera_pose_randomize: bool,
        camera_jitter: float,
        wrist_fisheye: float,
    ) -> None:
        self._env = env
        self._camera = camera
        self._wrist_camera = wrist_camera
        self._scene_randomize = scene_randomize
        self._camera_pose_randomize = camera_pose_randomize
        self._camera_jitter = camera_jitter
        self._wrist_fisheye = wrist_fisheye
        self._renderer = mujoco.Renderer(env.model, height=size[0], width=size[1])
        self._scene_camera: mujoco.MjvCamera | None = None
        self._light_diffuse = env.model.light_diffuse.copy()
        self._light_ambient = env.model.light_ambient.copy()

    def reset(self, rng: np.random.Generator) -> None:
        if self._camera_pose_randomize or self._camera_jitter > 0.0:
            self._scene_camera = self._make_free_camera(rng)
        else:
            self._scene_camera = None
        if self._scene_randomize:
            self._randomise_lighting(rng)

    def render_scene(self) -> np.ndarray:
        camera = self._scene_camera if self._scene_camera is not None else self._camera
        self._renderer.update_scene(self._env.data, camera=camera)
        return self.grab_frame()

    def render_wrist(self) -> np.ndarray:
        self._renderer.update_scene(self._env.data, camera=self._wrist_camera)
        return apply_barrel_distortion(self.grab_frame(), self._wrist_fisheye)

    def grab_frame(self) -> np.ndarray:
        """Return the renderer's pixels without orientation post-processing."""
        return self._renderer.render().copy()

    def close(self) -> None:
        self._renderer.close()

    def _make_free_camera(self, rng: np.random.Generator) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if self._camera_pose_randomize:
            camera.azimuth = float(rng.uniform(*_CAM_AZIMUTH_RANGE))
            camera.elevation = float(rng.uniform(*_CAM_ELEVATION_RANGE))
            camera.distance = float(rng.uniform(*_CAM_DISTANCE_RANGE))
        else:
            camera.azimuth = 150.0
            camera.elevation = -35.0
            camera.distance = 1.0
        lookat = _CAM_LOOKAT.copy()
        if self._camera_jitter > 0.0:
            lookat += rng.normal(0.0, self._camera_jitter, 3)
        camera.lookat[:] = lookat
        return camera

    def _randomise_lighting(self, rng: np.random.Generator) -> None:
        # Always scale the captured baseline; the model persists across resets.
        model = self._env.model
        for light in range(model.nlight):
            tint = rng.uniform(0.7, 1.0, 3).astype(np.float32)
            model.light_diffuse[light] = np.clip(self._light_diffuse[light] * tint, 0.0, 1.0)
            ambient = float(rng.uniform(0.5, 1.2))
            model.light_ambient[light] = np.clip(self._light_ambient[light] * ambient, 0.0, 1.0)
