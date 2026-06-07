"""Meta-World MT1 → Gymnasium 1.x adapter for Dreamer.

Wraps any Meta-World v3 MT1 task as a single-instance Gymnasium env with a
Dict observation space containing an ``image`` (uint8 RGB) and a ``state``
(float32 proprioceptive vector).  The ``DreamerObsWrapper`` in ``wrappers.py``
adds the ``is_first``/``is_last``/``is_terminal`` flags on top, so this class
must not emit them.

Task names use the raw Meta-World convention without the ``-v3`` suffix
(e.g. ``"reach"``, ``"pick-place"``, ``"door-open"``).  The factory prepends
the ``metaworld:`` suite prefix, which is stripped before passing to this class.

Rendering
---------
We bypass Meta-World's built-in ``mujoco_renderer`` and use ``mujoco.Renderer``
directly (the same pattern as the manip env).  This avoids OpenGL context
conflicts when the passive viewer is also open.

Success tracking
----------------
Meta-World does not terminate episodes on success; it signals success via
``info["success"]``.  We maintain a sticky ``_success`` flag that is set True
the first time ``info["success"]`` is truthy within an episode and remains True
until the next ``reset()``.  The factory's ``SyncVectorEnv`` auto-reset logic
reads ``final_info["success"]`` for logging (``trainer.py:176``), so the
truncation step's info must reflect the entire episode's success, not just the
last step's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import mujoco
    import mujoco.viewer as _viewer_mod

import gymnasium as gym
import numpy as np

ObsDict = dict[str, np.ndarray]


class MetaWorld(gym.Env):  # type: ignore[type-arg]
    """Single Meta-World MT1 task as a Gymnasium env with a Dict obs space.

    The obs dict carries ``image`` (uint8 RGB at ``size``) and ``state``
    (the raw proprioceptive vector from Meta-World's observation space).

    Parameters
    ----------
    name:
        MT1 task name without the ``-v3`` suffix, e.g. ``"reach"``.
    action_repeat:
        Number of inner ``step()`` calls per outer ``step()``; reward summed.
    size:
        ``(height, width)`` of the rendered image in pixels.
    camera:
        MuJoCo camera name (e.g. ``"corner2"``).
    seed:
        Used to seed the MT1 benchmark for reproducible task sampling.
    viewer:
        Open a passive MuJoCo viewer window (macOS + ``mjpython`` only).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(
        self,
        name: str,
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        camera: str = "corner2",
        seed: int = 0,
        viewer: bool = False,
    ) -> None:
        import metaworld
        import mujoco as _mj

        self._name = name
        self._action_repeat = int(action_repeat)
        self._size = size
        self._camera = camera

        mt1 = metaworld.MT1(name + "-v3", seed=seed)
        env = mt1.train_classes[name + "-v3"](
            render_mode="rgb_array",
            camera_name=camera,
        )
        env.set_task(mt1.train_tasks[0])

        # Adjust camera position for the corner2 view used in the paper.
        if camera == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        # Allow task randomisation across episodes.
        env._freeze_rand_vec = False

        self._env = env
        self._mt1 = mt1
        self._success: bool = False
        self._viewer_requested = viewer
        self._passive_viewer: _viewer_mod.Handle | None = None

        # Own renderer — bypasses Meta-World's mujoco_renderer to avoid
        # OpenGL context conflicts with the passive viewer.
        self._renderer: mujoco.Renderer = _mj.Renderer(env.model, height=size[0], width=size[1])
        cam_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, camera)
        self._camera_id: int = int(cam_id) if cam_id >= 0 else 0

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(  # type: ignore[assignment]
            {
                "image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
                "state": env.observation_space,
            }
        )
        self.action_space: gym.spaces.Box = gym.spaces.Box(  # type: ignore[assignment]
            env.action_space.low,
            env.action_space.high,
            dtype=np.float32,
        )

    # ---------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsDict, dict[str, Any]]:
        del options
        if seed is not None:
            import metaworld

            mt1 = metaworld.MT1(self._name + "-v3", seed=seed)
            self._env.set_task(mt1.train_tasks[0])

        self._success = False
        state, _ = self._env.reset()
        if self._viewer_requested:
            import mujoco.viewer as _mv

            self._passive_viewer = _mv.launch_passive(self._env.model, self._env.data)
            self._viewer_requested = False
        self._sync_viewer()
        return {"image": self.render(), "state": np.asarray(state, dtype=np.float32)}, {}

    def step(self, action: np.ndarray) -> tuple[ObsDict, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        state = None
        for _ in range(self._action_repeat):
            state, reward, terminated, truncated, info = self._env.step(action)
            total_reward += float(reward)
            if info.get("success", False):
                self._success = True
            if terminated or truncated:
                break

        assert state is not None
        self._sync_viewer()
        obs: ObsDict = {
            "image": self.render(),
            "state": np.asarray(state, dtype=np.float32),
        }
        return obs, total_reward, terminated, truncated, {"success": self._success}

    def render(self) -> np.ndarray:  # type: ignore[override]
        self._renderer.update_scene(self._env.data, camera=self._camera_id)
        frame: np.ndarray = self._renderer.render()
        if self._camera == "corner2":
            return np.flip(frame, axis=0)
        return frame

    def _sync_viewer(self) -> None:
        if self._passive_viewer is not None and self._passive_viewer.is_running():
            self._passive_viewer.sync()

    def close(self) -> None:
        self._renderer.close()
        if self._passive_viewer is not None:
            self._passive_viewer.close()
        self._env.close()
