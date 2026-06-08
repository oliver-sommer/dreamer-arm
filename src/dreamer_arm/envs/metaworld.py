"""Meta-World MT1 → Gymnasium 1.x adapter for Dreamer.

Wraps any Meta-World v3 MT1 task as a single-instance Gymnasium env with a
Dict observation space containing an ``image`` (uint8 RGB) and a ``state``
(float32 proprioceptive vector).  The ``DreamerObsWrapper`` in ``wrappers.py``
adds the ``is_first``/``is_last``/``is_terminal`` flags on top, so this class
must not emit them.

Task names use the raw Meta-World convention without the ``-v3`` suffix
(e.g. ``"door-open"``, ``"drawer-close"``).  The factory prepends
the ``metaworld:`` suite prefix, which is stripped before passing to this class.

Arm selection
-------------
Pass ``arm="yam"`` to drive YAM's real actuated arm through the
``EEController`` instead of Sawyer's mocap weld.  When ``arm="yam"``:

1. ``metaworld.set_active_arm("yam")`` is called before building the env so
   ``full_V3_path_for`` routes to the YAM task XML variants.
2. After building the env, an ``EEController`` is wired in via the fork's
   injectable ``_external_actuation`` and ``_external_reset_hand`` hooks.
3. Gripper sign is negated: Meta-World action ``+1 = close`` whereas
   ``EEController`` interprets ``+1 = open``.
4. Action-scale: Meta-World's ``action_scale=1/100`` over ``frame_skip=5``
   corresponds to a maximum displacement of ~1cm per controlled step.
   The YAM controller uses ``ee_step_m=0.01`` to match this scale.

Rendering
---------
We bypass Meta-World's built-in ``mujoco_renderer`` and use ``mujoco.Renderer``
directly (the same pattern as the manip env).  The Meta-World env is therefore
constructed with ``render_mode=None`` so gymnasium does not create a second,
unused renderer object.

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

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import mujoco
    import mujoco.viewer as _viewer_mod

import gymnasium as gym
import numpy as np

ObsDict = dict[str, np.ndarray]

# ee_step_m for YAM in Meta-World context: action_scale=1/100, frame_skip=5
# → max displacement ≈ 5 * (1/100) m = 0.05 m/step.  We use a per-substep
# step of 0.01 m so the IK is called once per outer step (no inner loop);
# the _apply_action hook handles the frame_skip loop.
_YAM_MW_EE_STEP_M = 0.01

# ---------------------------------------------------------------------------
# Scene domain-randomization constants
# ---------------------------------------------------------------------------

# Texture-pool name prefixes used in basic_scene.xml (resolved by name at init).
_DR_CUBE_POOL_PREFIX = "T_pool_"  # cube textures for table / retaining walls
_DR_FLOOR_POOL_PREFIX = "T_floorpool_"  # 2d textures for the floor plane
# RGBA channel jitter magnitude (applied independently per channel, clamped [0,1]).
_SCENE_RGBA_JITTER = 0.3
# Light-color jitter magnitude (headlight ambient + directional diffuse/specular).
_SCENE_LIGHT_JITTER = 0.15

# ---------------------------------------------------------------------------
# Camera-pose domain-randomization constants
# ---------------------------------------------------------------------------
# World layout (from the MetaWorld XMLs):
#   arm base  ≈ (0, 0.23, 0.01)   — mounts at the near table edge
#   table top ≈ x∈[-0.7,0.7], y∈[0.2,1.0], z≈0
#   "behind arm" = open near side (y < 0.23), looking in +y toward the table.
# Ranges are research-informed (robosuite / MV-MWM camera DR literature).
_CAM_TARGET = (0.0, 0.6, 0.08)  # look-at point: table centre, slightly above top
_CAM_TARGET_JITTER = 0.02  # ± m of noise on the look-at point per episode
_CAM_AZIMUTH_DEG = 60.0  # half-width azimuth sweep (°) around directly-behind
_CAM_ELEVATION_MIN_DEG = 20.0  # minimum elevation above table plane
_CAM_ELEVATION_MAX_DEG = 60.0  # maximum elevation above table plane
_CAM_DISTANCE_MIN_M = 0.6  # minimum camera-target distance (m)
_CAM_DISTANCE_MAX_M = 1.1  # maximum camera-target distance (m)


def _build_fisheye_map(h: int, w: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Precompute barrel-distortion source coordinates for an h x w image.

    Returns (y_src, x_src) float32 arrays of shape (h, w).  Each entry gives
    the source pixel to sample for the corresponding output pixel.  Coordinates
    outside [0, h/w) are clamped (boundary-fill) at sample time.

    ``strength`` controls barrel intensity: 0 = identity, 0.5 = visible fisheye
    (corners sample from ~50% beyond the image radius).
    """
    y_out, x_out = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    xn = (x_out - cx) / cx
    yn = (y_out - cy) / cy
    r = np.hypot(xn, yn)
    r_src = r * (1.0 + strength * r * r)
    safe_r = np.where(r > 0, r, 1.0)
    x_src = np.where(r > 0, xn / safe_r * r_src * cx + cx, cx).astype(np.float32)
    y_src = np.where(r > 0, yn / safe_r * r_src * cy + cy, cy).astype(np.float32)
    return y_src, x_src


def _apply_fisheye(frame: np.ndarray, fisheye_map: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Remap an (H, W, 3) uint8 frame through precomputed fisheye coordinates."""
    y_src, x_src = fisheye_map
    h, w = frame.shape[:2]
    x0 = np.floor(x_src).astype(np.int32)
    y0 = np.floor(y_src).astype(np.int32)
    dx = (x_src - x0)[..., None].astype(np.float32)
    dy = (y_src - y0)[..., None].astype(np.float32)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    result = (
        frame[y0c, x0c].astype(np.float32) * (1 - dy) * (1 - dx)
        + frame[y0c, x1c].astype(np.float32) * (1 - dy) * dx
        + frame[y1c, x0c].astype(np.float32) * dy * (1 - dx)
        + frame[y1c, x1c].astype(np.float32) * dy * dx
    )
    return result.astype(np.uint8)


class MetaWorld(gym.Env):  # type: ignore[type-arg]
    """Single Meta-World MT1 task as a Gymnasium env with a Dict obs space.

    The obs dict carries ``scene`` (uint8 RGB at ``size``) and ``state``
    (the raw proprioceptive vector from Meta-World's observation space).

    Parameters
    ----------
    name:
        MT1 task name without the ``-v3`` suffix, e.g. ``"door-open"``.
    arm:
        Arm identifier.  ``"sawyer"`` (default) uses the upstream Sawyer mocap
        control; ``"yam"`` drives the YAM arm via ``EEController`` IK.
    action_repeat:
        Number of inner ``step()`` calls per outer ``step()``; reward summed.
    size:
        ``(height, width)`` of the rendered image in pixels.
    camera:
        MuJoCo camera name (e.g. ``"corner2"``).
    wrist_camera:
        Optional second MuJoCo camera name for a wrist-mounted view.
    camera_jitter:
        Per-episode Gaussian noise magnitude (m) added to the scene-camera
        position.  ``0`` disables.  When ``camera_pose_randomize`` is also
        enabled this noise is layered on top of the sampled pose.
    scene_randomize:
        When ``True``, each ``reset()`` randomly swaps the table, retaining-wall,
        and floor textures from a pre-loaded pool and jitters RGBA tints and
        lighting colours.  Requires the DR texture pool to be present in the
        scene XML (``basic_scene.xml`` includes it by default).
    camera_pose_randomize:
        When ``True``, each ``reset()`` samples a new scene-camera pose on a
        wide hemisphere **behind the arm** (azimuth +-60 deg, elevation 20-60 deg,
        distance 0.6-1.1 m from the table centre), aimed at the table with a
        computed look-at quaternion.  This is independent of ``camera_jitter``.
    seed:
        Used to seed the MT1 benchmark for reproducible task sampling.
    viewer:
        Open a passive MuJoCo viewer window (macOS + ``mjpython`` only).
    task_idx, num_tasks:
        Multi-task conditioning.  When ``num_tasks`` is set, the obs dict gains a
        ``task_id`` key holding a one-hot of length ``num_tasks`` with a 1 at
        ``task_idx``.  This is how a generalist policy is told which task it is
        in; the factory pins one task per env (see ``_make_metaworld_mt``).  Left
        ``None`` for ordinary single-task runs, in which case no ``task_id`` key
        is emitted.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(
        self,
        name: str,
        arm: str = "sawyer",
        action_repeat: int = 1,
        size: tuple[int, int] = (64, 64),
        camera: str = "corner2",
        wrist_camera: str | None = None,
        camera_jitter: float = 0.0,
        scene_randomize: bool = False,
        camera_pose_randomize: bool = False,
        wrist_fisheye: float = 0.0,
        seed: int = 0,
        viewer: bool = False,
        task_idx: int | None = None,
        num_tasks: int | None = None,
    ) -> None:
        import metaworld
        import mujoco as _mj

        self._name = name
        self._arm = arm
        self._action_repeat = int(action_repeat)
        self._size = size
        self._camera = camera
        self._wrist_camera = wrist_camera
        self._camera_jitter = float(camera_jitter)
        self._scene_randomize = scene_randomize
        self._camera_pose_randomize = camera_pose_randomize
        self._rng = np.random.default_rng(seed)

        # One-hot task conditioning (multi-task runs only).  Constant per env
        # because the factory pins a single task class to each env instance.
        self._task_onehot: np.ndarray | None = None
        if num_tasks is not None:
            if task_idx is None or not 0 <= task_idx < num_tasks:
                raise ValueError(
                    f"task_idx must be in [0, {num_tasks}) when num_tasks is set; got {task_idx}"
                )
            self._task_onehot = np.eye(num_tasks, dtype=np.float32)[task_idx]

        # Route asset paths for the YAM arm before building any env class.
        if arm == "yam":
            metaworld.set_active_arm("yam")
        else:
            metaworld.set_active_arm("sawyer")

        mt1 = metaworld.MT1(name + "-v3", seed=seed)
        # render_mode=None: we manage all rendering via our own mujoco.Renderer
        # below; passing "rgb_array" would make gymnasium create a second,
        # unused MujocoRenderer object.
        env = mt1.train_classes[name + "-v3"](render_mode=None)
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

        # Wire in the YAM IK controller via the fork's injectable hooks.
        if arm == "yam":
            self._setup_yam_control(env)

        # Own renderer — gives us full control over camera and image format;
        # Meta-World's internal mujoco_renderer is never initialised (render_mode=None).
        self._renderer: mujoco.Renderer = _mj.Renderer(env.model, height=size[0], width=size[1])
        cam_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, camera)
        self._camera_id: int = int(cam_id) if cam_id >= 0 else 0
        # Snapshot the (possibly overridden) base position for per-episode jitter.
        self._cam_base_pos: np.ndarray = np.array(env.model.cam_pos[self._camera_id], copy=True)

        # corner2's original euler produces an upside-down image; render() corrects
        # it with a vertical flip.  When camera_pose_randomize computes an upright
        # look-at quaternion directly, no flip is needed.
        self._scene_flip: bool = (camera == "corner2") and not camera_pose_randomize

        # Per-episode appearance DR state (populated lazily by _dr_init).
        self._dr_cube_mat_ids: list[int] = []
        self._dr_floor_mat_ids: list[int] = []
        self._dr_mat_base_rgba: dict[int, np.ndarray] = {}
        self._dr_cube_pool: list[int] = []
        self._dr_floor_pool: list[int] = []
        self._dr_headlight_ambient_base: np.ndarray | None = None
        self._dr_light_diffuse_base: np.ndarray | None = None
        self._dr_light_specular_base: np.ndarray | None = None
        if scene_randomize:
            self._dr_init(env)

        self._wrist_camera_id: int | None = None
        if wrist_camera is not None:
            wc_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_CAMERA, wrist_camera)
            if wc_id < 0:
                raise RuntimeError(
                    f"Wrist camera {wrist_camera!r} not found in model. "
                    "Check that yam_xyz_base.xml defines it."
                )
            self._wrist_camera_id = int(wc_id)

        # Precompute barrel-distortion remap for the wrist camera once.
        self._fisheye_map: tuple[np.ndarray, np.ndarray] | None = None
        if wrist_fisheye > 0.0 and self._wrist_camera_id is not None:
            self._fisheye_map = _build_fisheye_map(size[0], size[1], wrist_fisheye)

        obs_spaces: dict[str, gym.spaces.Space] = {  # type: ignore[type-arg]
            "scene": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
            "state": env.observation_space,
        }
        if self._wrist_camera_id is not None:
            obs_spaces["wrist_image"] = gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8)
        if self._task_onehot is not None:
            obs_spaces["task_id"] = gym.spaces.Box(
                0.0, 1.0, (self._task_onehot.shape[0],), dtype=np.float32
            )
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(obs_spaces)
        self.action_space: gym.spaces.Box = gym.spaces.Box(
            env.action_space.low,
            env.action_space.high,
            dtype=np.float32,
        )

    # ---------------------------------------------------------------- DR helpers

    def _dr_init(self, env: Any) -> None:
        """Cache material IDs, texture pool IDs, and baseline values for DR.

        Called once in ``__init__`` when ``scene_randomize=True``.  Safe to call
        on scenes where the named pool textures are absent — the pools stay empty
        and ``_apply_scene_dr`` skips texture-swapping for those surfaces.
        """
        import mujoco as _mj

        model = env.model

        # Material IDs and baseline RGBA for the three randomised surfaces.
        for mat_name in ("table_wood", "wall_metal"):
            mid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_MATERIAL, mat_name)
            if mid >= 0:
                self._dr_cube_mat_ids.append(mid)
                self._dr_mat_base_rgba[mid] = np.array(model.mat_rgba[mid], copy=True)
        for mat_name in ("basic_floor",):
            mid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_MATERIAL, mat_name)
            if mid >= 0:
                self._dr_floor_mat_ids.append(mid)
                self._dr_mat_base_rgba[mid] = np.array(model.mat_rgba[mid], copy=True)

        # Collect named texture pools (T_pool_0, T_pool_1, … and T_floorpool_0, …).
        for prefix, pool in (
            (_DR_CUBE_POOL_PREFIX, self._dr_cube_pool),
            (_DR_FLOOR_POOL_PREFIX, self._dr_floor_pool),
        ):
            i = 0
            while True:
                tid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_TEXTURE, f"{prefix}{i}")
                if tid < 0:
                    break
                pool.append(tid)
                i += 1

        # Also include each material's original texture so the default appearance
        # is part of the random distribution (one-in-N chance per episode).
        is_multi_role = model.mat_texid.ndim == 2  # MuJoCo ≥3 multi-role layout
        for mat_ids, pool in (
            (self._dr_cube_mat_ids, self._dr_cube_pool),
            (self._dr_floor_mat_ids, self._dr_floor_pool),
        ):
            if not pool:
                continue
            for mid in mat_ids:
                orig = int(model.mat_texid[mid, 0] if is_multi_role else model.mat_texid[mid])
                if orig >= 0 and orig not in pool:
                    pool.insert(0, orig)

        # Snapshot baseline lighting for per-episode jitter.
        self._dr_headlight_ambient_base = np.array(model.vis.headlight.ambient, copy=True)
        self._dr_light_diffuse_base = np.array(model.light_diffuse, copy=True)
        self._dr_light_specular_base = np.array(model.light_specular, copy=True)

    def _apply_scene_dr(self) -> None:
        """Per-episode: swap textures, jitter RGBA tints, and jitter lighting."""
        model = self._env.model
        is_multi_role = model.mat_texid.ndim == 2  # MuJoCo ≥3 multi-role layout

        def _write_texid(mat_id: int, tex_id: int) -> None:
            if is_multi_role:
                model.mat_texid[mat_id, 0] = tex_id  # role 0 = legacy/user texture
            else:
                model.mat_texid[mat_id] = tex_id

        # Table + retaining walls: random cube texture + RGBA tint.
        if self._dr_cube_pool:
            for mid in self._dr_cube_mat_ids:
                _write_texid(mid, int(self._rng.choice(self._dr_cube_pool)))
                rgba = self._dr_mat_base_rgba[mid].copy()
                rgba[:3] = np.clip(
                    rgba[:3] + self._rng.uniform(-_SCENE_RGBA_JITTER, _SCENE_RGBA_JITTER, 3),
                    0.0,
                    1.0,
                )
                model.mat_rgba[mid] = rgba

        # Floor: random 2d texture + RGBA tint.
        if self._dr_floor_pool:
            for mid in self._dr_floor_mat_ids:
                _write_texid(mid, int(self._rng.choice(self._dr_floor_pool)))
                rgba = self._dr_mat_base_rgba[mid].copy()
                rgba[:3] = np.clip(
                    rgba[:3] + self._rng.uniform(-_SCENE_RGBA_JITTER, _SCENE_RGBA_JITTER, 3),
                    0.0,
                    1.0,
                )
                model.mat_rgba[mid] = rgba

        # Background: jitter headlight ambient and directional light colors.
        if self._dr_headlight_ambient_base is not None:
            model.vis.headlight.ambient[:] = np.clip(
                self._dr_headlight_ambient_base
                + self._rng.uniform(-_SCENE_LIGHT_JITTER, _SCENE_LIGHT_JITTER, 3),
                0.0,
                1.0,
            )
        if self._dr_light_diffuse_base is not None:
            model.light_diffuse[:] = np.clip(
                self._dr_light_diffuse_base
                + self._rng.uniform(
                    -_SCENE_LIGHT_JITTER,
                    _SCENE_LIGHT_JITTER,
                    self._dr_light_diffuse_base.shape,
                ),
                0.0,
                1.0,
            )
            model.light_specular[:] = np.clip(
                self._dr_light_specular_base  # type: ignore[operator]
                + self._rng.uniform(
                    -_SCENE_LIGHT_JITTER,
                    _SCENE_LIGHT_JITTER,
                    self._dr_light_specular_base.shape,  # type: ignore[union-attr]
                ),
                0.0,
                1.0,
            )

    def _sample_camera_pose(self) -> None:
        """Per-episode: place the scene camera on the hemisphere behind the arm.

        Samples azimuth, elevation, and distance uniformly within the ranges
        defined by the module-level ``_CAM_*`` constants, then computes a
        look-at quaternion so the camera frames the table centre.

        MuJoCo cameras look along local **-z** with local **+y** as up.  We
        build the rotation matrix from the sampled position to the look-at
        target and convert it to a quaternion with ``mju_mat2Quat``.
        """
        import mujoco as _mj

        target = np.array(_CAM_TARGET) + self._rng.uniform(
            -_CAM_TARGET_JITTER, _CAM_TARGET_JITTER, 3
        )
        a = np.deg2rad(self._rng.uniform(-_CAM_AZIMUTH_DEG, _CAM_AZIMUTH_DEG))
        e = np.deg2rad(self._rng.uniform(_CAM_ELEVATION_MIN_DEG, _CAM_ELEVATION_MAX_DEG))
        d = self._rng.uniform(_CAM_DISTANCE_MIN_M, _CAM_DISTANCE_MAX_M)

        # Camera sits on the open near side (-y), elevated by e, swept +-a left/right.
        # a=0 -> directly behind the arm (world -y direction from table centre).
        pos = target + d * np.array(
            [
                np.cos(e) * np.sin(a),
                -np.cos(e) * np.cos(a),
                np.sin(e),
            ]
        )

        # Look-at quaternion: camera->world rotation with columns [x_cam, y_cam, z_cam].
        #   z_cam  = -forward  (camera looks along local -z)
        #   x_cam  = normalise(cross(world_up, z_cam))  (rightward)
        #   y_cam  = cross(z_cam, x_cam)                (upward, orthogonalised)
        forward = target - pos
        forward /= np.linalg.norm(forward)
        z_cam = -forward
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(world_up, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        mat = np.array(
            [
                x_cam[0],
                y_cam[0],
                z_cam[0],
                x_cam[1],
                y_cam[1],
                z_cam[1],
                x_cam[2],
                y_cam[2],
                z_cam[2],
            ]
        )
        quat = np.zeros(4)
        _mj.mju_mat2Quat(quat, mat)

        self._env.model.cam_pos[self._camera_id] = pos
        self._env.model.cam_quat[self._camera_id] = quat

    # ---------------------------------------------------------------- YAM control

    def _setup_yam_control(self, env: Any) -> None:
        """Wire the YAM EEController into the Meta-World env via hook injection.

        Steps:
        1. Forward kinematics at the default qpos to capture the TCP orientation
           (gripper-down) as the IK target.
        2. Build an EEController with this orientation locked in.
        3. Inject _external_actuation: loops frame_skip IK steps per outer step,
           negating the gripper channel (MW convention: +1=close; EE: +1=open).
        4. Inject _external_reset_hand: resets arm to init_qpos, syncs ctrl,
           settles physics, and sets init_tcp.
        """
        import mujoco as _mj

        from dreamer_arm.envs.arms.yam import YAM_ARM
        from dreamer_arm.envs.control import EEController

        # Capture the gripper-down TCP orientation from the default state.
        # The XML sets joint2.ref=1.047 and joint3.ref=1.047, so after
        # mj_resetData the arm is already in home (gripper-down) pose.
        scratch = _mj.MjData(env.model)
        _mj.mj_resetData(env.model, scratch)
        _mj.mj_forward(env.model, scratch)
        tcp_id = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_SITE, "grasp_site")
        if tcp_id < 0:
            raise RuntimeError(
                "grasp_site not found in YAM Meta-World model.  "
                "Check that yam_xyz_base.xml is included correctly."
            )
        quat_down = np.zeros(4, dtype=np.float64)
        _mj.mju_mat2Quat(quat_down, scratch.site_xmat[tcp_id])

        # Build YAM_ARM variant with MW-specific settings:
        #   - tcp_target_quat: the gripper-down orientation captured above
        #   - ee_step_m: scaled to match MW's action_scale/frame_skip
        yam_arm_mw = dataclasses.replace(
            YAM_ARM,
            tcp_target_quat=(
                float(quat_down[0]),
                float(quat_down[1]),
                float(quat_down[2]),
                float(quat_down[3]),
            ),
            ee_step_m=_YAM_MW_EE_STEP_M,
        )
        controller = EEController(yam_arm_mw, env.model)
        # Reduce orientation gain for MW: position tracking matters more than
        # holding the exact home orientation.  A small gain still prevents wrist
        # drift without fighting the IK's position component.
        controller._ori_gain = 0.1

        frame_skip: int = int(env.frame_skip)

        # Gripper joint qpos/dof addresses.
        # left_finger is the actuated joint; right_finger mirrors it via equality
        # (polycoef "0 -1 0 0 0" → right = -left).  Both must be kinematically
        # anchored every substep so the equality constraint doesn't generate
        # large corrective impulses during mj_step.
        gripper_jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, "left_finger")
        _gripper_qpos_adr = int(env.model.jnt_qposadr[gripper_jid])
        _gripper_dof_adr = int(env.model.jnt_dofadr[gripper_jid])
        right_jid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, "right_finger")
        _right_qpos_adr = int(env.model.jnt_qposadr[right_jid])
        _right_dof_adr = int(env.model.jnt_dofadr[right_jid])

        def _apply_action(mw_env: Any, action: np.ndarray) -> None:
            """IK-based actuation matching Sawyer's mocap-weld semantics.

            Sawyer's mocap weld re-anchors the hand body every physics step.
            We replicate this by:
              1. Computing IK once to get new joint position targets.
              2. Re-anchoring arm qpos + zeroing arm velocities every substep.
            This ensures the arm holds its new pose against gravity throughout
            the frame_skip window while object physics runs normally.
            Gripper sign negated: MW +1=close, EEController +1=open.
            """
            action_ee = np.array(action, dtype=np.float64)
            action_ee[3] = -action_ee[3]  # negate gripper channel

            # IK: compute new joint position targets → stored in data.ctrl.
            controller.apply(action_ee, mw_env.model, mw_env.data)

            # Cache targets (read once; don't re-run IK per substep).
            q_arm = [float(mw_env.data.ctrl[aid]) for aid in controller._arm_act_ids]
            g_ctrl = float(mw_env.data.ctrl[controller._gripper_act_id])

            for _ in range(frame_skip):
                # Re-anchor arm kinematically (equivalent to mocap weld).
                for qpos_adr, dof_adr, q in zip(
                    controller._arm_qpos_adrs, controller._arm_dof_adrs, q_arm, strict=True
                ):
                    mw_env.data.qpos[qpos_adr] = q
                    mw_env.data.qvel[dof_adr] = 0.0
                mw_env.data.qpos[_gripper_qpos_adr] = g_ctrl
                mw_env.data.qvel[_gripper_dof_adr] = 0.0
                mw_env.data.qpos[_right_qpos_adr] = -g_ctrl  # equality: right = -left
                mw_env.data.qvel[_right_dof_adr] = 0.0
                # Keep ctrl in sync so position actuators generate zero force
                # (arm is held kinematically; only object physics matters).
                for aid, q in zip(controller._arm_act_ids, q_arm, strict=True):
                    mw_env.data.ctrl[aid] = q
                mw_env.data.ctrl[controller._gripper_act_id] = g_ctrl
                _mj.mj_step(mw_env.model, mw_env.data)

        def _reset_hand(mw_env: Any, steps: int = 50) -> None:
            """Reset arm to home pose kinematically, then settle object physics."""
            mw_env.set_state(mw_env.init_qpos, mw_env.init_qvel)
            # Sync ctrl to qpos so actuators hold the pose between steps.
            for i, aid in enumerate(controller._arm_act_ids):
                mw_env.data.ctrl[aid] = mw_env.data.qpos[controller._arm_qpos_adrs[i]]
            mw_env.data.ctrl[controller._gripper_act_id] = controller._g_lo
            _mj.mj_forward(mw_env.model, mw_env.data)
            for _ in range(steps):
                _mj.mj_step(mw_env.model, mw_env.data)
            mw_env.init_tcp = mw_env.tcp_center

        env._external_actuation = _apply_action
        env._external_reset_hand = _reset_hand
        env._yam_controller = controller  # expose for tuning / testing

    # ---------------------------------------------------------------- gym API

    def _render_camera(self, cam_id: int, flip: bool) -> np.ndarray:
        self._renderer.update_scene(self._env.data, camera=cam_id)
        frame: np.ndarray = self._renderer.render()
        return np.flip(frame, axis=0) if flip else frame

    def _make_obs(self, state: np.ndarray) -> ObsDict:
        """Build the Dreamer obs dict, adding the one-hot ``task_id`` if present."""
        obs: ObsDict = {
            "scene": self.render(),
            "state": np.asarray(state, dtype=np.float32),
        }
        if self._wrist_camera_id is not None:
            wrist = self._render_camera(self._wrist_camera_id, flip=False)
            if self._fisheye_map is not None:
                wrist = _apply_fisheye(wrist, self._fisheye_map)
            obs["wrist_image"] = wrist
        if self._task_onehot is not None:
            obs["task_id"] = self._task_onehot
        return obs

    def _info(self, **extra: Any) -> dict[str, Any]:
        """Step/reset info, tagged with the task name for per-task logging."""
        return {"task": self._name, **extra}

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

        # Camera randomization: full hemisphere pose OR small position jitter.
        if self._camera_pose_randomize:
            self._sample_camera_pose()
            # camera_jitter stacks on top of the sampled pose as extra noise.
            if self._camera_jitter > 0.0:
                j = self._camera_jitter
                self._env.model.cam_pos[self._camera_id] += self._rng.uniform(-j, j, 3)
        elif self._camera_jitter > 0.0:
            j = self._camera_jitter
            self._env.model.cam_pos[self._camera_id] = self._cam_base_pos + self._rng.uniform(
                -j, j, 3
            )

        # Appearance randomization: textures + RGBA tints + lighting.
        if self._scene_randomize:
            self._apply_scene_dr()

        state, _ = self._env.reset()
        if self._viewer_requested:
            import mujoco.viewer as _mv

            self._passive_viewer = _mv.launch_passive(self._env.model, self._env.data)
            self._viewer_requested = False
        self._sync_viewer()
        return self._make_obs(state), self._info()

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
        return (
            self._make_obs(state),
            total_reward,
            terminated,
            truncated,
            self._info(success=self._success),
        )

    def render(self) -> np.ndarray:
        return self._render_camera(self._camera_id, flip=self._scene_flip)

    def _sync_viewer(self) -> None:
        if self._passive_viewer is not None and self._passive_viewer.is_running():
            self._passive_viewer.sync()

    def close(self) -> None:
        self._renderer.close()
        if self._passive_viewer is not None:
            self._passive_viewer.close()
        self._env.close()
