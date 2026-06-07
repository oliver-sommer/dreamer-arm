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


class MetaWorld(gym.Env):  # type: ignore[type-arg]
    """Single Meta-World MT1 task as a Gymnasium env with a Dict obs space.

    The obs dict carries ``image`` (uint8 RGB at ``size``) and ``state``
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

        obs_spaces: dict[str, gym.spaces.Space] = {  # type: ignore[type-arg]
            "image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8),
            "state": env.observation_space,
        }
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

    # ---------------------------------------------------------------- gym API

    def _make_obs(self, state: np.ndarray) -> ObsDict:
        """Build the Dreamer obs dict, adding the one-hot ``task_id`` if present."""
        obs: ObsDict = {
            "image": self.render(),
            "state": np.asarray(state, dtype=np.float32),
        }
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
