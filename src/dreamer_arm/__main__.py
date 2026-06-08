"""Entry point for ``python -m dreamer_arm`` (also ``pixi run train``).

This is a thin Hydra wrapper around :func:`run`: build envs → agent →
buffer → logger → trainer, then call ``trainer.begin(agent)``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dreamer_arm.agent.dreamer import Dreamer
from dreamer_arm.data.buffer import BufferConfig, ReplayBuffer
from dreamer_arm.envs.factory import make_vector_env
from dreamer_arm.train.logger import WandbLogger
from dreamer_arm.train.trainer import OnlineTrainer, TrainerConfig
from dreamer_arm.utils.seed import set_seed_everywhere

# Must be set before any MuJoCo context is created.  On headless Linux servers
# MuJoCo defaults to GLFW, which requires an X display.  EGL is the right
# backend for GPU-equipped servers; on CPU-only machines export
# MUJOCO_GL=osmesa before launching.  setdefault preserves any user override.
os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "cgl")

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_compile() -> bool:
    return torch.cuda.is_available()


OmegaConf.register_new_resolver("auto_device", _auto_device, replace=True)
OmegaConf.register_new_resolver("auto_compile", _auto_compile, replace=True)


def run(cfg: DictConfig) -> None:
    """Build the full Dreamer stack from ``cfg`` and run the online loop."""
    set_seed_everywhere(int(cfg.seed))

    env_name = f"{cfg.envs.name}:{cfg.envs.task}"

    _viewer = bool(cfg.envs.get("viewer", False))

    def _make_envs(*, viewer: bool = False) -> Any:
        # Forward only the optional kwargs defined by the active env config.
        extra: dict[str, Any] = {}
        if "success_threshold" in cfg.envs:
            extra["success_threshold"] = float(cfg.envs.success_threshold)
        if "camera" in cfg.envs:
            extra["camera"] = str(cfg.envs.camera)
        if "wrist_camera" in cfg.envs and cfg.envs.wrist_camera is not None:
            extra["wrist_camera"] = str(cfg.envs.wrist_camera)
        if "camera_jitter" in cfg.envs:
            extra["camera_jitter"] = float(cfg.envs.camera_jitter)
        if "scene_randomize" in cfg.envs:
            extra["scene_randomize"] = bool(cfg.envs.scene_randomize)
        if "camera_pose_randomize" in cfg.envs:
            extra["camera_pose_randomize"] = bool(cfg.envs.camera_pose_randomize)
        if hasattr(cfg, "arm") and cfg.arm is not None:
            extra["arm"] = str(cfg.arm.name)
        if "action_rate_cost" in cfg.envs:
            extra["action_rate_cost"] = float(cfg.envs.action_rate_cost)
        if "action_mag_cost" in cfg.envs:
            extra["action_mag_cost"] = float(cfg.envs.action_mag_cost)
        return make_vector_env(
            env_name,
            num_envs=int(cfg.envs.env_num),
            seed=int(cfg.envs.seed),
            size=tuple(cfg.envs.size),
            action_repeat=int(cfg.envs.action_repeat),
            time_limit=int(cfg.envs.time_limit),
            viewer=viewer,
            **extra,
        )

    train_envs = _make_envs(viewer=_viewer)
    # Reuse train_envs for eval to avoid doubling the EGL renderer count.
    # With MT50 (50 train + 50 eval envs), 100 EGL contexts exhaust VRAM on
    # most GPUs.  The trainer resets shared envs after eval to resync obs state.
    eval_envs = train_envs if int(cfg.envs.eval_episode_num) > 0 else None

    agent = Dreamer(cfg.model, train_envs.observation_space, train_envs.action_space).to(cfg.device)

    buffer = ReplayBuffer(
        BufferConfig(
            max_size=int(cfg.buffer.max_size),
            batch_size=int(cfg.buffer.batch_size),
            batch_length=int(cfg.buffer.batch_length),
            device=str(cfg.buffer.device),
            storage_device=str(cfg.buffer.storage_device),
        )
    )

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        project=str(cfg.wandb.project),
        config=wandb_cfg if isinstance(wandb_cfg, dict) else None,  # type: ignore[arg-type]
        name=str(cfg.wandb.name) if cfg.wandb.name is not None else None,
        entity=cfg.wandb.entity,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
        mode=cfg.wandb.mode,
        logdir=str(cfg.logdir),
    )

    trainer_cfg = TrainerConfig(
        steps=int(cfg.trainer.steps),
        pretrain=int(cfg.trainer.pretrain),
        train_ratio=float(cfg.trainer.train_ratio),
        batch_size=int(cfg.trainer.batch_size),
        batch_length=int(cfg.trainer.batch_length),
        action_repeat=int(cfg.trainer.action_repeat),
        eval_every=int(cfg.trainer.eval_every),
        eval_episode_num=int(cfg.trainer.eval_episode_num),
        update_log_every=int(cfg.trainer.update_log_every),
    )

    try:
        trainer = OnlineTrainer(trainer_cfg, buffer, logger, train_envs, eval_envs)
        trainer.begin(agent)
    finally:
        logger.finish()
        train_envs.close()
        if eval_envs is not None and eval_envs is not train_envs:
            eval_envs.close()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
