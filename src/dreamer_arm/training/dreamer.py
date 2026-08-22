"""Composition root for online Dreamer training.

Turns a resolved config into the objects a run needs -- envs, agent, buffer,
logger -- then hands them to :class:`~dreamer_arm.training.trainer.OnlineTrainer`.
Named to mirror ``configs/training/dreamer.yaml``, which points its
``entrypoint._target_`` at :func:`_run`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from dreamer_arm.core.buffer import BufferConfig, ReplayBuffer
from dreamer_arm.core.model import Dreamer
from dreamer_arm.envs.factory import build_from_config
from dreamer_arm.training.trainer import OnlineTrainer, TrainerConfig
from dreamer_arm.utils.seed import set_seed_everywhere
from dreamer_arm.utils.tracking import WandbLogger

log = logging.getLogger(__name__)


def _log_run_shape(env_name: str, cfg: DictConfig, envs: Any, agent: Any) -> None:
    """Print spaces and parameter count up front, so a run's shape is visible immediately."""
    log.info("env %s | %d envs | device %s", env_name, int(cfg.envs.env_num), cfg.device)
    log.info("observation space:")
    for key in sorted(envs.observation_space.spaces):
        space = envs.observation_space.spaces[key]
        log.info("  %-12s shape=%s dtype=%s", key, tuple(space.shape), space.dtype)
    log.info("action space   %s", envs.action_space)
    log.info("agent %s parameters", f"{sum(p.numel() for p in agent.parameters()):,}")


def _run(cfg: DictConfig) -> None:
    """Build the full Dreamer stack from ``cfg`` and run the online loop."""
    set_seed_everywhere(int(cfg.seed))

    train_envs = build_from_config(cfg, viewer=bool(cfg.envs.get("viewer", False)))
    # Reuse train_envs for eval to avoid doubling the EGL renderer count.
    # With MT50 (50 train + 50 eval envs), 100 EGL contexts exhaust VRAM on
    # most GPUs.  The trainer resets shared envs after eval to resync obs state.
    eval_envs = train_envs if int(cfg.envs.eval_episode_num) > 0 else None

    agent = Dreamer(cfg.core.model, train_envs.observation_space, train_envs.action_space).to(cfg.device)
    _log_run_shape(f"{cfg.envs.name}:{cfg.envs.task}", cfg, train_envs, agent)

    buffer = ReplayBuffer(
        BufferConfig(
            max_size=int(cfg.core.buffer.max_size),
            batch_size=int(cfg.core.buffer.batch_size),
            batch_length=int(cfg.core.buffer.batch_length),
            device=str(cfg.core.buffer.device),
            storage_device=str(cfg.core.buffer.storage_device),
        )
    )

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        project=str(cfg.logging.wandb.project),
        config=wandb_cfg if isinstance(wandb_cfg, dict) else None,  # type: ignore[arg-type]
        name=str(cfg.logging.wandb.name) if cfg.logging.wandb.name is not None else None,
        entity=cfg.logging.wandb.entity,
        tags=list(cfg.logging.wandb.tags) if cfg.logging.wandb.tags else None,
        mode=cfg.logging.wandb.mode,
        logdir=str(cfg.logdir),
        sync_interval=float(cfg.logging.wandb.get("sync_interval", 600.0)),
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
        checkpoint_every=int(cfg.trainer.checkpoint_every),
        checkpoint_path=str(Path(cfg.logdir) / "checkpoint.pt"),
    )

    # Crash-resume: restore agent weights/optimiser and continue the step
    # counter.  The replay buffer is not part of the checkpoint; it refills
    # from scratch (model updates pause until the warmup minimum is met).
    start_step = 0
    if cfg.resume is not None:
        ckpt = torch.load(str(cfg.resume), map_location=cfg.device, weights_only=False)
        agent.load_checkpoint_state(ckpt["agent"])
        start_step = int(ckpt["step"])
        log.info("resumed from %s at step %d", cfg.resume, start_step)

    try:
        trainer = OnlineTrainer(trainer_cfg, buffer, logger, train_envs, eval_envs)
        trainer.begin(agent, start_step=start_step)
    finally:
        logger.finish()
        train_envs.close()
        if eval_envs is not None and eval_envs is not train_envs:
            eval_envs.close()
