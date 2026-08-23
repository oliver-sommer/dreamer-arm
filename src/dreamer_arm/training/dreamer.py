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
from dreamer_arm.envs.sim.factory import build_from_config
from dreamer_arm.training.trainer import OnlineTrainer, TrainerConfig
from dreamer_arm.utils.seed import set_seed_everywhere
from dreamer_arm.utils.tracking import WandbLogger

log = logging.getLogger(__name__)


def _log_run_shape(env_name: str, cfg: DictConfig, envs: Any, agent: Any) -> None:
    """Print spaces and parameter count up front, so a run's shape is visible immediately."""
    log.info("env %s | %d envs | device %s", env_name, int(cfg.envs.sim.env_num), cfg.device)
    log.info("observation space:")
    for key in sorted(envs.observation_space.spaces):
        space = envs.observation_space.spaces[key]
        log.info("  %-12s shape=%s dtype=%s", key, tuple(space.shape), space.dtype)
    log.info("action space   %s", envs.action_space)
    log.info("agent %s parameters", f"{sum(p.numel() for p in agent.parameters()):,}")


def _restore_buffer(checkpoint: Path, buffer: Any) -> None:
    """Load the replay buffer saved beside ``checkpoint``, if there is one.

    The trainer writes the dump as ``<stem>_buffer`` next to the file it
    accompanies, and only ever for ``latest.pt``.  Resolving by that convention
    means ``best.pt`` and the archives -- which carry no buffer -- simply find
    nothing and start cold, with no extra config to get wrong.

    A buffer that fails to load is not fatal: the run continues on an empty one,
    which is exactly the old behaviour.
    """
    path = checkpoint.with_name(f"{checkpoint.stem}_buffer")
    if not path.is_dir():
        log.info("no replay buffer beside %s; starting with an empty buffer", checkpoint.name)
        return
    try:
        buffer.load(path)
        log.info("restored replay buffer from %s (%d transitions)", path, len(buffer))
    except Exception:  # noqa: BLE001 - a bad dump must not block a resume
        log.exception("replay buffer restore failed (%s); starting with an empty buffer", path)


def _run(cfg: DictConfig) -> None:
    """Build the full Dreamer stack from ``cfg`` and run the online loop."""
    set_seed_everywhere(int(cfg.seed))

    train_envs = build_from_config(cfg, viewer=bool(cfg.envs.sim.get("viewer", False)))
    # Reuse train_envs for eval to avoid doubling the EGL renderer count.
    # With MT50 (50 train + 50 eval envs), 100 EGL contexts exhaust VRAM on
    # most GPUs.  The trainer resets shared envs after eval to resync obs state.
    eval_envs = train_envs if int(cfg.envs.sim.eval_episode_num) > 0 else None

    agent = Dreamer(cfg.core.model, train_envs.observation_space, train_envs.action_space).to(cfg.device)
    _log_run_shape(f"{cfg.envs.sim.name}:{cfg.envs.sim.task}", cfg, train_envs, agent)

    buffer = ReplayBuffer(
        BufferConfig(
            max_size=int(cfg.core.buffer.max_size),
            batch_size=int(cfg.core.buffer.batch_size),
            batch_length=int(cfg.core.buffer.batch_length),
            device=str(cfg.core.buffer.device),
            storage_device=str(cfg.core.buffer.storage_device),
            prefetch=int(cfg.core.buffer.prefetch),
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
        checkpoint_dir=str(cfg.logdir),
        checkpoint_keep_every=int(cfg.trainer.checkpoint_keep_every),
        checkpoint_buffer=bool(cfg.trainer.checkpoint_buffer),
        eval_at_start=bool(cfg.trainer.eval_at_start),
        eval_warmup_steps=tuple(int(step) for step in cfg.trainer.eval_warmup_steps),
        heartbeat_secs=float(cfg.trainer.heartbeat_secs),
    )

    # Crash-resume: restore agent weights/optimiser, the step counter, and the
    # trainer's own counters (episode ids, best eval score).
    start_step = 0
    trainer_state: dict[str, Any] = {}
    if cfg.resume is not None:
        ckpt = torch.load(str(cfg.resume), map_location=cfg.device, weights_only=False)
        agent.load_checkpoint_state(ckpt["agent"])
        start_step = int(ckpt["step"])
        trainer_state = dict(ckpt.get("trainer", {}))
        log.info("resumed from %s at step %d", cfg.resume, start_step)
        _restore_buffer(Path(str(cfg.resume)), buffer)

    try:
        trainer = OnlineTrainer(trainer_cfg, buffer, logger, train_envs, eval_envs)
        trainer.begin(agent, start_step=start_step, trainer_state=trainer_state)
    finally:
        logger.finish()
        train_envs.close()
        if eval_envs is not None and eval_envs is not train_envs:
            eval_envs.close()
