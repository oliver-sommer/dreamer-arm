"""Entry point for ``python -m dreamer_arm`` (also ``pixi run train``).

This is a thin Hydra wrapper around :func:`run`: build envs → agent →
buffer → logger → trainer, then call ``trainer.begin(agent)``.
"""

from __future__ import annotations

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

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


OmegaConf.register_new_resolver("auto_device", _auto_device, replace=True)


def run(cfg: DictConfig) -> None:
    """Build the full Dreamer stack from ``cfg`` and run the online loop."""
    set_seed_everywhere(int(cfg.seed))

    env_name = f"{cfg.task.name}:{cfg.task.task}"

    _viewer = bool(cfg.task.get("viewer", False))  # type: ignore[union-attr]

    def _make_envs(*, viewer: bool = False) -> Any:
        # Forward only the optional kwargs defined by the active task config.
        # This keeps dmc_vision (no success_threshold) and manip (no camera)
        # from crashing on each other's keys.
        extra: dict[str, Any] = {}
        if "success_threshold" in cfg.task:
            extra["success_threshold"] = float(cfg.task.success_threshold)
        if "camera" in cfg.task:
            extra["camera"] = str(cfg.task.camera)
        # Forward the arm name for manip envs; ignored by dmc.
        if hasattr(cfg, "arm") and cfg.arm is not None:
            extra["arm"] = str(cfg.arm.name)
        return make_vector_env(
            env_name,
            num_envs=int(cfg.task.env_num),
            seed=int(cfg.task.seed),
            size=tuple(cfg.task.size),
            action_repeat=int(cfg.task.action_repeat),
            time_limit=int(cfg.task.time_limit),
            viewer=viewer,
            **extra,
        )

    train_envs = _make_envs(viewer=_viewer)
    eval_envs = _make_envs() if int(cfg.task.eval_episode_num) > 0 else None

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
        if eval_envs is not None:
            eval_envs.close()


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
