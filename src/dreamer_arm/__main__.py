"""Entry point for ``python -m dreamer_arm`` (also ``pixi run train``).

This is a thin Hydra wrapper around :func:`run`: build envs → agent →
buffer → logger → trainer, then call ``trainer.begin(agent)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dreamer_arm.agent.dreamer import Dreamer
from dreamer_arm.data.buffer import BufferConfig, ReplayBuffer
from dreamer_arm.envs.factory import make_vector_env
from dreamer_arm.train.logger import WandbLogger
from dreamer_arm.train.trainer import OnlineTrainer, TrainerConfig
from dreamer_arm.utils.seed import set_seed_everywhere

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def run(cfg: DictConfig) -> None:
    """Build the full Dreamer stack from ``cfg`` and run the online loop."""
    set_seed_everywhere(int(cfg.seed))

    env_name = f"{cfg.env.name}:{cfg.env.task}"

    def _make_envs() -> Any:
        return make_vector_env(
            env_name,
            num_envs=int(cfg.env.env_num),
            seed=int(cfg.env.seed),
            size=tuple(cfg.env.size),
            action_repeat=int(cfg.env.action_repeat),
            time_limit=int(cfg.env.time_limit),
        )

    train_envs = _make_envs()
    eval_envs = _make_envs() if int(cfg.env.eval_episode_num) > 0 else None

    agent = Dreamer(cfg.model, train_envs.observation_space, train_envs.action_space).to(
        cfg.device
    )

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
        name=str(cfg.wandb.name),
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
        video_pred_log=bool(cfg.trainer.video_pred_log),
        params_hist_log=bool(cfg.trainer.params_hist_log),
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
