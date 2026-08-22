"""Online training: the collect -> update -> eval -> checkpoint loop.

:mod:`dreamer_arm.training.dreamer` is the composition root invoked by Hydra
(it turns a config into envs, agent, buffer and logger); the loop itself is
:class:`dreamer_arm.training.trainer.OnlineTrainer`.
"""

from dreamer_arm.training.trainer import OnlineTrainer, TrainerConfig

__all__ = ["OnlineTrainer", "TrainerConfig"]
