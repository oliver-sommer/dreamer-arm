"""Model, algorithm and replay: everything the agent *is*, independent of how it is run.

Holds the network architectures (:mod:`~dreamer_arm.core.networks`, per-world-
model modules under :mod:`~dreamer_arm.core.world_model`), the
:class:`~dreamer_arm.core.actor_critic.ActorCritic`, the :class:`Dreamer` agent
that composes them, the loss functions used to train it, the optimizer
building blocks, and the trajectory replay buffer.

The *orchestration* of a run — collection loop, logging cadence,
checkpointing — lives in :mod:`dreamer_arm.training`, and rollout-only
evaluation in :mod:`dreamer_arm.inference`.
"""

from dreamer_arm.core.actor_critic import ActorCritic, ReturnEMA
from dreamer_arm.core.buffer import BufferConfig, ReplayBuffer
from dreamer_arm.core.losses import barlow_twins_loss, lambda_return
from dreamer_arm.core.model import Dreamer
from dreamer_arm.core.networks import (
    MLP,
    BlockLinear,
    Conv2dSamePad,
    ConvDecoder,
    ConvEncoder,
    MLPHead,
    MultiDecoder,
    MultiEncoder,
    RMSNorm2D,
)
from dreamer_arm.core.optim import LaProp, adaptive_grad_clip
from dreamer_arm.core.world_model.rssm import RSSM, Deter, Projector

__all__ = [
    "MLP",
    "RSSM",
    "ActorCritic",
    "BlockLinear",
    "BufferConfig",
    "Conv2dSamePad",
    "ConvDecoder",
    "ConvEncoder",
    "Deter",
    "Dreamer",
    "LaProp",
    "MLPHead",
    "MultiDecoder",
    "MultiEncoder",
    "Projector",
    "RMSNorm2D",
    "ReplayBuffer",
    "ReturnEMA",
    "adaptive_grad_clip",
    "barlow_twins_loss",
    "lambda_return",
]
