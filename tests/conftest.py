"""Shared test fixtures.

The fixtures here build a *tiny* Dreamer config so unit tests fit in CPU
RAM and finish in seconds. Production hyperparameters are not used for
testing — we just check shapes, gradient flow and loss-formula correctness.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf


def _ns(d: dict[str, Any]) -> Any:
    """Recursive dict → namespace, so ``cfg.foo.bar`` indexing works in tests."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_rssm_cfg() -> Any:
    return _ns(
        {
            "stoch": 4,
            "discrete": 4,
            "deter": 32,
            "hidden": 32,
            "obs_layers": 1,
            "img_layers": 1,
            "dyn_layers": 1,
            "blocks": 4,
            "act": "SiLU",
            "norm": True,
            "unimix_ratio": 0.01,
            "initial": "learned",
            "device": "cpu",
        }
    )


@pytest.fixture
def tiny_encoder_cfg() -> Any:
    return _ns(
        {
            "mlp_keys": "state",
            "cnn_keys": "scene",
            "mlp": {
                "shape": None,
                "layers": 1,
                "units": 16,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "outscale": None,
                "symlog_inputs": True,
                "name": "mlp_encoder",
            },
            "cnn": {
                "act": "SiLU",
                "norm": True,
                "kernel_size": 5,
                "minres": 4,
                "depth": 4,
                "mults": [1, 1, 1, 1],
            },
        }
    )


@pytest.fixture
def tiny_dino_encoder_cfg() -> Any:
    """``pretrained=False`` and a small ``image_size`` -- random weights, no
    network access, but the real architecture (patch size, register tokens).
    """
    return _ns({"name": "vit_small_patch16_dinov3.lvd1689m", "pretrained": False, "image_size": 32})


@pytest.fixture
def tiny_actor_critic_cfg() -> Any:
    """Minimal config for :class:`~dreamer_arm.core.actor_critic.ActorCritic`.

    A real ``DictConfig`` (not the plain ``_ns`` namespace the RSSM/DINO-WM
    fixtures use): ``ActorCritic.__init__`` calls ``OmegaConf.set_readonly``
    on its local copy of the actor config, which needs an actual OmegaConf
    container.
    """

    def _head(dist_name: str, shape: list[int], **dist_kwargs: Any) -> dict[str, Any]:
        return {
            "shape": shape,
            "layers": 1,
            "units": 16,
            "act": "SiLU",
            "norm": True,
            "device": "cpu",
            "outscale": 1.0,
            "symlog_inputs": False,
            "name": "head",
            "dist": {"name": dist_name, **dist_kwargs},
        }

    return OmegaConf.create(
        {
            "imag_horizon": 3,
            "horizon": 10,
            "lamb": 0.95,
            "act_entropy": 3e-4,
            "constraint_cost_scale": 0.0,
            "slow_target_update": 1,
            "slow_target_fraction": 0.02,
            "reward": _head("symexp_twohot", [255], bin_num=255),
            "cont": _head("binary", [1]),
            "success": {
                "enabled": False,
                "bonus": 0.0,
                **_head("binary", [1]),
            },
            "critic": _head("symexp_twohot", [255], bin_num=255),
            "actor": {
                "shape": None,
                "layers": 1,
                "units": 16,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "outscale": 1.0,
                "symlog_inputs": False,
                "name": "actor",
                "dist": {
                    "cont": {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                    "disc": {"name": "onehot", "unimix_ratio": 0.01},
                },
            },
        }
    )


@pytest.fixture
def tiny_dinowm_cfg() -> Any:
    return _ns(
        {
            "context": 3,
            "action_dim_embed": 4,
            "feat_pool": "task_attention",
            "pool_heads": 4,
            "cnn_keys": "scene",
            "mlp_keys": "proprio",
            "predictor": {"depth": 1, "heads": 2, "dim_head": 8, "mlp_dim": 16},
        }
    )
