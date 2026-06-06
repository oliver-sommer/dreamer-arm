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
            "cnn_keys": "image",
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
