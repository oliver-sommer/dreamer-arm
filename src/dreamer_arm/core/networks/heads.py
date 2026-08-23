"""Generic MLP backbones and distribution-producing heads."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn

from dreamer_arm.core import distributions as dists
from dreamer_arm.core.networks.layers import weight_init_
from dreamer_arm.utils.tensor import symlog


class MLP(nn.Module):
    def __init__(self, config: Any, inp_dim: int) -> None:
        super().__init__()
        act_cls = getattr(nn, config.act)
        self._symlog_inputs = bool(getattr(config, "symlog_inputs", False))
        self.layers = nn.Sequential()
        for index in range(int(config.layers)):
            self.layers.add_module(f"{config.name}_linear{index}", nn.Linear(inp_dim, int(config.units)))
            self.layers.add_module(
                f"{config.name}_norm{index}",
                nn.RMSNorm(int(config.units), eps=1e-4, dtype=torch.float32),
            )
            self.layers.add_module(f"{config.name}_act{index}", act_cls())
            inp_dim = int(config.units)
        self.out_dim = int(config.units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(symlog(x) if self._symlog_inputs else x)


class MLPHead(nn.Module):
    def __init__(self, config: Any, inp_dim: int) -> None:
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(getattr(config, "outscale", 1.0))
        dist_factory: Callable[..., Any] = getattr(dists, self._dist_name)

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2)
            kwargs: dict[str, Any] = {
                "min_std": float(config.dist.min_std),
                "max_std": float(config.dist.max_std),
            }
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0])
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim, sum(config.shape))
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio), "shape": tuple(config.shape)}
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0])
            kwargs = {"bin_num": int(config.dist.bin_num)}
        elif self._dist_name in ("binary", "identity"):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0])
            kwargs = {}
        else:
            raise NotImplementedError(self._dist_name)

        self._dist: Callable[[torch.Tensor], Any] = partial(dist_factory, **kwargs)
        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x: torch.Tensor) -> Any:
        return self._dist(self.last(self.mlp(x)))
