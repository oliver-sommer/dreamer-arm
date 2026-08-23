"""Image and multimodal encoder/decoder networks."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from dreamer_arm.core import distributions as dists
from dreamer_arm.core.networks.heads import MLP, MLPHead
from dreamer_arm.core.networks.layers import BlockLinear, Conv2dSamePad, RMSNorm2D, weight_init_


class ConvEncoder(nn.Module):
    def __init__(self, config: Any, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        act_cls = getattr(nn, config.act)
        height, width, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(multiplier) for multiplier in config.mults)
        self.kernel_size = int(config.kernel_size)
        layers: list[nn.Module] = []
        in_dim = input_ch
        for depth in self.depths:
            layers.append(Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True))
            layers.append(nn.MaxPool2d(2, 2))
            if bool(config.norm):
                layers.append(RMSNorm2D(depth, eps=1e-4, dtype=torch.float32))
            layers.append(act_cls())
            in_dim = depth
            height, width = height // 2, width // 2
        self.out_dim = self.depths[-1] * height * width
        self.layers = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs - 0.5
        batch_time = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:]).permute(0, 3, 1, 2)
        x = self.layers(x).reshape(x.shape[0], -1)
        return x.reshape(*batch_time, x.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        config: Any,
        deter: int,
        flat_stoch: int,
        shape: tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        super().__init__()
        act_cls = getattr(nn, config.act)
        self._shape = shape
        self.depths = tuple(int(config.depth) * int(multiplier) for multiplier in config.mults)
        factor = 2 ** len(self.depths)
        minres = [int(value // factor) for value in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)

        units = math.prod(self.min_shape)
        self.sp0 = BlockLinear(deter, units, self.bspace)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units),
            nn.RMSNorm(2 * self.units, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self.sp2 = nn.Linear(2 * self.units, units)
        self.sp_norm = nn.Sequential(nn.RMSNorm(self.depths[-1], eps=1e-4, dtype=torch.float32), act_cls())

        layers: list[nn.Module] = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True),
                    RMSNorm2D(depth, eps=1e-4, dtype=torch.float32),
                    act_cls(),
                ]
            )
            in_dim = depth
        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        batch_time = deter.shape[:-1]
        count = math.prod(batch_time) if batch_time else 1
        x0 = deter.reshape(count, deter.shape[-1])
        x1 = stoch.reshape(count, -1)
        height, width, channels = self.min_shape
        x0 = self.sp0(x0).reshape(-1, self.bspace, height, width, channels // self.bspace)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, height, width, channels)
        x1 = self.sp2(self.sp1(x1)).reshape(-1, height, width, channels)
        x = self.sp_norm(x0 + x1).permute(0, 3, 1, 2)
        x = torch.sigmoid(self.layers(x).permute(0, 2, 3, 1))
        return x.reshape(*batch_time, *x.shape[1:])


_EXCLUDED_OBS_KEYS = ("is_first", "is_last", "is_terminal", "reward")


class MultiEncoder(nn.Module):
    def __init__(self, config: Any, shapes: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        shapes = {
            key: value for key, value in shapes.items() if key not in _EXCLUDED_OBS_KEYS and not key.startswith("log_")
        }
        self.cnn_shapes = {
            key: value for key, value in shapes.items() if len(value) == 3 and re.match(config.cnn_keys, key)
        }
        self.mlp_shapes = {
            key: value for key, value in shapes.items() if len(value) in (1, 2) and re.match(config.mlp_keys, key)
        }
        encoders: list[nn.Module] = []
        self._has_cnn = bool(self.cnn_shapes)
        self._has_mlp = bool(self.mlp_shapes)
        self.out_dim = 0
        if self._has_cnn:
            input_ch = sum(value[-1] for value in self.cnn_shapes.values())
            input_shape = (*next(iter(self.cnn_shapes.values()))[:2], input_ch)
            self.cnn = ConvEncoder(config.cnn, input_shape)
            encoders.append(self.cnn)
            self.out_dim += self.cnn.out_dim
        if self._has_mlp:
            inp_dim = sum(sum(value) for value in self.mlp_shapes.values())
            self.mlp = MLP(config.mlp, inp_dim)
            encoders.append(self.mlp)
            self.out_dim += self.mlp.out_dim
        if not encoders:
            raise ValueError("MultiEncoder requires at least one of cnn / mlp inputs.")
        self.encoders = nn.ModuleList(encoders)
        self.apply(weight_init_)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        if self._has_cnn:
            outputs.append(self.cnn(torch.cat([obs[key] for key in self.cnn_shapes], dim=-1)))
        if self._has_mlp:
            outputs.append(self.mlp(torch.cat([obs[key] for key in self.mlp_shapes], dim=-1)))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


class MultiDecoder(nn.Module):
    def __init__(self, config: Any, deter: int, flat_stoch: int, shapes: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {key: value for key, value in shapes.items() if key not in excluded}
        self.cnn_shapes = {
            key: value for key, value in shapes.items() if len(value) == 3 and re.match(config.cnn_keys, key)
        }
        self.mlp_shapes = {
            key: value for key, value in shapes.items() if len(value) in (1, 2) and re.match(config.mlp_keys, key)
        }
        self._has_cnn = bool(self.cnn_shapes)
        self._has_mlp = bool(self.mlp_shapes)
        if self._has_cnn:
            example = next(iter(self.cnn_shapes.values()))
            out_shape = (sum(value[-1] for value in self.cnn_shapes.values()), *example[:-1])
            self.cnn = ConvDecoder(config.cnn, deter, flat_stoch, out_shape)
            self._image_dist: Callable[[torch.Tensor], Any] = partial(
                getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist
            )
        if self._has_mlp:
            shape = (sum(sum(value) for value in self.mlp_shapes.values()),)
            mlp_cfg = copy.deepcopy(config.mlp)
            OmegaConf.set_readonly(mlp_cfg, False)
            mlp_cfg.shape = shape
            self.mlp = MLPHead(mlp_cfg, deter + flat_stoch)
            self._mlp_dist: Callable[[torch.Tensor], Any] = partial(
                getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist
            )

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor) -> dict[str, Any]:
        output: dict[str, Any] = {}
        if self._has_cnn:
            chunks = torch.split(self.cnn(stoch, deter), [value[-1] for value in self.cnn_shapes.values()], dim=-1)
            for key, chunk in zip(self.cnn_shapes, chunks, strict=True):
                output[key] = self._image_dist(chunk)
        if self._has_mlp:
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], dim=-1)
            chunks = torch.split(self.mlp(feat), [int(sum(value)) for value in self.mlp_shapes.values()], dim=-1)
            for key, chunk in zip(self.mlp_shapes, chunks, strict=True):
                output[key] = self._mlp_dist(chunk)
        return output
