"""Neural-network building blocks used by the world model and actor-critic.

Covers:
- :class:`BlockLinear`, :class:`Conv2dSamePad`, :class:`RMSNorm2D` — primitive
  layers with block-wise / TF-style behaviours.
- :class:`ConvEncoder`, :class:`ConvDecoder` — image encoder / decoder.
- :class:`MultiEncoder`, :class:`MultiDecoder` — fuse image + state streams.
- :class:`MLP`, :class:`MLPHead` — generic MLP backbone and a distributional
  head whose output type is selected by ``config.dist.name`` (see
  :mod:`dreamer_arm.core.distributions`).

:class:`~dreamer_arm.core.world_model.rssm.Projector` (the R2-Dreamer
Barlow-Twins head) and :class:`~dreamer_arm.core.actor_critic.ReturnEMA`
(λ-return normalisation) live with their one respective consumer instead of
here.

Configs are expected to be OmegaConf ``DictConfig`` nodes; runtime parsing
into plain Python ints/floats happens inside ``__init__`` so the layers can
be reused without Hydra.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dreamer_arm.core import distributions as dists
from dreamer_arm.utils.modules import weight_init_
from dreamer_arm.utils.tensor import symlog

# ---------------- primitive layers ----------------


class BlockLinear(nn.Module):
    """Block-diagonal linear layer.

    Weight is stored as ``(out_ch / blocks, in_ch / blocks, blocks)`` so that
    PyTorch's fan-in / fan-out calculation in init produces correct values.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0) -> None:
        super().__init__()
        assert in_ch % blocks == 0 and out_ch % blocks == 0
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., I) → (..., O)`` with per-block matmul."""
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """Conv2d that emulates TensorFlow's ``padding="SAME"`` behaviour."""

    @staticmethod
    def _calc_same_pad(i: int, k: int, s: int, d: int) -> int:
        ceil = (i + s - 1) // s
        return max((ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h or pad_w:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over the channel dim of a 4D ``(B, C, H, W)`` tensor."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype: torch.dtype | None = None) -> None:
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _StochReshape(nn.Module):
    """Reshape ``(..., stoch * discrete) → (..., stoch, discrete)`` as a Module.

    Replaces the original ``LambdaLayer`` so the module graph is fully
    serialisable and friendly to torch.compile / state_dict round-trips.
    """

    def __init__(self, stoch: int, discrete: int) -> None:
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)


# ---------------- image encoder / decoder ----------------


class ConvEncoder(nn.Module):
    """Strided/maxpooled CNN over ``(B, T, H, W, C)`` images → ``(B, T, F)``."""

    def __init__(self, config: Any, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        act_cls = getattr(nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(int(config.depth) * int(m) for m in config.mults)
        self.kernel_size = int(config.kernel_size)

        layers: list[nn.Module] = []
        in_dim = input_ch
        for depth in self.depths:
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if bool(config.norm):
                layers.append(RMSNorm2D(depth, eps=1e-4, dtype=torch.float32))
            layers.append(act_cls())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (B, T, H, W, C) -> centred (B*T, C, H, W) -> conv -> flatten -> (B, T, F)
        obs = obs - 0.5
        b_t = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:]).permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)
        return x.reshape(*b_t, x.shape[-1])


class ConvDecoder(nn.Module):
    """Upsampling decoder back to ``(B, T, H, W, C)``.

    The deterministic state is reduced to a low-resolution spatial feature map
    via a block-linear projection; the stochastic state is reduced through a
    standard MLP. The two are summed and upsampled to the target resolution.
    """

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
        self.depths = tuple(int(config.depth) * int(m) for m in config.mults)
        factor = 2 ** len(self.depths)
        minres = [int(x // factor) for x in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)

        u = math.prod(self.min_shape)
        self.sp0 = BlockLinear(deter, u, self.bspace)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units),
            nn.RMSNorm(2 * self.units, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self.sp2 = nn.Linear(2 * self.units, u)
        self.sp_norm = nn.Sequential(
            nn.RMSNorm(self.depths[-1], eps=1e-4, dtype=torch.float32),
            act_cls(),
        )

        layers: list[nn.Module] = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True))
            layers.append(RMSNorm2D(depth, eps=1e-4, dtype=torch.float32))
            layers.append(act_cls())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        # (B, T, S, K), (B, T, D) -> (B, T, H, W, C)
        b_t = deter.shape[:-1]
        n = int(torch.tensor(b_t).prod().item()) if b_t else 1
        x0 = deter.reshape(n, deter.shape[-1])
        x1 = stoch.reshape(n, -1)

        H_feat, W_feat, C_feat = self.min_shape
        x0 = self.sp0(x0)
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        x1 = self.sp1(x1)
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        x = self.sp_norm(x0 + x1)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.sigmoid(x)
        return x.reshape(*b_t, *x.shape[1:])


# ---------------- multi-modality encoder / decoder ----------------


_EXCLUDED_OBS_KEYS = ("is_first", "is_last", "is_terminal", "reward")


class MultiEncoder(nn.Module):
    """Fuse CNN encoder over image obs + MLP encoder over flat-state obs."""

    def __init__(self, config: Any, shapes: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        shapes = {k: v for k, v in shapes.items() if k not in _EXCLUDED_OBS_KEYS and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}

        encoders: list[nn.Module] = []
        self._has_cnn = bool(self.cnn_shapes)
        self._has_mlp = bool(self.mlp_shapes)
        self.out_dim = 0

        if self._has_cnn:
            input_ch = sum(v[-1] for v in self.cnn_shapes.values())
            input_shape = (*next(iter(self.cnn_shapes.values()))[:2], input_ch)
            self.cnn = ConvEncoder(config.cnn, input_shape)
            encoders.append(self.cnn)
            self.out_dim += self.cnn.out_dim
        if self._has_mlp:
            inp_dim = sum(sum(v) for v in self.mlp_shapes.values())
            self.mlp = MLP(config.mlp, inp_dim)
            encoders.append(self.mlp)
            self.out_dim += self.mlp.out_dim

        if not encoders:
            raise ValueError("MultiEncoder requires at least one of cnn / mlp inputs.")
        self.encoders = nn.ModuleList(encoders)
        self.apply(weight_init_)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        if self._has_cnn:
            outs.append(self.cnn(torch.cat([obs[k] for k in self.cnn_shapes], dim=-1)))
        if self._has_mlp:
            outs.append(self.mlp(torch.cat([obs[k] for k in self.mlp_shapes], dim=-1)))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


class MultiDecoder(nn.Module):
    """Inverse of :class:`MultiEncoder` — produces per-key output distributions."""

    def __init__(self, config: Any, deter: int, flat_stoch: int, shapes: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        self._has_cnn = bool(self.cnn_shapes)
        self._has_mlp = bool(self.mlp_shapes)

        if self._has_cnn:
            some = next(iter(self.cnn_shapes.values()))
            out_shape = (sum(v[-1] for v in self.cnn_shapes.values()), *some[:-1])
            self.cnn = ConvDecoder(config.cnn, deter, flat_stoch, out_shape)
            self._image_dist: Callable[[torch.Tensor], Any] = partial(
                getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist
            )
        if self._has_mlp:
            shape = (sum(sum(v) for v in self.mlp_shapes.values()),)
            mlp_cfg = config.mlp
            mlp_cfg.shape = shape
            self.mlp = MLPHead(mlp_cfg, deter + flat_stoch)
            self._mlp_dist: Callable[[torch.Tensor], Any] = partial(
                getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist
            )

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self._has_cnn:
            split = [v[-1] for v in self.cnn_shapes.values()]
            chunks = torch.split(self.cnn(stoch, deter), split, dim=-1)
            for k, c in zip(self.cnn_shapes, chunks, strict=True):
                out[k] = self._image_dist(c)
        if self._has_mlp:
            split = [int(sum(v)) for v in self.mlp_shapes.values()]
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], dim=-1)
            chunks = torch.split(self.mlp(feat), split, dim=-1)
            for k, c in zip(self.mlp_shapes, chunks, strict=True):
                out[k] = self._mlp_dist(c)
        return out


# ---------------- generic MLP + head ----------------


class MLP(nn.Module):
    """Configurable feed-forward MLP with RMSNorm + activation between layers."""

    def __init__(self, config: Any, inp_dim: int) -> None:
        super().__init__()
        act_cls = getattr(nn, config.act)
        self._symlog_inputs = bool(getattr(config, "symlog_inputs", False))
        self.layers = nn.Sequential()
        for i in range(int(config.layers)):
            self.layers.add_module(f"{config.name}_linear{i}", nn.Linear(inp_dim, int(config.units)))
            self.layers.add_module(
                f"{config.name}_norm{i}",
                nn.RMSNorm(int(config.units), eps=1e-4, dtype=torch.float32),
            )
            self.layers.add_module(f"{config.name}_act{i}", act_cls())
            inp_dim = int(config.units)
        self.out_dim = int(config.units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._symlog_inputs:
            x = symlog(x)
        return self.layers(x)


class MLPHead(nn.Module):
    """MLP + final linear that returns a distribution selected by ``config.dist.name``."""

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
            kwargs = {
                "unimix_ratio": float(config.dist.unimix_ratio),
                "shape": tuple(config.shape),
            }
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


__all__ = [
    "MLP",
    "BlockLinear",
    "Conv2dSamePad",
    "ConvDecoder",
    "ConvEncoder",
    "MLPHead",
    "MultiDecoder",
    "MultiEncoder",
    "RMSNorm2D",
    "_StochReshape",
]
