"""Recurrent state-space model (RSSM) with categorical stochastic state.

Mirrors the DreamerV3 / R2-Dreamer RSSM:

- ``Deter``: block-GRU-style deterministic transition. Each input stream
  (deterministic state, stochastic state, action) is first projected to a
  shared hidden dim by a small RMSNorm-MLP, then broadcast across ``blocks``
  block-linear groups together with a per-block slice of the deterministic
  state. ``dyn_layers`` block-linear hidden layers are followed by a
  block-linear GRU gate (reset / candidate / update).
- ``RSSM``: posterior (``obs_step``) and prior (``img_step`` / ``prior``)
  conditioned on the deterministic state, with categorical stochastic state
  sampled via straight-through Gumbel-softmax (see
  :class:`dreamer_arm.core.distributions.OneHotDist`).
- ``Projector``: bias-free linear used by the R2-Dreamer Barlow-Twins head.
- ``RSSMWorldModel``: adapts the three above (+ optionally a
  :class:`~dreamer_arm.core.networks.MultiDecoder` for the DreamerV3
  reconstruction variant) to the
  :class:`~dreamer_arm.core.world_model.protocol.WorldModel` protocol.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import distributions as torchd
from torch import nn

from dreamer_arm.core import distributions as dists
from dreamer_arm.core.losses import barlow_twins_loss
from dreamer_arm.core.networks import BlockLinear, MultiDecoder, MultiEncoder, _StochReshape
from dreamer_arm.core.networks.layers import weight_init_
from dreamer_arm.utils.tensor import rpad


class Deter(nn.Module):
    """Block-GRU deterministic transition: ``(stoch, deter, action) → deter'``."""

    def __init__(
        self,
        deter: int,
        stoch: int,
        act_dim: int,
        hidden: int,
        blocks: int,
        dynlayers: int,
        act: str = "SiLU",
    ) -> None:
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        self._deter = int(deter)
        act_cls = getattr(nn, act)

        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )

        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-4, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act_cls())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)

    def _flat_to_group(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _group_to_flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``(B, S, K), (B, D), (B, A) → (B, D)``."""
        b = action.shape[0]
        stoch = stoch.reshape(b, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()

        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self._group_to_flat(torch.cat([self._flat_to_group(deter), x], dim=-1))

        x = self._dyn_hid(x)
        x = self._dyn_gru(x)

        gates = torch.chunk(self._flat_to_group(x), 3, dim=-1)
        reset, cand, update = (self._group_to_flat(g) for g in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    """Categorical RSSM with posterior ``observe`` and prior ``imagine_with_action``."""

    def __init__(self, config: Any, embed_size: int, act_dim: int) -> None:
        super().__init__()
        self._stoch = int(config.stoch)
        self._discrete = int(config.discrete)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        act_cls = getattr(nn, config.act)

        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=int(config.dyn_layers),
            act=str(config.act),
        )

        self._obs_net = self._build_stoch_head(
            input_dim=self._deter + embed_size,
            num_layers=self._obs_layers,
            prefix="obs",
            act_cls=act_cls,
        )
        self._img_net = self._build_stoch_head(
            input_dim=self._deter, num_layers=self._img_layers, prefix="img", act_cls=act_cls
        )
        self.apply(weight_init_)

    def _build_stoch_head(
        self, input_dim: int, num_layers: int, prefix: str, act_cls: type[nn.Module]
    ) -> nn.Sequential:
        seq = nn.Sequential()
        inp = input_dim
        for i in range(num_layers):
            seq.add_module(f"{prefix}_net_{i}", nn.Linear(inp, self._hidden))
            seq.add_module(f"{prefix}_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-4, dtype=torch.float32))
            seq.add_module(f"{prefix}_net_a_{i}", act_cls())
            inp = self._hidden
        seq.add_module(f"{prefix}_net_logit", nn.Linear(inp, self._stoch * self._discrete))
        seq.add_module(f"{prefix}_net_reshape", _StochReshape(self._stoch, self._discrete))
        return seq

    def initial(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        initial: tuple[torch.Tensor, torch.Tensor],
        reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior rollout over ``T`` steps.

        Shapes: ``embed (B, T, E)``, ``action (B, T, A)``, ``reset (B, T)`` →
        ``stochs (B, T, S, K), deters (B, T, D), logits (B, T, S, K)``.
        """
        t = action.shape[1]
        stoch, deter = initial
        stochs: list[torch.Tensor] = []
        deters: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for i in range(t):
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        return (
            torch.stack(stochs, dim=1),
            torch.stack(deters, dim=1),
            torch.stack(logits, dim=1),
        )

    def imagine_with_action(
        self, stoch: torch.Tensor, deter: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prior rollout given a sequence of actions ``(B, T, A)``."""
        t = actions.shape[1]
        stochs: list[torch.Tensor] = []
        deters: list[torch.Tensor] = []
        for i in range(t):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)

    def obs_step(
        self,
        stoch: torch.Tensor,
        deter: torch.Tensor,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior step: deterministic transition + observation-conditioned logits."""
        stoch = torch.where(rpad(reset, stoch.dim() - reset.dim()), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - reset.dim()), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - reset.dim()),
            torch.zeros_like(prev_action),
            prev_action,
        )
        deter = self._deter_net(stoch, deter, prev_action)
        logit = self._obs_net(torch.cat([deter, embed], dim=-1))
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(
        self, stoch: torch.Tensor, deter: torch.Tensor, prev_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prior step: deterministic transition then sample from imagination logits."""
        deter = self._deter_net(stoch, deter, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the prior stochastic state from a deterministic state."""
        logit = self._img_net(deter)
        return self.get_dist(logit).rsample(), logit

    def get_feat(self, stoch: torch.Tensor, deter: torch.Tensor) -> torch.Tensor:
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, deter], dim=-1)

    def get_dist(self, logit: torch.Tensor) -> torchd.Independent:  # type: ignore[type-arg]
        return torchd.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(
        self, post_logit: torch.Tensor, prior_logit: torch.Tensor, free: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced KL: ``(dyn_loss, rep_loss)`` with per-side stop-grad and free nats."""
        rep_loss = dists.kl(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = dists.kl(post_logit.detach(), prior_logit).sum(-1)
        return torch.clip(dyn_loss, min=free), torch.clip(rep_loss, min=free)


class Projector(nn.Module):
    """Bias-free linear projection used by the R2-Dreamer Barlow-Twins head."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.w = nn.Linear(in_ch, out_ch, bias=False)
        self.apply(weight_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class RSSMWorldModel:
    """Adapts :class:`RSSM` + :class:`~dreamer_arm.core.networks.MultiEncoder`
    (plus, for the representation-loss branch, a
    :class:`~dreamer_arm.core.networks.MultiDecoder` or :class:`Projector`) to
    the :class:`~dreamer_arm.core.world_model.protocol.WorldModel` protocol.

    A plain object, not an ``nn.Module``: it holds no parameters of its own,
    only references to the modules passed in. Construct one wrapping the live
    (trainable) modules for :meth:`loss`, and a second wrapping
    :func:`~dreamer_arm.core.frozen.freeze_clone` copies for rollout inference
    and imagination -- the representation-loss keyword arguments are only used
    by :meth:`loss`, so the frozen instance can omit them.

    State keys: ``stoch (B, [T,] S, K)``, ``deter (B, [T,] D)``.
    """

    replay_cache_keys = ("stoch", "deter")

    def __init__(
        self,
        rssm: RSSM,
        encoder: MultiEncoder,
        *,
        kl_free: float = 0.0,
        rep_loss: str = "r2dreamer",
        decoder: MultiDecoder | None = None,
        projector: Projector | None = None,
        barlow_lambd: float = 0.0,
    ) -> None:
        self.rssm = rssm
        self.encoder = encoder
        self.kl_free = kl_free
        self.rep_loss = rep_loss
        self.decoder = decoder
        self.projector = projector
        self.barlow_lambd = barlow_lambd

    @property
    def feat_size(self) -> int:
        return self.rssm.feat_size

    def initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        stoch, deter = self.rssm.initial(batch_size)
        return {"stoch": stoch, "deter": deter}

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        stoch, deter = self.rssm.img_step(state["stoch"], state["deter"], action)
        return {"stoch": stoch, "deter": deter}

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rssm.get_feat(state["stoch"], state["deter"])

    def encode_for_act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(obs)

    def observe_step(
        self,
        prev_state: dict[str, torch.Tensor],
        encoded: torch.Tensor,
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        stoch, deter, _ = self.rssm.obs_step(prev_state["stoch"], prev_state["deter"], prev_action, encoded, is_first)
        return {"stoch": stoch, "deter": deter}

    def loss(
        self, data: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        b, t = data["action"].shape[:2]

        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], (initial["stoch"], initial["deter"]), data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses: dict[str, torch.Tensor] = {"dyn": dyn_loss.mean(), "rep": rep_loss.mean()}
        metrics: dict[str, torch.Tensor] = {
            "dyn_entropy": self.rssm.get_dist(prior_logit).entropy().mean(),
            "rep_entropy": self.rssm.get_dist(post_logit).entropy().mean(),
        }

        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamerv3":
            assert self.decoder is not None
            for key, dist in self.decoder(post_stoch, post_deter).items():
                losses[key] = -dist.log_prob(data[key]).mean()
        else:  # r2dreamer
            assert self.projector is not None
            x1 = self.projector(feat.reshape(b * t, -1))
            x2 = embed.reshape(b * t, -1).detach()
            total, invariance, redundancy = barlow_twins_loss(x1, x2, self.barlow_lambd)
            losses["barlow"] = total
            metrics["barlow/invariance"] = invariance.detach()
            metrics["barlow/redundancy"] = redundancy.detach()

        return {"stoch": post_stoch, "deter": post_deter}, losses, metrics
