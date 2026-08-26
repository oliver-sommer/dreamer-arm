"""Actor-critic: reward/continue heads, imagination-based policy/value training,
and replay-based value learning -- everything trained purely from
``world_model.get_feat(state)``, independent of which world model produced it.

λ-returns with return-EMA normalisation, a slow-moving value target, and the
replay-value bootstrap (imagined return when every replay position was rolled
out from, the value function itself when only a subsample was -- see
:meth:`ActorCritic.loss`) are all DreamerV3 / R2-Dreamer machinery shared
unchanged across world models.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from dreamer_arm.core import networks
from dreamer_arm.core.frozen import freeze_clone
from dreamer_arm.core.losses import lambda_return
from dreamer_arm.core.world_model.protocol import WorldModel
from dreamer_arm.utils.tensor import tensorstats, to_f32


def sanitize_action(action: torch.Tensor, discrete: bool) -> torch.Tensor:
    """Scrub + clamp a continuous action to the space the env executes.

    Continuous policy distributions are expected to be bounded already, but
    this remains the final contract guard: anything that conditions on an
    action — the replay buffer, the world model's prev_action, and the
    imagination rollout the actor/critic train on — must see exactly the
    [-1, 1] value the environment executes.

    nan_to_num first: MPS occasionally produces a non-finite sample, and a
    single NaN action NaNs every downstream latent and loss (this was the
    source of the intermittent policy/value NaNs).  clamp alone leaves NaN
    as NaN, so scrub before clamping.
    """
    if discrete:
        return action
    return torch.nan_to_num(action).clamp(-1.0, 1.0)


class ReturnEMA(nn.Module):
    """Track task-local return percentiles for actor-critic normalisation.

    A single-task agent has one row and is exactly the usual DreamerV3 return
    EMA.  Multi-task agents keep one row per one-hot task.  Sharing these
    statistics lets a high-return task set the policy-gradient scale for every
    other task, even when replay itself is balanced.
    """

    def __init__(self, device: torch.device, num_tasks: int = 1, alpha: float = 1e-2) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.alpha = alpha
        self.num_tasks = num_tasks
        self.register_buffer("range", torch.tensor([0.05, 0.95], device=device))
        self.register_buffer("ema_vals", torch.zeros(num_tasks, 2, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, task_id: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Update percentiles and return row-aligned ``(offset, scale)``.

        ``x`` is ``(N, horizon, 1)`` and ``task_id`` is the one-hot identity
        of each imagination start, ``(N, num_tasks)``.  The returned tensors
        are ``(N, 1, 1)`` for multi-task use so broadcasting cannot
        accidentally mix tasks.
        """
        if self.num_tasks == 1:
            x_quantile = torch.quantile(x.detach().flatten(), self.range)
            current = self.ema_vals[0]
            updated = self.alpha * x_quantile.detach() + (1 - self.alpha) * current
            self.ema_vals[0].copy_(torch.where(torch.isfinite(x_quantile), updated, current))
            scale = torch.clip(self.ema_vals[0, 1] - self.ema_vals[0, 0], min=1.0)
            return self.ema_vals[0, 0].detach(), scale.detach()

        if task_id is None:
            raise ValueError("multi-task return normalisation requires task_id")
        if task_id.shape != (x.shape[0], self.num_tasks):
            raise ValueError(f"task_id must have shape {(x.shape[0], self.num_tasks)}, got {tuple(task_id.shape)}")
        task_index = task_id.detach().argmax(dim=-1)
        for index in range(self.num_tasks):
            values = x.detach()[task_index == index]
            if values.numel() == 0:
                continue
            quantile = torch.quantile(values.flatten(), self.range)
            current = self.ema_vals[index]
            updated = self.alpha * quantile.detach() + (1 - self.alpha) * current
            current.copy_(torch.where(torch.isfinite(quantile), updated, current))

        selected = self.ema_vals[task_index]
        offset = selected[:, 0, None, None]
        scale = torch.clip(selected[:, 1] - selected[:, 0], min=1.0)[:, None, None]
        return offset.detach(), scale.detach()


class ActorCritic(nn.Module):
    """Reward/continue heads + actor + critic + λ-return imagination training."""

    _frozen_reward: networks.MLPHead
    _frozen_cont: networks.MLPHead
    _frozen_actor: networks.MLPHead
    _frozen_slow_value: networks.MLPHead

    def __init__(
        self,
        config: Any,
        feat_size: int,
        actor_shape: tuple[int, ...],
        act_discrete: bool,
        imag_starts: int | None,
        device: torch.device,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.act_discrete = act_discrete
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.act_entropy = float(config.act_entropy)
        self.imag_starts = imag_starts
        self.device = device

        self.reward = networks.MLPHead(config.reward, feat_size)
        self.cont = networks.MLPHead(config.cont, feat_size)

        # Build a local actor-head config (don't mutate the shared cfg object):
        # ``shape`` comes from the env action space, and ``dist`` collapses the
        # disc/cont branch into the single one the agent will actually use.
        actor_cfg = copy.deepcopy(config.actor)
        # deepcopy preserves the read-only flag that dispatch() sets on the run
        # config, so clear it -- this copy exists precisely to be written to.
        OmegaConf.set_readonly(actor_cfg, False)
        actor_cfg.shape = actor_shape
        actor_cfg.dist = config.actor.dist.disc if act_discrete else config.actor.dist.cont
        self.actor = networks.MLPHead(actor_cfg, feat_size)
        self.value = networks.MLPHead(config.critic, feat_size)
        self.return_ema = ReturnEMA(device=device, num_tasks=num_tasks)

        # slow target value (EMA) — no grad.
        self._slow_value = copy.deepcopy(self.value)
        for p in self._slow_value.parameters():
            p.requires_grad_(False)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self.slow_value_updates = 0

        self.refresh_frozen()

    def train(self, mode: bool = True) -> ActorCritic:
        super().train(mode)
        # Slow value target is always in eval mode.
        self._slow_value.train(False)
        return self

    def refresh_frozen(self) -> None:
        """Rebuild non-persistent rollout views after storage-moving operations.

        These are derived views over live parameter storage, not independent
        model state. Bypass ``nn.Module.__setattr__`` so they do not duplicate
        keys in ``state_dict`` or appear in parameter/module traversal.
        """

        def frozen[ModuleT: nn.Module](module: ModuleT) -> ModuleT:
            view = freeze_clone(module)
            view.train(False)
            return view

        object.__setattr__(self, "_frozen_reward", frozen(self.reward))
        object.__setattr__(self, "_frozen_cont", frozen(self.cont))
        object.__setattr__(self, "_frozen_actor", frozen(self.actor))
        object.__setattr__(self, "_frozen_slow_value", frozen(self._slow_value))

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Params the optimiser should train -- excludes the slow value target."""
        return [
            (f"{name}.{pname}", p)
            for name in ("reward", "cont", "actor", "value")
            for pname, p in getattr(self, name).named_parameters()
        ]

    def update_slow_target(self) -> None:
        if self.slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters(), strict=True):
                    s.data.copy_(mix * v.data + (1.0 - mix) * s.data)
        self.slow_value_updates += 1

    def act(self, feat: torch.Tensor, eval_mode: bool) -> torch.Tensor:
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval_mode else action_dist.rsample()
        return sanitize_action(action, self.act_discrete)

    @torch.no_grad()
    def policy_diagnostics(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose actor values on the exact frozen rollout path used by ``act``."""
        dist = self._frozen_actor(feat)
        diagnostics = {"post_mode": sanitize_action(dist.mode, self.act_discrete)}
        if not self.act_discrete and hasattr(dist, "pre_mean") and hasattr(dist, "pre_std"):
            diagnostics["pre_mean"] = dist.pre_mean
            diagnostics["pre_std"] = dist.pre_std
        return diagnostics

    def _gather_imag_starts(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Flatten a ``(B, T, ...)`` posterior trajectory into imagination starts.

        ``None`` (the RSSM default) keeps every ``B*T`` posterior, matching the
        original unconditional behaviour exactly -- including RNG consumption,
        since no ``randperm`` is drawn. Token-space world models (DINO-WM,
        Dreamer 4) make imagination far more expensive per start and must
        subsample.

        Subsampling indexes the ``(b, t)`` pair directly rather than reshaping
        first: a context-window world model's state is a strided view over its
        token buffer, and reshaping that would copy every window when only
        ``imag_starts`` of them are wanted.
        """
        b, t = next(iter(state.values())).shape[:2]
        if self.imag_starts is None:
            return {key: v.reshape(-1, *v.shape[2:]).detach() for key, v in state.items()}
        k = min(self.imag_starts, b * t)
        flat = torch.randperm(b * t, device=self.device)[:k]
        rows, cols = flat // t, flat % t
        return {key: v[rows, cols].detach() for key, v in state.items()}

    @torch.no_grad()
    def _imagine(
        self, frozen_wm: WorldModel, start: dict[str, torch.Tensor], imag_horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy in latent space for ``imag_horizon`` steps."""
        feats: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        state = start
        for _ in range(imag_horizon):
            feat = frozen_wm.get_feat(state)
            action = sanitize_action(self._frozen_actor(feat).rsample(), self.act_discrete)
            feats.append(feat)
            actions.append(action)
            state = frozen_wm.img_step(state, action)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    def loss(
        self,
        feat: torch.Tensor,
        data: dict[str, torch.Tensor],
        frozen_wm: WorldModel,
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Reward/continue + imagination-based policy/value + replay-value losses.

        ``feat``/``data`` are already aligned to the world model's valid
        window (see :meth:`~dreamer_arm.core.model.Dreamer._cal_grad`);
        ``state`` is the world model's observed trajectory, used as
        imagination start states.
        """
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        b, t = data["action"].shape[:2]

        losses["rew"] = -self.reward(feat).log_prob(to_f32(data["reward"])).mean()
        cont_target = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        losses["con"] = -self.cont(feat).log_prob(cont_target).mean()

        # Gather task identity with exactly the same random (b, t) indices as
        # the imagination state.  RSSM absorbs task_id into its latent and does
        # not retain the raw one-hot, whereas DINO-WM does; sourcing it from the
        # aligned training data works for both without changing WM state.
        start_source = dict(state)
        if self.return_ema.num_tasks > 1:
            if "task_id" not in data:
                raise ValueError("multi-task actor-critic batch omitted task_id")
            start_source["_return_task_id"] = data["task_id"]
        start = self._gather_imag_starts(start_source)
        return_task_id = start.pop("_return_task_id", None)
        imag_feat, imag_action = self._imagine(frozen_wm, start, self.imag_horizon + 1)
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        # self.value, not the frozen clone: imag_feat is already detached
        # above, so this carries no gradient regardless -- and the value loss
        # further down needs this exact distribution object anyway. Computing
        # it once here removes a redundant forward over the same 1024x(H+1)
        # features.
        imag_value_dist = self.value(imag_feat)
        imag_value = imag_value_dist.mode().detach()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()

        disc = 1.0 - 1.0 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1.0 - imag_cont
        # DreamerV3 uses the live critic for λ-return bootstrapping and keeps
        # the slow critic as a value-loss regulariser.  This also keeps the
        # actor baseline and target on the same critic snapshot.
        ret = lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret, return_task_id)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = (weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)).mean()

        tar_padded = torch.cat([ret, torch.zeros_like(ret[:, -1:])], dim=1)
        losses["value"] = (
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        ).mean()

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = ret_normed.mean()
        metrics["ret_005"] = self.return_ema.ema_vals[:, 0].mean()
        metrics["ret_095"] = self.return_ema.ema_vals[:, 1].mean()
        if self.return_ema.num_tasks > 1:
            for index in range(self.return_ema.num_tasks):
                metrics[f"ret_005_task_{index}"] = self.return_ema.ema_vals[index, 0]
                metrics[f"ret_095_task_{index}"] = self.return_ema.ema_vals[index, 1]
        metrics["adv"] = adv.mean()
        metrics["adv_std"] = adv.std()
        metrics["con"] = imag_cont.mean()
        metrics["rew"] = imag_reward.mean()
        metrics["val"] = imag_value.mean()
        metrics["tar"] = ret.mean()
        metrics["slowval"] = imag_slow_value.mean()
        metrics["weight"] = weight.mean()
        metrics["action_entropy"] = entropy.mean()
        metrics.update(tensorstats(imag_action, "action"))
        if not self.act_discrete:
            # The aggregate action_mean mixes XYZ motion with the gripper and
            # can make a gripper pinned at -1 look like a directional arm
            # bias.  Keep the aggregate for continuity, but expose the four
            # semantic control axes separately, including how often each one
            # is at the environment clamp.
            action_labels = ("x", "y", "z", "gripper")
            for index in range(imag_action.shape[-1]):
                label = action_labels[index] if index < len(action_labels) else str(index)
                component = imag_action[..., index]
                metrics[f"action_{label}_mean"] = component.mean()
                metrics[f"action_{label}_std"] = component.std()
                metrics[f"action_{label}_frac_saturated"] = (component.abs() >= 1.0 - 1e-6).float().mean()
                if hasattr(policy, "pre_mean") and hasattr(policy, "pre_std"):
                    metrics[f"action_{label}_pre_mean"] = policy.pre_mean[..., index].mean()
                    metrics[f"action_{label}_pre_std"] = policy.pre_std[..., index].mean()

        rep_last = to_f32(data["is_last"]).unsqueeze(-1)
        rep_term = to_f32(data["is_terminal"]).unsqueeze(-1)
        rep_reward = to_f32(data["reward"])
        # self.value, not the frozen clone -- see the imag_value comment above;
        # the value loss below needs value_dist regardless, so compute it once.
        value_dist = self.value(feat)
        rep_value = value_dist.mode().detach()
        rep_slow = self._frozen_slow_value(feat).mode()
        if self.imag_starts is None:  # noqa: SIM108 (branch comments matter more than brevity here)
            # ret[:, 0] is, for every flattened (b, t) posterior, the return of
            # the imagined trajectory starting there -- a strictly better
            # bootstrap than the value function alone, but only meaningful
            # when every posterior was actually imagined from.
            boot = ret[:, 0].reshape(b, t, 1)
        else:
            # Imagination only rolled out a subsample of starts, so ret[:, 0]
            # no longer aligns 1:1 with the replay (b, t) grid -- fall back to
            # bootstrapping with the value function itself.
            boot = rep_value.detach()
        rep_ret = lambda_return(rep_last, rep_term, rep_reward, rep_value, boot, disc, self.lamb)
        rep_padded = torch.cat([rep_ret, torch.zeros_like(rep_ret[:, -1:])], dim=1)
        rep_weight = 1.0 - rep_last
        losses["repval"] = (
            rep_weight[:, :-1]
            * (-value_dist.log_prob(rep_padded.detach()) - value_dist.log_prob(rep_slow.detach()))[:, :-1].unsqueeze(-1)
        ).mean()
        metrics.update(tensorstats(rep_ret, "ret_replay"))
        metrics.update(tensorstats(rep_value, "value_replay"))
        metrics.update(tensorstats(rep_slow, "slow_value_replay"))

        return losses, metrics


__all__ = ["ActorCritic", "ReturnEMA", "sanitize_action"]
