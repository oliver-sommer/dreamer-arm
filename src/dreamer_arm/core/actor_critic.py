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


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean over Dreamer's discounted imagination weights."""

    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def _weighted_stats(value: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = _weighted_mean(value, weight)
    variance = _weighted_mean((value - mean).square(), weight)
    return mean, variance.sqrt()


def _explained_variance(target: torch.Tensor, prediction: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted explained variance, defined as zero for constant targets."""

    target_mean = _weighted_mean(target, weight)
    target_variance = _weighted_mean((target - target_mean).square(), weight)
    residual = target - prediction
    residual_mean = _weighted_mean(residual, weight)
    residual_variance = _weighted_mean((residual - residual_mean).square(), weight)
    score = 1.0 - residual_variance / target_variance.clamp_min(1e-8)
    return torch.where(target_variance > 1e-8, score, torch.zeros_like(score))


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
    _frozen_success: networks.MLPHead | None

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
        self.constraint_cost_scale = float(config.get("constraint_cost_scale", 0.0))
        self.imag_starts = imag_starts
        self.device = device

        self.reward = networks.MLPHead(config.reward, feat_size)
        self.cont = networks.MLPHead(config.cont, feat_size)
        success_cfg = config.get("success", None)
        self.success_bonus = float(success_cfg.get("bonus", 0.0)) if success_cfg is not None else 0.0
        self.success: networks.MLPHead | None = None
        if success_cfg is not None and bool(success_cfg.get("enabled", False)):
            self.success = networks.MLPHead(success_cfg, feat_size)

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
        object.__setattr__(self, "_frozen_success", frozen(self.success) if self.success is not None else None)

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Params the optimiser should train -- excludes the slow value target."""
        names = ["reward", "cont", "actor", "value"]
        if self.success is not None:
            names.append("success")
        return [(f"{name}.{pname}", p) for name in names for pname, p in getattr(self, name).named_parameters()]

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
        if self._frozen_success is not None:
            diagnostics["success_probability"] = self._frozen_success(feat).mean.squeeze(-1)
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
        task_id = state.get("_return_task_id")
        if task_id is None or task_id.shape[-1] <= 1:
            flat = torch.randperm(b * t, device=self.device)[:k]
        else:
            flat_task = task_id.reshape(-1, task_id.shape[-1]).argmax(dim=-1)
            present_tasks = torch.unique(flat_task, sorted=True)
            task_count = int(present_tasks.numel())
            per_task, remainder = divmod(k, task_count)
            selected: list[torch.Tensor] = []
            for rank, task in enumerate(present_tasks.unbind()):
                candidates = torch.nonzero(flat_task == task, as_tuple=False).squeeze(-1)
                count = per_task + int(rank < remainder)
                if candidates.numel() < count:
                    raise RuntimeError(
                        f"task-balanced imagination needs {count} states for task {int(task)}, "
                        f"but replay supplied {candidates.numel()}"
                    )
                order = torch.randperm(candidates.numel(), device=self.device)[:count]
                selected.append(candidates[order])
            flat = torch.cat(selected)
            flat = flat[torch.randperm(flat.numel(), device=self.device)]
        rows, cols = flat // t, flat % t
        return {key: v[rows, cols].detach() for key, v in state.items()}

    @torch.no_grad()
    def _imagine(
        self, frozen_wm: WorldModel, start: dict[str, torch.Tensor], imag_horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy in latent space for ``imag_horizon`` steps."""
        feats: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        constraint_probs: list[torch.Tensor] = []
        constraint_axes: list[torch.Tensor] = []
        state = start
        for _ in range(imag_horizon):
            feat = frozen_wm.get_feat(state)
            action = sanitize_action(self._frozen_actor(feat).rsample(), self.act_discrete)
            predict_constraints = getattr(frozen_wm, "predict_constraints", None)
            constraint = predict_constraints(state, action) if predict_constraints is not None else None
            if constraint is None:
                axis_probability = torch.zeros(*action.shape[:-1], 3, device=action.device)
            else:
                axis_probability = constraint["clamp_logits"].sigmoid()
            feats.append(feat)
            actions.append(action)
            constraint_axes.append(axis_probability)
            constraint_probs.append(axis_probability.max(dim=-1, keepdim=True).values)
            state = frozen_wm.img_step(state, action)
        return (
            torch.stack(feats, dim=1),
            torch.stack(actions, dim=1),
            torch.stack(constraint_probs, dim=1),
            torch.stack(constraint_axes, dim=1),
        )

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

        reward_target = to_f32(data["reward"])
        reward_dist = self.reward(feat)
        losses["rew"] = -reward_dist.log_prob(reward_target).mean()
        metrics["reward/mae"] = (reward_dist.mode() - reward_target).abs().mean().detach()
        cont_target = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        cont_dist = self.cont(feat)
        losses["con"] = -cont_dist.log_prob(cont_target).mean()
        metrics["continue/brier"] = (cont_dist.mean - cont_target).square().mean().detach()
        if self.success is not None and "success" in data:
            success_target = to_f32(data["success"])
            success_dist = self.success(feat)
            success_nll = -success_dist.log_prob(success_target)
            flat_target = success_target.squeeze(-1)
            total = torch.as_tensor(float(flat_target.numel()), device=flat_target.device)
            positives = flat_target.sum()
            negatives = total - positives
            positive_weight = (total / (2.0 * positives.clamp_min(1.0))).clamp(max=20.0)
            negative_weight = total / (2.0 * negatives.clamp_min(1.0))
            success_weight = torch.where(flat_target > 0.5, positive_weight, negative_weight)
            losses["success"] = (success_nll * success_weight).sum() / success_weight.sum().clamp_min(1.0)
            success_probability = success_dist.mean
            metrics["success/target_rate"] = success_target.mean().detach()
            metrics["success/predicted_rate"] = success_probability.mean().detach()
            metrics["success/brier"] = (success_probability - success_target).square().mean().detach()
            positive_mask = success_target > 0.5
            negative_mask = ~positive_mask
            metrics["success/predicted_positive"] = (
                success_probability[positive_mask].mean().detach()
                if positive_mask.any()
                else torch.zeros((), device=feat.device)
            )
            metrics["success/predicted_negative"] = (
                success_probability[negative_mask].mean().detach()
                if negative_mask.any()
                else torch.zeros((), device=feat.device)
            )

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
        imag_feat, imag_action, imag_constraint, imag_constraint_axes = self._imagine(
            frozen_wm, start, self.imag_horizon + 1
        )
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()
        imag_constraint = imag_constraint.detach()
        imag_constraint_axes = imag_constraint_axes.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_success = (
            self._frozen_success(imag_feat).mean if self._frozen_success is not None else torch.zeros_like(imag_reward)
        )
        shaped_reward = imag_reward + self.success_bonus * imag_success - self.constraint_cost_scale * imag_constraint
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
        ret = lambda_return(last, term, shaped_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret, return_task_id)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        actor_weight = weight[:, :-1].detach()
        losses["policy"] = (actor_weight * -(logpi * adv.detach() + self.act_entropy * entropy)).mean()

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
        advantage_mean, advantage_std = _weighted_stats(adv, actor_weight)
        metrics["adv"] = advantage_mean
        metrics["adv_std"] = advantage_std
        metrics["actor/advantage_mean"] = advantage_mean
        metrics["actor/advantage_std"] = advantage_std
        metrics["actor/advantage_abs_mean"] = _weighted_mean(adv.abs(), actor_weight)
        metrics["actor/advantage_positive_fraction"] = _weighted_mean((adv > 0).float(), actor_weight)
        metrics["actor/log_probability_mean"] = _weighted_mean(logpi, actor_weight)
        metrics["con"] = imag_cont.mean()
        metrics["rew"] = imag_reward.mean()
        metrics["shaped_rew"] = shaped_reward.mean()
        metrics["imag_success"] = imag_success.mean()
        metrics["imag_success_bonus"] = (self.success_bonus * imag_success).mean()
        metrics["imag_constraint_prob"] = imag_constraint.mean()
        metrics["imag_constraint_cost"] = (self.constraint_cost_scale * imag_constraint).mean()
        for index, label in enumerate(("workspace", "lag", "joint_limit")):
            metrics[f"imag_constraint_{label}"] = imag_constraint_axes[..., index].mean()
        metrics["val"] = imag_value.mean()
        metrics["tar"] = ret.mean()
        metrics["slowval"] = imag_slow_value.mean()
        metrics["weight"] = weight.mean()
        metrics["action_entropy"] = entropy.mean()
        if hasattr(policy, "pre_std"):
            metrics["actor/policy_std_mean"] = policy.pre_std.mean()
        metrics["ret_scale_mean"] = ret_scale.mean()
        metrics["ret_scale_min"] = ret_scale.min()
        metrics["ret_scale_max"] = ret_scale.max()
        if not self.act_discrete:
            xyz = imag_action[..., : min(3, imag_action.shape[-1])]
            metrics["action_xyz_near_bound_fraction"] = (xyz.abs() >= 0.95).float().mean()
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

        imag_prediction = imag_value[:, :-1]
        imag_slow = imag_slow_value[:, :-1]
        imag_error = imag_prediction - ret
        critic_value_mean, _ = _weighted_stats(imag_prediction, actor_weight)
        critic_target_mean, _ = _weighted_stats(ret, actor_weight)
        metrics["critic/value_mean"] = critic_value_mean
        metrics["critic/target_mean"] = critic_target_mean
        metrics["critic/value_bias"] = _weighted_mean(imag_error, actor_weight)
        metrics["critic/value_mae"] = _weighted_mean(imag_error.abs(), actor_weight)
        metrics["critic/value_rmse"] = _weighted_mean(imag_error.square(), actor_weight).sqrt()
        metrics["critic/explained_variance"] = _explained_variance(ret, imag_prediction, actor_weight)
        metrics["critic/slow_value_mean"] = _weighted_mean(imag_slow, actor_weight)
        metrics["critic/slow_value_gap_mae"] = _weighted_mean((imag_prediction - imag_slow).abs(), actor_weight)

        replay_prediction = rep_value[:, :-1]
        replay_error = replay_prediction - rep_ret
        replay_metric_weight = rep_weight[:, :-1]
        metrics["critic/replay_value_mean"] = _weighted_mean(replay_prediction, replay_metric_weight)
        metrics["critic/replay_target_mean"] = _weighted_mean(rep_ret, replay_metric_weight)
        metrics["critic/replay_value_mae"] = _weighted_mean(replay_error.abs(), replay_metric_weight)
        metrics["critic/replay_explained_variance"] = _explained_variance(
            rep_ret, replay_prediction, replay_metric_weight
        )

        # The trainer turns these bounded samples into occasional histograms;
        # they never enter the scalar contract.
        metrics["diagnostic/advantage"] = adv.detach().reshape(-1)[:4096]
        metrics["diagnostic/action"] = imag_action.detach().reshape(-1)[:4096]
        metrics["diagnostic/value_error"] = imag_error.detach().reshape(-1)[:4096]
        metrics["diagnostic/replay_value_error"] = replay_error.detach().reshape(-1)[:4096]

        return losses, metrics


__all__ = ["ActorCritic", "ReturnEMA", "sanitize_action"]
