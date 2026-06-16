"""Dreamer agent (R2-Dreamer / DreamerV3) — world model + actor-critic + optimiser.

The world-model representation loss is selected by ``config.rep_loss``:

- ``"r2dreamer"`` (default): decoder-free. A linear projector maps the RSSM
  latent feature to the encoder embedding space and the two are pushed
  together via :func:`dreamer_arm.agent.losses.barlow_twins_loss` (eq. 5 of
  the R2-Dreamer paper).
- ``"dreamerv3"``: decoder-based reconstruction. A
  :class:`~dreamer_arm.architecture.networks.MultiDecoder` reconstructs each
  observation key from the latent, with per-key NLL contributing the recon
  loss term.

Everything else (RSSM, KL balancing, reward / continue / actor / value
heads, λ-returns with return-EMA normalisation, slow-moving value target,
LaProp + adaptive gradient clipping + GradScaler) is shared.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR

from dreamer_arm.agent.losses import barlow_twins_loss, lambda_return
from dreamer_arm.architecture import MultiDecoder, MultiEncoder, Projector, ReturnEMA, networks
from dreamer_arm.architecture.rssm import RSSM
from dreamer_arm.optim import LaProp, adaptive_grad_clip
from dreamer_arm.utils.tensor import (
    compute_global_norm,
    compute_rms,
    tensorstats,
    to_f32,
)


class Dreamer(nn.Module):
    """End-to-end Dreamer agent.

    Construct with a Hydra config, gymnasium ``obs_space`` (a ``Dict`` of
    arrays) and ``act_space``. After construction call :meth:`act` for
    rollout and :meth:`update` for a single training step.
    """

    def __init__(self, config: Any, obs_space: Any, act_space: Any) -> None:
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.rep_loss = str(config.rep_loss)
        if self.rep_loss not in ("r2dreamer", "dreamerv3"):
            raise ValueError(
                f"Unsupported rep_loss={self.rep_loss!r}. "
                "This implementation supports 'r2dreamer' or 'dreamerv3'."
            )

        # --- shapes / action space ---
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.act_dim, self.act_discrete = _resolve_action_space(act_space)

        # --- world model ---
        self.encoder = MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = RSSM(config.rssm, self.embed_size, self.act_dim)
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        # --- actor / critic ---
        # Build a local actor-head config (don't mutate the shared cfg object):
        # ``shape`` comes from the env action space, and ``dist`` collapses the
        # disc/cont branch into the single one the agent will actually use.
        actor_cfg = copy.deepcopy(config.actor)
        actor_cfg.shape = (
            (act_space.n,) if hasattr(act_space, "n") else tuple(int(x) for x in act_space.shape)
        )
        actor_cfg.dist = config.actor.dist.disc if self.act_discrete else config.actor.dist.cont
        self.actor = networks.MLPHead(actor_cfg, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.return_ema = ReturnEMA(device=self.device)

        # slow target value (EMA) — no grad.
        self._slow_value = copy.deepcopy(self.value)
        for p in self._slow_value.parameters():
            p.requires_grad_(False)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value_updates = 0

        # --- rep-loss branch ---
        self.decoder: MultiDecoder | None = None
        self.projector: Projector | None = None
        loss_scales = dict(config.loss_scales)
        if self.rep_loss == "dreamerv3":
            self.decoder = MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = float(loss_scales.pop("recon"))
            decoder_keys = list(self.decoder.cnn_shapes) + list(self.decoder.mlp_shapes)
            for k in decoder_keys:
                loss_scales[k] = recon
            # drop unused r2dreamer scale if present
            loss_scales.pop("barlow", None)
        else:  # r2dreamer
            self.projector = Projector(self.rssm.feat_size, self.embed_size)
            self.barlow_lambd = float(config.r2dreamer.lambd)
            loss_scales.pop("recon", None)
        self._loss_scales = loss_scales
        self._log_grads = bool(config.log_grads)

        # --- optimiser ---
        self._named_params: OrderedDict[str, nn.Parameter] = OrderedDict()
        for module_name, module in self._modules_for_optim().items():
            for pname, p in module.named_parameters():
                self._named_params[f"{module_name}.{pname}"] = p

        self._optimizer = LaProp(
            list(self._named_params.values()),
            lr=float(config.lr),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
        )
        self._amp_enabled = self.device.type == "cuda"
        self._scaler = GradScaler(device=self.device.type, enabled=self._amp_enabled)
        self._agc_clip = float(config.agc)
        self._agc_pmin = float(config.pmin)

        warmup = int(config.warmup)

        def _lr_lambda(step: int) -> float:
            return min(1.0, (step + 1) / warmup) if warmup else 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=_lr_lambda)

        self.train()
        self._clone_and_freeze()

        if bool(config.compile):
            # "default" compiles in seconds; "reduce-overhead" (CUDA graphs) can
            # take 5-10 min on first run before any progress shows in the logs.
            self._cal_grad = torch.compile(self._cal_grad, mode="default")  # type: ignore[method-assign]

    # ------------------------------------------------------------------ utils

    def _modules_for_optim(self) -> dict[str, nn.Module]:
        mods: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "rssm": self.rssm,
            "reward": self.reward,
            "cont": self.cont,
            "actor": self.actor,
            "value": self.value,
        }
        if self.decoder is not None:
            mods["decoder"] = self.decoder
        if self.projector is not None:
            mods["projector"] = self.projector
        return mods

    def train(self, mode: bool = True) -> Dreamer:
        super().train(mode)
        # Slow value target is always in eval mode.
        self._slow_value.train(False)
        return self

    def _clone_and_freeze(self) -> None:
        """Maintain frozen views of every model that's queried during rollouts.

        The frozen copies share parameter *data* with the live modules but
        have ``requires_grad=False`` on every parameter, which prevents
        gradients from flowing back during the policy/value queries inside
        :meth:`_imagine` and :meth:`act`.
        """
        for name in ("encoder", "rssm", "reward", "cont", "actor", "value", "_slow_value"):
            live = getattr(self, name)
            frozen = copy.deepcopy(live)
            for (n_o, p_o), (n_n, p_n) in zip(
                live.named_parameters(), frozen.named_parameters(), strict=True
            ):
                assert n_o == n_n
                p_n.data = p_o.data  # share storage
                p_n.requires_grad_(False)
            setattr(self, f"_frozen_{name.lstrip('_')}", frozen)

    def to(self, *args: Any, **kwargs: Any) -> Dreamer:
        super().to(*args, **kwargs)
        # Frozen views point at the old storages — rebuild after .to().
        self._clone_and_freeze()
        return self

    # -------------------------------------------------------------- checkpoint

    def checkpoint_state(self) -> dict[str, Any]:
        """Full training state for crash-resume (weights, optimiser, counters)."""
        return {
            "model": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "slow_value_updates": self._slow_value_updates,
        }

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(state["model"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._scaler.load_state_dict(state["scaler"])
        self._scheduler.load_state_dict(state["scheduler"])
        self._slow_value_updates = int(state["slow_value_updates"])
        # load_state_dict copies into the existing parameter storages (which
        # the frozen rollout views share), but the frozen views' *buffers* are
        # independent deepcopies — rebuild them so act()/imagine see the
        # loaded state.
        self._clone_and_freeze()

    def _update_slow_target(self) -> None:
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(
                    self.value.parameters(), self._slow_value.parameters(), strict=True
                ):
                    s.data.copy_(mix * v.data + (1.0 - mix) * s.data)
        self._slow_value_updates += 1

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def get_initial_state(self, batch_size: int) -> TensorDict:
        stoch, deter = self.rssm.initial(batch_size)
        action = torch.zeros(batch_size, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(batch_size,)
        )

    @torch.no_grad()
    def act(
        self, obs: Mapping[str, torch.Tensor], state: TensorDict, eval_mode: bool = False
    ) -> tuple[torch.Tensor, TensorDict]:
        """Policy inference. Returns ``(action, next_state)``."""
        # CUDA-graphs step marker; only meaningful (and only importable on
        # some installs) with CUDA — importing it pulls in the full inductor
        # stack, which CPU-only dev environments may not have working.
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        p_obs = self._preprocess(dict(obs))
        embed = self._frozen_encoder(p_obs)
        stoch, deter, _ = self._frozen_rssm.obs_step(
            state["stoch"], state["deter"], state["prev_action"], embed, p_obs["is_first"]
        )
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval_mode else action_dist.rsample()
        action = self._sanitize_action(action)
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    def _sanitize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scrub + clamp a continuous action to the space the env executes.

        bounded_normal bounds only the *mean* (tanh); samples add Gaussian
        noise and are unbounded — they reached ±4 in practice (4x the EE
        controller's design velocity, 16x the jerk-penalty calibration).  The
        env clamps to [-1, 1] on execution, so anything that conditions on an
        action — the replay buffer, the RSSM's prev_action, and the imagination
        rollout the actor/critic train on — must see the *same* clamped value,
        or the policy is trained against a distribution the env never runs.

        nan_to_num first: MPS occasionally produces a non-finite sample, and a
        single NaN action NaNs every downstream latent and loss (this was the
        source of the intermittent policy/value NaNs).  clamp alone leaves NaN
        as NaN, so scrub before clamping.
        """
        if self.act_discrete:
            return action
        return torch.nan_to_num(action).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ training step

    def update(self, replay_buffer: Any) -> dict[str, torch.Tensor]:
        """Sample one batch from ``replay_buffer`` and take one optimiser step."""
        data, index, initial = replay_buffer.sample()
        p_data = self._preprocess(dict(data))
        self._update_slow_target()

        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp_enabled):
            (stoch, deter), mets = self._cal_grad(p_data, initial)

        self._scaler.unscale_(self._optimizer)
        params = list(self._named_params.values())

        if self._log_grads:
            old_params = [p.data.clone().detach() for p in params]
            grads = [p.grad for p in params if p.grad is not None]
            mets["opt/grad_norm"] = compute_global_norm(grads)
            mets["opt/grad_rms"] = compute_rms(grads)

        adaptive_grad_clip(params, self._agc_clip, self._agc_pmin)

        # Non-finite gradient guard.  A single NaN/inf gradient must never reach
        # the optimizer: without GradScaler (CUDA-only here), a disabled scaler
        # steps unconditionally, so one transient NaN (e.g. an MPS sampling
        # glitch) would corrupt every weight permanently.  On CUDA the scaler
        # already skips such steps and lowers its scale; off-CUDA we check the
        # (already-unscaled) grads ourselves.
        if self._amp_enabled:
            scale_before = self._scaler.get_scale()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            stepped = self._scaler.get_scale() >= scale_before
        else:
            stepped = all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            if stepped:
                self._optimizer.step()
        # Only advance the LR schedule when the optimizer actually ran.
        if stepped:
            self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)

        mets["opt/grad_skipped"] = torch.tensor(0.0 if stepped else 1.0)
        mets["opt/lr"] = torch.tensor(self._scheduler.get_last_lr()[0])
        mets["opt/grad_scale"] = torch.tensor(self._scaler.get_scale())
        if self._log_grads:
            updates = [(p.data - old) for p, old in zip(params, old_params, strict=True)]
            mets["opt/param_rms"] = compute_rms([p.data for p in params])
            mets["opt/update_rms"] = compute_rms(updates)

        # Don't persist latent initial states from a skipped (non-finite) step:
        # they may be NaN and would re-poison every future sample of these
        # slices, turning one transient glitch into a burst of skipped updates.
        if stepped:
            replay_buffer.update_initial_state(index, stoch.detach(), deter.detach())
        return mets

    # ------------------------------------------------------------------ losses

    def _cal_grad(
        self, data: dict[str, torch.Tensor], initial: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute world-model + actor-critic + replay-value losses, backprop them."""
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}
        b, t = data["action"].shape[:2]

        # --- world model: posterior + KL ---
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = dyn_loss.mean()
        losses["rep"] = rep_loss.mean()

        feat = self.rssm.get_feat(post_stoch, post_deter)

        # --- representation branch (decoder OR barlow-twins) ---
        if self.rep_loss == "dreamerv3":
            assert self.decoder is not None
            for key, dist in self.decoder(post_stoch, post_deter).items():
                losses[key] = -dist.log_prob(data[key]).mean()
        else:
            assert self.projector is not None
            x1 = self.projector(feat.reshape(b * t, -1))
            x2 = embed.reshape(b * t, -1).detach()
            total, invariance, redundancy = barlow_twins_loss(x1, x2, self.barlow_lambd)
            losses["barlow"] = total
            metrics["barlow/invariance"] = invariance.detach()
            metrics["barlow/redundancy"] = redundancy.detach()

        # --- reward / continue heads ---
        losses["rew"] = -self.reward(feat).log_prob(to_f32(data["reward"])).mean()
        cont_target = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        losses["con"] = -self.cont(feat).log_prob(cont_target).mean()

        metrics["dyn_entropy"] = self.rssm.get_dist(prior_logit).entropy().mean()
        metrics["rep_entropy"] = self.rssm.get_dist(post_logit).entropy().mean()

        # --- imagination rollout for actor-critic ---
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()

        disc = 1.0 - 1.0 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1.0 - imag_cont
        ret = lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = (
            weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)
        ).mean()

        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, torch.zeros_like(ret[:, -1:])], dim=1)
        losses["value"] = (
            weight[:, :-1].detach()
            * (
                -imag_value_dist.log_prob(tar_padded.detach())
                - imag_value_dist.log_prob(imag_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        ).mean()

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = ret_normed.mean()
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
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

        # --- replay-based value learning ---
        rep_last = to_f32(data["is_last"]).unsqueeze(-1)
        rep_term = to_f32(data["is_terminal"]).unsqueeze(-1)
        rep_reward = to_f32(data["reward"])
        rep_value = self._frozen_value(feat).mode()
        rep_slow = self._frozen_slow_value(feat).mode()
        boot = ret[:, 0].reshape(b, t, 1)
        rep_ret = lambda_return(rep_last, rep_term, rep_reward, rep_value, boot, disc, self.lamb)
        rep_padded = torch.cat([rep_ret, torch.zeros_like(rep_ret[:, -1:])], dim=1)
        rep_weight = 1.0 - rep_last
        value_dist = self.value(feat)
        losses["repval"] = (
            rep_weight[:, :-1]
            * (-value_dist.log_prob(rep_padded.detach()) - value_dist.log_prob(rep_slow.detach()))[
                :, :-1
            ].unsqueeze(-1)
        ).mean()
        metrics.update(tensorstats(rep_ret, "ret_replay"))
        metrics.update(tensorstats(rep_value, "value_replay"))
        metrics.update(tensorstats(rep_slow, "slow_value_replay"))

        # --- sum + backward ---
        total_loss = sum(
            (v * self._loss_scales.get(k, 1.0) for k, v in losses.items()),
            start=torch.zeros((), device=self.device),
        )
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{k}": v.detach() for k, v in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        return (post_stoch, post_deter), metrics

    # ------------------------------------------------------------------ imagination

    @torch.no_grad()
    def _imagine(
        self, start: tuple[torch.Tensor, torch.Tensor], imag_horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen policy in latent space for ``imag_horizon`` steps."""
        feats: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        stoch, deter = start
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = self._sanitize_action(self._frozen_actor(feat).rsample())
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    # ------------------------------------------------------------------ preproc

    @torch.no_grad()
    def _preprocess(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor) and data[k].dtype == torch.uint8:
                data[k] = to_f32(data[k]) / 255.0
        return data


def _resolve_action_space(act_space: Any) -> tuple[int, bool]:
    """Return ``(act_dim, is_discrete)`` for a gymnasium space."""
    if hasattr(act_space, "n"):  # Discrete
        return int(act_space.n), True
    if hasattr(act_space, "nvec"):  # MultiDiscrete
        return int(sum(act_space.nvec)), True
    # Box / continuous
    return int(sum(int(x) for x in act_space.shape)), False
