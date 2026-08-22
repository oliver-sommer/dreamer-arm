"""Dreamer agent — composition root wiring a world model + actor-critic + optimiser.

The world model is selected by ``config.wm`` (``"rssm"`` default, or
``"dinowm"``; see :mod:`dreamer_arm.core.world_model`). For the RSSM, the
representation loss is further selected by ``config.rep_loss``:

- ``"r2dreamer"`` (default): decoder-free. A linear projector maps the RSSM
  latent feature to the encoder embedding space and the two are pushed
  together via :func:`dreamer_arm.core.losses.barlow_twins_loss` (eq. 5 of
  the R2-Dreamer paper).
- ``"dreamerv3"``: decoder-based reconstruction. A
  :class:`~dreamer_arm.core.networks.MultiDecoder` reconstructs each
  observation key from the latent, with per-key NLL contributing the recon
  loss term.

Everything else -- reward / continue / actor / value heads, λ-returns with
return-EMA normalisation, slow-moving value target (all in
:mod:`dreamer_arm.core.actor_critic`), and LaProp + adaptive gradient clipping
+ GradScaler (:mod:`dreamer_arm.core.optim.step`) -- is shared across every
world model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import autocast  # type: ignore[attr-defined]

from dreamer_arm.core.actor_critic import ActorCritic
from dreamer_arm.core.optim.step import OptimStep
from dreamer_arm.core.world_model import WorldModel, build_world_model
from dreamer_arm.utils.tensor import to_f32


class Dreamer(nn.Module):
    """End-to-end Dreamer agent.

    Construct with a Hydra config, gymnasium ``obs_space`` (a ``Dict`` of
    arrays) and ``act_space``. After construction call :meth:`act` for
    rollout and :meth:`update` for a single training step.
    """

    def __init__(self, config: Any, obs_space: Any, act_space: Any) -> None:
        super().__init__()
        self.device = torch.device(config.device)
        imag_starts = config.get("imag_starts", None)
        self.imag_starts = int(imag_starts) if imag_starts is not None else None

        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.act_dim, self.act_discrete = _resolve_action_space(act_space)

        self._wm_bundle = build_world_model(config, shapes, self.act_dim, self.device)
        # Registers every world-model module (including permanently-frozen
        # ones, e.g. DINO-WM's ViT backbone) for state_dict / .to() /
        # parameter counting; the optimiser only sees `trainable_modules`.
        self.wm_modules = nn.ModuleDict(self._wm_bundle.all_modules)

        # `shape` mirrors the env action space: Discrete uses `act_space.n`,
        # continuous the raw shape tuple (not `self.act_dim`, which for
        # MultiDiscrete is `sum(nvec)` rather than the vector length).
        actor_shape = (act_space.n,) if hasattr(act_space, "n") else tuple(int(x) for x in act_space.shape)
        self.ac = ActorCritic(
            config, self._wm_bundle.feat_size, actor_shape, self.act_discrete, self.imag_starts, self.device
        )

        self._loss_scales = dict(config.loss_scales)

        self._named_params: OrderedDict[str, nn.Parameter] = OrderedDict()
        for module_name, module in self._wm_bundle.trainable_modules.items():
            for pname, p in module.named_parameters():
                self._named_params[f"wm.{module_name}.{pname}"] = p
        for pname, p in self.ac.trainable_named_parameters():
            self._named_params[f"ac.{pname}"] = p

        self._optim = OptimStep(self._named_params, config, self.device)

        self.train()

        if bool(config.compile):
            # "default" compiles in seconds; "reduce-overhead" (CUDA graphs) can
            # take 5-10 min on first run before any progress shows in the logs.
            self._cal_grad = torch.compile(self._cal_grad, mode="default")  # type: ignore[method-assign]

    # ------------------------------------------------------------------ world model views

    @property
    def wm(self) -> WorldModel:
        """The live (trainable) world-model view, used to compute the representation loss."""
        return self._wm_bundle.live

    @property
    def frozen_wm(self) -> WorldModel:
        """The no-grad world-model view, used for rollout inference and imagination."""
        return self._wm_bundle.frozen

    @property
    def replay_cache_keys(self) -> tuple[str, ...]:
        """State keys the trainer should zero-store per transition and write
        back after each update (see :class:`~dreamer_arm.core.world_model.protocol.WorldModel`).

        Not the same as every key :meth:`get_initial_state` returns: DINO-WM's
        rollout state carries a token-history window that ``act`` maintains
        in-process, but that window is a pure function of past observations
        and must never be written into the replay buffer.
        """
        return self._wm_bundle.replay_cache_keys

    def to(self, *args: Any, **kwargs: Any) -> Dreamer:
        super().to(*args, **kwargs)
        # Frozen views point at the old storages — rebuild after .to().
        self._wm_bundle.refresh_frozen()
        self.ac.refresh_frozen()
        return self

    # -------------------------------------------------------------- checkpoint

    def checkpoint_state(self) -> dict[str, Any]:
        """Full training state for crash-resume (weights, optimiser, counters)."""
        return {
            "model": self.state_dict(),
            "optim": self._optim.state_dict(),
            "slow_value_updates": self.ac.slow_value_updates,
        }

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(state["model"])
        self._optim.load_state_dict(state["optim"])
        self.ac.slow_value_updates = int(state["slow_value_updates"])
        # load_state_dict copies into the existing parameter storages (which
        # the frozen rollout views share), but the frozen views' *buffers* are
        # independent deepcopies — rebuild them so act()/imagine see the
        # loaded state.
        self._wm_bundle.refresh_frozen()
        self.ac.refresh_frozen()

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def get_initial_state(self, batch_size: int) -> TensorDict:
        state = self.wm.initial(batch_size)
        action = torch.zeros(batch_size, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({**state, "prev_action": action}, batch_size=(batch_size,))

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
        encoded = self.frozen_wm.encode_for_act(p_obs)
        next_state = self.frozen_wm.observe_step(state, encoded, state["prev_action"], p_obs["is_first"])
        feat = self.frozen_wm.get_feat(next_state)
        action = self.ac.act(feat, eval_mode)
        return action, TensorDict({**next_state, "prev_action": action}, batch_size=state.batch_size)

    # ------------------------------------------------------------------ training step

    def update(self, replay_buffer: Any) -> dict[str, torch.Tensor]:
        """Sample one batch from ``replay_buffer`` and take one optimiser step."""
        data, index, initial = replay_buffer.sample(self.replay_cache_keys)
        p_data = self._preprocess(dict(data))
        self.ac.update_slow_target()

        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._optim.amp_enabled):
            state, mets = self._cal_grad(p_data, initial)

        mets.update(self._optim.step())

        # Don't persist latent initial states from a skipped (non-finite) step:
        # they may be NaN and would re-poison every future sample of these
        # slices, turning one transient glitch into a burst of skipped updates.
        if self._optim.stepped:
            cache = {k: state[k].detach() for k in self.replay_cache_keys}
            replay_buffer.update_initial_state(index, cache)
        return mets

    # ------------------------------------------------------------------ losses

    def _cal_grad(
        self, data: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute world-model + actor-critic losses and backprop them."""
        state, losses, metrics = self.wm.loss(data, initial)
        feat = self.wm.get_feat(state)

        # A context-window world model (DINO-WM, Dreamer 4) consumes its first
        # `context` steps as warm-up and returns state/feat for only the
        # remaining `t_valid = t - context` steps; the RSSM returns one state
        # per input step (`t_valid == t`), so this is a no-op for it. `state`
        # is always aligned to `data`'s *last* `t_valid` steps.
        t_valid = feat.shape[1]
        if t_valid != data["action"].shape[1]:
            data = {k: v[:, -t_valid:] for k, v in data.items()}

        ac_losses, ac_metrics = self.ac.loss(feat, data, self.frozen_wm, state)
        losses.update(ac_losses)
        metrics.update(ac_metrics)

        total_loss = sum(
            (v * self._loss_scales.get(k, 1.0) for k, v in losses.items()),
            start=torch.zeros((), device=self.device),
        )
        self._optim.backward(total_loss)

        metrics.update({f"loss/{k}": v.detach() for k, v in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        return state, metrics

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


__all__ = ["Dreamer"]
