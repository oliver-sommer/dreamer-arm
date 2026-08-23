"""Builds the world model named by ``config.wm``.

Each world model is a (trainable modules, live adapter, frozen adapter)
bundle:

- ``trainable_modules`` -- handed to the agent's optimiser.
- ``all_modules`` -- every module that must be registered on the agent for
  ``state_dict()`` / ``.to()`` / parameter counting, including permanently
  frozen pieces (e.g. DINO-WM's ViT backbone) that are excluded from
  ``trainable_modules``.
- ``live`` -- the :class:`~dreamer_arm.core.world_model.protocol.WorldModel`
  view used to compute the representation loss (trains).
- ``frozen`` -- the no-grad view used for rollout inference and imagination;
  call :meth:`WorldModelBundle.refresh_frozen` after anything that reallocates
  the live modules' storage (``.to()``, checkpoint load).

Adding a new world model (e.g. Dreamer 4) means adding one ``_build_*``
function here plus a branch in :func:`build_world_model`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from dreamer_arm.core.frozen import freeze_clone
from dreamer_arm.core.networks import MultiDecoder, MultiEncoder
from dreamer_arm.core.world_model.dinowm import DinoEncoder, DinoWM, DinoWorldModel
from dreamer_arm.core.world_model.protocol import WorldModel
from dreamer_arm.core.world_model.rssm import RSSM, Projector, RSSMWorldModel


class WorldModelBundle:
    """Everything :class:`~dreamer_arm.core.model.Dreamer` needs from a constructed world model."""

    def __init__(
        self,
        live: WorldModel,
        frozen: WorldModel,
        trainable_modules: dict[str, nn.Module],
        all_modules: dict[str, nn.Module],
        replay_cache_keys: tuple[str, ...],
        feat_size: int,
        rebuild_frozen: Callable[[], WorldModel],
    ) -> None:
        self.live = live
        self.frozen = frozen
        self.trainable_modules = trainable_modules
        self.all_modules = all_modules
        self.replay_cache_keys = replay_cache_keys
        self.feat_size = feat_size
        self._rebuild_frozen = rebuild_frozen

    def refresh_frozen(self) -> None:
        """Rebuild the frozen view after ``.to()`` / checkpoint load moves storage."""
        self.frozen = self._rebuild_frozen()


def build_world_model(
    config: Any, shapes: dict[str, tuple[int, ...]], act_dim: int, device: torch.device
) -> WorldModelBundle:
    wm = str(config.get("wm", "rssm"))
    if wm not in ("rssm", "dinowm"):
        raise NotImplementedError(f"core.model.wm={wm!r} is not implemented yet")

    imag_starts = config.get("imag_starts", None)
    if wm != "rssm" and imag_starts is None:
        # The replay-value bootstrap (see `ActorCritic.loss`) assumes the
        # world model's per-step state has one entry per replay (b, t)
        # position, which only the RSSM's recurrent state does. Token-space
        # models produce fewer (context-shortened) states, so they must
        # always take the value-function bootstrap branch instead.
        raise ValueError(f"core.model.wm={wm!r} requires core.model.imag_starts to be set (not null)")

    if wm == "rssm":
        return _build_rssm(config, shapes, act_dim)
    return _build_dinowm(config, shapes, act_dim, device)


def _build_rssm(config: Any, shapes: dict[str, tuple[int, ...]], act_dim: int) -> WorldModelBundle:
    encoder = MultiEncoder(config.encoder, shapes)
    rssm = RSSM(config.rssm, encoder.out_dim, act_dim)
    feat_size = rssm.feat_size

    rep_loss = str(config.rep_loss)
    if rep_loss not in ("r2dreamer", "dreamerv3"):
        raise ValueError(f"Unsupported rep_loss={rep_loss!r}. This implementation supports 'r2dreamer' or 'dreamerv3'.")

    decoder: MultiDecoder | None = None
    projector: Projector | None = None
    barlow_lambd = 0.0
    if rep_loss == "dreamerv3":
        decoder = MultiDecoder(config.decoder, rssm._deter, rssm.flat_stoch, shapes)
    else:  # r2dreamer
        projector = Projector(feat_size, encoder.out_dim)
        barlow_lambd = float(config.r2dreamer.lambd)

    all_modules: dict[str, nn.Module] = {"encoder": encoder, "rssm": rssm}
    if decoder is not None:
        all_modules["decoder"] = decoder
    if projector is not None:
        all_modules["projector"] = projector

    live = RSSMWorldModel(
        rssm,
        encoder,
        kl_free=float(config.kl_free),
        rep_loss=rep_loss,
        decoder=decoder,
        projector=projector,
        barlow_lambd=barlow_lambd,
    )

    def _rebuild_frozen() -> RSSMWorldModel:
        return RSSMWorldModel(freeze_clone(rssm), freeze_clone(encoder))

    return WorldModelBundle(
        live=live,
        frozen=_rebuild_frozen(),
        trainable_modules=dict(all_modules),  # everything here is trainable for RSSM
        all_modules=all_modules,
        replay_cache_keys=RSSMWorldModel.replay_cache_keys,
        feat_size=feat_size,
        rebuild_frozen=_rebuild_frozen,
    )


def _build_dinowm(
    config: Any, shapes: dict[str, tuple[int, ...]], act_dim: int, device: torch.device
) -> WorldModelBundle:
    dino_cfg = config.dinowm
    # Frozen, never trained/cloned: `dino_backbone` is registered on the agent
    # (for state_dict / parameter counting) but excluded from both the
    # optimiser and frozen-view rebuilding -- it's already frozen and in
    # eval mode, so the live object doubles as its own "frozen view".
    dino_backbone = DinoEncoder(dino_cfg.encoder)
    dinowm = DinoWM(dino_cfg, shapes, act_dim, dino_backbone.num_patches, dino_backbone.embed_dim)
    feat_size = dinowm.feat_size

    live = DinoWorldModel(dinowm, dino_backbone, device)

    def _rebuild_frozen() -> DinoWorldModel:
        return DinoWorldModel(freeze_clone(dinowm), dino_backbone, device)

    return WorldModelBundle(
        live=live,
        frozen=_rebuild_frozen(),
        trainable_modules={"dinowm": dinowm},
        all_modules={"dino_backbone": dino_backbone, "dinowm": dinowm},
        replay_cache_keys=DinoWorldModel.replay_cache_keys,
        feat_size=feat_size,
        rebuild_frozen=_rebuild_frozen,
    )
