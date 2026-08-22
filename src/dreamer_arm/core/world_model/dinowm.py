"""DINO-WM: frozen ViT patch tokens + a frame-causal transformer predicting
next-frame tokens (Zhou et al., 2024, arXiv:2411.04983).

Faithful token-space integration: the frozen encoder's patch tokens *are*
the world-model state (no reconstruction, no stochastic latent, no KL). Two
pieces:

- :class:`DinoEncoder` — a frozen, pretrained DINOv3 ViT (via ``timm``) that
  turns an image into a grid of patch tokens. Never trained; excluded from
  the agent's optimiser and from its frozen/live cloning (it is already
  frozen, so a clone would only waste memory).
- :class:`DinoWM` — the trainable part: small linear embedders for
  proprio/task-id and action, and :class:`CausalPredictor`, a frame-causal
  transformer that predicts the next frame's patch tokens from a fixed
  ``context``-frame window (teacher-forced during training).
- :class:`DinoWorldModel` — adapts ``DinoWM`` + ``DinoEncoder`` to the
  :class:`~dreamer_arm.core.world_model.protocol.WorldModel` protocol.

``DinoWM`` does not itself implement
:class:`~dreamer_arm.core.world_model.protocol.WorldModel` (it needs the
frozen encoder's output as an *input*, not something it owns) -- that is
exactly what :class:`DinoWorldModel` is for.
"""

from __future__ import annotations

import re
from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import nn

from dreamer_arm.utils.modules import weight_init_
from dreamer_arm.utils.tensor import rpad

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_EXCLUDED_OBS_KEYS = ("is_first", "is_last", "is_terminal", "reward")


class DinoEncoder(nn.Module):
    """Frozen pretrained ViT patch-token encoder. Always ``eval()``, never trained."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.image_size = int(config.image_size)
        self._backbone = timm.create_model(
            str(config.name),
            pretrained=bool(config.pretrained),
            num_classes=0,
            img_size=self.image_size,
        )
        self._backbone.requires_grad_(False)
        self._backbone.eval()
        self.num_prefix_tokens = int(self._backbone.num_prefix_tokens)
        self.patch_size = int(self._backbone.patch_embed.patch_size[0])
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}")
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.embed_dim = int(self._backbone.embed_dim)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> DinoEncoder:
        # Frozen backbone: always eval, regardless of the owning agent's mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, T, H, W, C)`` float ``[0, 1]`` (or uint8) → ``(B, T, P, D)`` patch tokens."""
        b, t, h, w, c = image.shape
        x = image.reshape(b * t, h, w, c).permute(0, 3, 1, 2).float()
        if image.dtype == torch.uint8:
            x = x / 255.0
        x = (x - self._mean) / self._std  # ty: ignore[unsupported-operator]
        tokens = self._backbone.forward_features(x)
        tokens = tokens[:, self.num_prefix_tokens :, :]
        return tokens.reshape(b, t, self.num_patches, self.embed_dim)


class Embedder(nn.Module):
    """Per-timestep linear embed (proprio/task-id, or action)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_mask_matrix(num_frames: int, tokens_per_frame: int) -> torch.Tensor:
    """Block-lower-triangular frame-causal mask: ``(F*P, F*P)`` bool, ``True`` = attend.

    Token ``i`` may attend to token ``j`` iff ``j``'s frame is not later than
    ``i``'s frame -- full attention within a frame, causal across frames.
    """
    frame_idx = torch.arange(num_frames).repeat_interleave(tokens_per_frame)
    return frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.norm = nn.RMSNorm(dim, eps=1e-4, dtype=torch.float32)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, self.dim_head).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-4, dtype=torch.float32)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class CausalPredictor(nn.Module):
    """Frame-causal transformer over a fixed ``num_frames``-frame token window."""

    def __init__(self, config: Any, dim: int, tokens_per_frame: int, num_frames: int) -> None:
        super().__init__()
        depth = int(config.depth)
        heads = int(config.heads)
        dim_head = int(config.dim_head)
        mlp_dim = int(config.mlp_dim)
        self.tokens_per_frame = tokens_per_frame
        self.num_frames = num_frames
        self.pos_emb = nn.Parameter(torch.zeros(1, num_frames * tokens_per_frame, dim))
        self.layers = nn.ModuleList(
            [nn.ModuleList([Attention(dim, heads, dim_head), FeedForward(dim, mlp_dim)]) for _ in range(depth)]
        )
        self.norm = nn.RMSNorm(dim, eps=1e-4, dtype=torch.float32)
        self.register_buffer("_mask", generate_mask_matrix(num_frames, tokens_per_frame), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, F, P, C) → (B, F, P, C)``, ``F`` must equal ``num_frames``."""
        b, f, p, c = tokens.shape
        assert f == self.num_frames, f"CausalPredictor expects exactly {self.num_frames} frames, got {f}"
        x = tokens.reshape(b, f * p, c) + self.pos_emb
        for attn, ff in self.layers:
            x = x + attn(x, self._mask)
            x = x + ff(x)
        x = self.norm(x)
        return x.reshape(b, f, p, c)


class DinoWM(nn.Module):
    """Trainable half of DINO-WM: modality embedders + the causal predictor.

    Does *not* hold the frozen :class:`DinoEncoder` (see module docstring).

    State (as consumed by the ``agent.py`` adapter): ``tokens (B, [T,]
    context, P, tok)`` -- the trailing ``context``-frame window of patch(+extra)
    tokens; ``actions_out (B, [T,] context-1, action_dim_embed)`` -- the
    *embedded* actions that led out of the first ``context-1`` frames of that
    window (the action leading out of the last frame is supplied fresh to
    :meth:`img_step`, since it is what is being decided).
    """

    def __init__(
        self, config: Any, shapes: dict[str, tuple[int, ...]], act_dim: int, num_patches: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.context = int(config.context)
        if self.context < 2:
            raise ValueError("DINO-WM needs context >= 2 (at least one action transition per window)")

        obs_shapes = {k: v for k, v in shapes.items() if k not in _EXCLUDED_OBS_KEYS and not k.startswith("log_")}
        cnn_shapes = {k: v for k, v in obs_shapes.items() if len(v) == 3 and re.match(config.cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in obs_shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        if len(cnn_shapes) != 1:
            raise ValueError(f"DinoWM needs exactly one image observation key, got {list(cnn_shapes)}")
        self.image_key = next(iter(cnn_shapes))

        self.extra_dim = int(config.extra_dim) if self.mlp_shapes else 0
        self.proprio_embed: Embedder | None = None
        if self.mlp_shapes:
            proprio_dim = sum(sum(v) for v in self.mlp_shapes.values())
            self.proprio_embed = Embedder(proprio_dim, self.extra_dim)

        self.action_dim_embed = int(config.action_dim_embed)
        self.action_embed = Embedder(act_dim, self.action_dim_embed)

        self.tok_dim = embed_dim + self.extra_dim
        predictor_dim = self.tok_dim + self.action_dim_embed
        self.predictor = CausalPredictor(config.predictor, predictor_dim, num_patches, self.context)

        feat_pool = str(config.feat_pool)
        if feat_pool not in ("mean", "flatten"):
            raise ValueError(f"Unsupported feat_pool: {feat_pool!r}")
        self.feat_pool = feat_pool
        self.num_patches = num_patches
        self.feat_size = self.tok_dim if feat_pool == "mean" else self.tok_dim * num_patches

        for module in (self.proprio_embed, self.action_embed, self.predictor):
            if module is not None:
                module.apply(weight_init_)

    # ------------------------------------------------------------ embedding

    def embed_extra(self, data: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Proprio/task-id → ``(B, T, extra_dim)``, or ``None`` if there is none."""
        if self.proprio_embed is None:
            return None
        first = next(iter(self.mlp_shapes))
        b, t = data[first].shape[:2]
        proprio = torch.cat([data[k].reshape(b, t, -1) for k in self.mlp_shapes], dim=-1)
        return self.proprio_embed(proprio)

    def tile_and_cat(self, tokens: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """``(..., P, C), (..., E) → (..., P, C+E)`` by tiling ``extra`` over patches."""
        extra = extra.unsqueeze(-2).expand(*extra.shape[:-1], tokens.shape[-2], extra.shape[-1])
        return torch.cat([tokens, extra], dim=-1)

    # ------------------------------------------------------------- rollout

    def initial(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        tokens = torch.zeros(batch_size, self.context, self.num_patches, self.tok_dim, device=device)
        actions_out = torch.zeros(batch_size, self.context - 1, self.action_dim_embed, device=device)
        return {"tokens": tokens, "actions_out": actions_out}

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = state["tokens"]  # (N, context, P, tok)
        actions_out = state["actions_out"]  # (N, context - 1, AE)
        action_e = self.action_embed(action)  # (N, AE)
        ctx_actions = torch.cat([actions_out, action_e.unsqueeze(1)], dim=1)  # (N, context, AE)
        pred_in = self.tile_and_cat(tokens, ctx_actions)
        pred_out = self.predictor(pred_in)[:, -1, :, : self.tok_dim]
        new_tokens = torch.cat([tokens[:, 1:], pred_out.unsqueeze(1)], dim=1)
        new_actions_out = ctx_actions[:, 1:]
        return {"tokens": new_tokens, "actions_out": new_actions_out}

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        last_frame = state["tokens"][..., -1, :, :]  # (..., P, tok)
        if self.feat_pool == "mean":
            return last_frame.mean(dim=-2)
        return last_frame.reshape(*last_frame.shape[:-2], -1)

    # ---------------------------------------------------------------- loss

    def loss(self, tokens: torch.Tensor, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Teacher-forced next-frame-token prediction.

        ``tokens (B, T, P, tok)`` (already frozen-encoder + extra-modality
        tokens), ``action (B, T, A)`` (``action[t]`` produced ``tokens[t]``,
        matching the rest of the codebase's convention).

        Returns ``(state, pred_loss)`` where ``state`` holds the observed
        trajectory (``tokens``/``actions_out``, one window per valid target
        position) for imagination start states.
        """
        b, t = action.shape[:2]
        h = self.context
        if t <= h:
            raise ValueError(f"DINO-WM needs batch_length > context ({h}); got T={t}")
        action_embed = self.action_embed(action)  # (B, T, AE); action_embed[:, j] led INTO frame j

        num_windows = t - h
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        state_tokens: list[torch.Tensor] = []
        state_actions: list[torch.Tensor] = []
        for i in range(num_windows):
            ctx_tokens = tokens[:, i : i + h]  # (B, H, P, tok)
            # Action tag per context frame = the action that led *out* of it,
            # i.e. the action that produced the next frame: action[i+1:i+h+1].
            ctx_actions = action_embed[:, i + 1 : i + h + 1]
            pred_in = self.tile_and_cat(ctx_tokens, ctx_actions)
            pred_out = self.predictor(pred_in)[:, -1, :, : self.tok_dim]
            preds.append(pred_out)
            targets.append(tokens[:, i + h].detach())

            state_tokens.append(tokens[:, i + 1 : i + h + 1].detach())
            state_actions.append(action_embed[:, i + 2 : i + h + 1].detach())

        pred = torch.stack(preds, dim=1)
        target = torch.stack(targets, dim=1)
        pred_loss = F.mse_loss(pred, target, reduction="none").mean(dim=(-1, -2))  # (B, num_windows)

        state = {
            "tokens": torch.stack(state_tokens, dim=1),
            "actions_out": torch.stack(state_actions, dim=1),
        }
        return state, pred_loss


class DinoWorldModel:
    """Adapts a frozen :class:`DinoEncoder` + trainable :class:`DinoWM` to the
    :class:`~dreamer_arm.core.world_model.protocol.WorldModel` protocol.

    ``dino_backbone`` is shared, unmodified, between the live and frozen
    instances of this adapter -- it is already frozen and permanently in
    ``eval()`` mode, so it needs no "frozen clone" of its own (unlike
    ``dinowm``, whose *trainable* parameters do need one for imagination).

    State keys: ``tokens (B, [T,] context, P, tok)``, ``actions_out (B, [T,]
    context - 1, action_dim_embed)`` -- see :class:`DinoWM`.
    """

    replay_cache_keys: tuple[str, ...] = ()

    def __init__(self, dinowm: DinoWM, dino_backbone: DinoEncoder, device: torch.device) -> None:
        self.dinowm = dinowm
        self.dino_backbone = dino_backbone
        self.device = device

    @property
    def feat_size(self) -> int:
        return self.dinowm.feat_size

    def initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        return self.dinowm.initial(batch_size, self.device)

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.dinowm.img_step(state, action)

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.dinowm.get_feat(state)

    def encode(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Frozen backbone + proprio embed → per-frame tokens.

        ``data`` values are ``(B, T, ...)``.
        """
        tokens = self.dino_backbone(data[self.dinowm.image_key])
        extra = self.dinowm.embed_extra(data)
        if extra is not None:
            tokens = self.dinowm.tile_and_cat(tokens, extra)
        return tokens

    def encode_for_act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """``encode`` needs a ``(B, T, ...)`` sequence; rollout sees one frame ``(B, ...)``."""
        obs_seq = {k: (v.unsqueeze(1) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        return self.encode(obs_seq).squeeze(1)

    def observe_step(
        self,
        prev_state: dict[str, torch.Tensor],
        encoded: torch.Tensor,
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fold one freshly-observed frame's tokens into the context window.

        Mirrors :meth:`~dreamer_arm.core.world_model.rssm.RSSM.obs_step`'s
        reset handling: the *old* window/action-history is zeroed for envs at
        an episode boundary before the new observation is folded in, so a
        fresh episode never sees the previous one's context.
        """
        dinowm = self.dinowm
        tokens = torch.where(rpad(is_first, prev_state["tokens"].dim() - is_first.dim()), 0.0, prev_state["tokens"])
        actions_out = torch.where(
            rpad(is_first, prev_state["actions_out"].dim() - is_first.dim()), 0.0, prev_state["actions_out"]
        )
        prev_action_e = dinowm.action_embed(prev_action)
        prev_action_e = torch.where(rpad(is_first, prev_action_e.dim() - is_first.dim()), 0.0, prev_action_e)
        new_actions_out = torch.cat([actions_out[:, 1:], prev_action_e.unsqueeze(1)], dim=1)
        new_window = torch.cat([tokens[:, 1:], encoded.unsqueeze(1)], dim=1)
        return {"tokens": new_window, "actions_out": new_actions_out}

    def loss(
        self, data: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        tokens = self.encode(data)
        state, pred_loss = self.dinowm.loss(tokens, data["action"])
        return state, {"pred": pred_loss.mean()}, {}
