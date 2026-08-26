"""DINO-WM: frozen ViT patch tokens + explicit robot-state dynamics.

Faithful token-space integration: the frozen encoder's patch tokens *are*
the world-model state (no reconstruction, no stochastic latent, no KL). Two
pieces:

- :class:`DinoEncoder` — a frozen, pretrained DINOv3 ViT (via ``timm``) that
  turns an image into a grid of patch tokens. Never trained; excluded from
  the agent's optimiser and from its frozen/live cloning (it is already
  frozen, so a clone would only waste memory).
- :class:`DinoWM` — the trainable part: a small action embedder and
  :class:`CausalPredictor`, a frame-causal transformer that predicts both the
  next frame's patch tokens and explicit normalized proprioception from a
  fixed ``context``-frame window (teacher-forced during training). Exact task
  identity conditions the transition without becoming a prediction target.
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
from torch.utils.checkpoint import checkpoint

from dreamer_arm.core.networks.layers import weight_init_
from dreamer_arm.utils.tensor import rpad, symlog

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


class TaskConditionedPool(nn.Module):
    """Cross-attention pooling over spatial DINO tokens.

    The query is built from the current robot state and exact task identity,
    so different tasks may select different objects/regions while the output
    width remains one DINO token.  A residual global mean gives the heads a
    stable scene summary from the first update; cross-attention adds the
    selective part instead of forcing every reward/value/policy head to infer
    it from a spatially destructive mean.
    """

    def __init__(self, token_dim: int, condition_dim: int, heads: int) -> None:
        super().__init__()
        if token_dim % heads:
            raise ValueError(f"pool token_dim={token_dim} must be divisible by heads={heads}")
        self.query = nn.Linear(condition_dim, token_dim)
        self.token_norm = nn.RMSNorm(token_dim, eps=1e-4, dtype=torch.float32)
        self.query_norm = nn.RMSNorm(token_dim, eps=1e-4, dtype=torch.float32)
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.output_norm = nn.RMSNorm(token_dim, eps=1e-4, dtype=torch.float32)
        self.apply(weight_init_)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Pool ``(..., patches, token_dim)`` using ``(..., condition_dim)``."""
        lead = tokens.shape[:-2]
        patches, token_dim = tokens.shape[-2:]
        flat_tokens = tokens.reshape(-1, patches, token_dim)
        flat_condition = condition.reshape(-1, condition.shape[-1])
        query = self.query_norm(self.query(flat_condition)).unsqueeze(1)
        values = self.token_norm(flat_tokens)
        selected, _ = self.attention(query, values, values, need_weights=False)
        pooled = flat_tokens.mean(dim=1) + selected.squeeze(1)
        return self.output_norm(pooled).reshape(*lead, token_dim)


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
    """Trainable visual + proprioceptive dynamics with static task context.

    The old implementation learned one small embedding of proprioception and
    ``task_id``, tiled it over all image patches, and optimized one aggregate
    token MSE.  Image dimensions dominated that objective, so a constant
    auxiliary embedding was a cheap solution.  Actor/critic then saw a state
    that could be almost independent of the robot pose.

    This model assigns every modality an explicit role:

    * ``tokens`` are frozen-DINO visual state and are predicted per patch.
    * ``proprio`` is symlog-normalized robot state, retained explicitly in the
      rollout state and predicted with its own modality-balanced loss.
    * ``task_id`` is immutable context.  It conditions dynamics and all heads,
      but is copied exactly through imagination and is never a prediction
      target.
    * ``actions_out`` records the embedded action transitions in the context
      window, as before.

    Consequently the feature consumed by actor/reward/value is pooled visual
    state + current explicit proprioception + exact task identity.
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
        mlp_shapes = {k: v for k, v in obs_shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        if len(cnn_shapes) != 1:
            raise ValueError(f"DinoWM needs exactly one image observation key, got {list(cnn_shapes)}")
        self.image_key = next(iter(cnn_shapes))

        self.task_key = "task_id" if "task_id" in mlp_shapes else None
        self.task_dim = int(sum(mlp_shapes[self.task_key])) if self.task_key is not None else 0
        self.proprio_shapes = {k: v for k, v in mlp_shapes.items() if k != self.task_key}
        if not self.proprio_shapes:
            raise ValueError("DINO-WM requires at least one non-task vector observation (normally 'proprio')")
        self.proprio_dim = sum(sum(v) for v in self.proprio_shapes.values())

        self.action_dim_embed = int(config.action_dim_embed)
        self.action_embed = Embedder(act_dim, self.action_dim_embed)
        # Sliding windows per predictor call in `loss`; trades activation memory
        # for GPU occupancy.  1 measured fastest on MPS -- see the config.
        self.window_chunk = int(getattr(config, "window_chunk", 1))

        # Keep this public alias for callers/tests: tokens now contain visual
        # DINO dimensions only; state/task/action are explicit conditioning.
        self.tok_dim = embed_dim
        predictor_dim = self.tok_dim + self.proprio_dim + self.task_dim + self.action_dim_embed
        self.predictor = CausalPredictor(config.predictor, predictor_dim, num_patches, self.context)

        feat_pool = str(config.feat_pool)
        if feat_pool not in ("mean", "flatten", "task_attention"):
            raise ValueError(f"Unsupported feat_pool: {feat_pool!r}")
        self.feat_pool = feat_pool
        self.num_patches = num_patches
        self.task_pool: TaskConditionedPool | None = None
        if feat_pool == "task_attention":
            self.task_pool = TaskConditionedPool(
                self.tok_dim,
                self.proprio_dim + self.task_dim,
                int(getattr(config, "pool_heads", 8)),
            )
        pooled_size = self.tok_dim * num_patches if feat_pool == "flatten" else self.tok_dim
        self.feat_size = pooled_size + self.proprio_dim + self.task_dim

        for module in (self.action_embed, self.predictor):
            module.apply(weight_init_)

    def encode_proprio(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate vector observations and apply fixed, invertible scaling."""
        first = next(iter(self.proprio_shapes))
        b, t = data[first].shape[:2]
        proprio = torch.cat([data[k].reshape(b, t, -1).float() for k in self.proprio_shapes], dim=-1)
        return symlog(proprio)

    def tile_and_cat(self, tokens: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """``(..., P, C), (..., E) → (..., P, C+E)`` by tiling ``extra`` over patches."""
        extra = extra.unsqueeze(-2).expand(*extra.shape[:-1], tokens.shape[-2], extra.shape[-1])
        return torch.cat([tokens, extra], dim=-1)

    def initial(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        tokens = torch.zeros(batch_size, self.context, self.num_patches, self.tok_dim, device=device)
        proprio = torch.zeros(batch_size, self.context, self.proprio_dim, device=device)
        actions_out = torch.zeros(batch_size, self.context - 1, self.action_dim_embed, device=device)
        state = {"tokens": tokens, "proprio": proprio, "actions_out": actions_out}
        if self.task_key is not None:
            state[self.task_key] = torch.zeros(batch_size, self.task_dim, device=device)
        return state

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = state["tokens"]  # (N, context, P, visual_dim)
        proprio = state["proprio"]  # (N, context, proprio_dim)
        actions_out = state["actions_out"]  # (N, context - 1, AE)
        action_e = self.action_embed(action)  # (N, AE)
        ctx_actions = torch.cat([actions_out, action_e.unsqueeze(1)], dim=1)  # (N, context, AE)
        conditioning = [proprio]
        if self.task_key is not None:
            conditioning.append(state[self.task_key].unsqueeze(1).expand(-1, self.context, -1))
        conditioning.append(ctx_actions)
        pred_in = self.tile_and_cat(tokens, torch.cat(conditioning, dim=-1))
        pred_out = self.predictor(pred_in)[:, -1]
        pred_tokens = pred_out[..., : self.tok_dim]
        proprio_start = self.tok_dim
        pred_proprio = pred_out[..., proprio_start : proprio_start + self.proprio_dim].mean(dim=-2)
        new_tokens = torch.cat([tokens[:, 1:], pred_tokens.unsqueeze(1)], dim=1)
        new_proprio = torch.cat([proprio[:, 1:], pred_proprio.unsqueeze(1)], dim=1)
        new_actions_out = ctx_actions[:, 1:]
        next_state = {"tokens": new_tokens, "proprio": new_proprio, "actions_out": new_actions_out}
        if self.task_key is not None:
            next_state[self.task_key] = state[self.task_key]
        return next_state

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        last_frame = state["tokens"][..., -1, :, :]  # (..., P, tok)
        current_proprio = state["proprio"][..., -1, :]
        task_id = state.get(self.task_key) if self.task_key is not None else None
        if self.feat_pool == "mean":
            pooled = last_frame.mean(dim=-2)
        elif self.feat_pool == "flatten":
            pooled = last_frame.reshape(*last_frame.shape[:-2], -1)
        else:
            assert self.task_pool is not None
            condition = current_proprio if task_id is None else torch.cat([current_proprio, task_id], dim=-1)
            pooled = self.task_pool(last_frame, condition)
        features = [pooled, current_proprio]
        if task_id is not None:
            features.append(task_id)
        return torch.cat(features, dim=-1)

    def loss(
        self,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Teacher-forced visual and proprioceptive state prediction.

        ``action[t]`` produced state ``t``, matching the rest of the codebase.
        Task identity conditions every frame but is never predicted.

        Returns observed imagination states plus independent visual/proprio
        losses.  The adapter combines them with equal modality weight.
        """
        b, t = action.shape[:2]
        h = self.context
        if t <= h:
            raise ValueError(f"DINO-WM needs batch_length > context ({h}); got T={t}")
        if self.task_key is not None and task_id is None:
            raise ValueError("DINO-WM configured with task_id but the training batch omitted it")
        action_embed = self.action_embed(action)  # (B, T, AE); action_embed[:, j] led INTO frame j

        num_windows = t - h
        p = tokens.shape[-2]
        # Windows are folded into the batch dim rather than run one at a time:
        # a single window is (B, context) sequences, far too small to fill a
        # GPU, and 61 of them at batch_length=64 is 61 launch rounds.  The
        # chunk bounds peak activation memory, which is what forced the
        # checkpointing below -- an unchunked pass holds every window's
        # activations at once.
        chunk = max(1, min(self.window_chunk, num_windows))
        token_preds: list[torch.Tensor] = []
        proprio_preds: list[torch.Tensor] = []
        for start in range(0, num_windows, chunk):
            stop = min(start + chunk, num_windows)
            w = stop - start
            ctx_tokens = torch.stack([tokens[:, i : i + h] for i in range(start, stop)], dim=1)
            ctx_proprio = torch.stack([proprio[:, i : i + h] for i in range(start, stop)], dim=1)
            # Action tag per context frame = the action that led *out* of it,
            # i.e. the action that produced the next frame: action[i+1:i+h+1].
            ctx_actions = torch.stack([action_embed[:, i + 1 : i + h + 1] for i in range(start, stop)], dim=1)
            conditioning = [ctx_proprio]
            if self.task_key is not None:
                assert task_id is not None
                ctx_task = torch.stack([task_id[:, i : i + h] for i in range(start, stop)], dim=1)
                conditioning.append(ctx_task)
            conditioning.append(ctx_actions)
            pred_in = self.tile_and_cat(ctx_tokens, torch.cat(conditioning, dim=-1)).reshape(b * w, h, p, -1)
            # Recompute this chunk's forward during backward rather than hold
            # every chunk's internal activations live for the single combined
            # backward in model.py::_cal_grad.
            pred_out = checkpoint(self.predictor, pred_in, use_reentrant=False)
            pred_last = pred_out[:, -1]
            token_preds.append(pred_last[..., : self.tok_dim].reshape(b, w, p, self.tok_dim))
            proprio_start = self.tok_dim
            proprio_preds.append(
                pred_last[..., proprio_start : proprio_start + self.proprio_dim]
                .mean(dim=-2)
                .reshape(b, w, self.proprio_dim)
            )

        token_pred = torch.cat(token_preds, dim=1)
        proprio_pred = torch.cat(proprio_preds, dim=1)
        token_target = tokens[:, h:].detach()
        proprio_target = proprio[:, h:].detach()
        visual_loss = F.mse_loss(token_pred, token_target, reduction="none").mean(dim=(-1, -2))
        proprio_loss = F.mse_loss(proprio_pred, proprio_target, reduction="none").mean(dim=-1)

        # Strided views, not copies: `unfold` exposes every sliding window
        # without materialising any of them.  ActorCritic reads only
        # `imag_starts` of the B*W windows, and `get_feat` touches just each
        # window's last frame, so stacking all of them allocated ~300MB per
        # update at batch_length=64 to throw 93% of it away.
        state = {
            "tokens": tokens[:, 1:].unfold(1, h, 1).permute(0, 1, 4, 2, 3).detach(),
            "proprio": proprio[:, 1:].unfold(1, h, 1).permute(0, 1, 3, 2).detach(),
            "actions_out": action_embed[:, 2:].unfold(1, h - 1, 1).permute(0, 1, 3, 2).detach(),
        }
        if self.task_key is not None:
            assert task_id is not None
            # Each observed imagination state ends at tokens[h:], so its exact
            # task id is aligned to the same current state.  It remains raw and
            # detached: conditioning is not something the dynamics predicts.
            state[self.task_key] = task_id[:, h:].reshape(b, num_windows, -1).detach()
        return state, visual_loss, proprio_loss


class DinoWorldModel:
    """Adapts a frozen :class:`DinoEncoder` + trainable :class:`DinoWM` to the
    :class:`~dreamer_arm.core.world_model.protocol.WorldModel` protocol.

    ``dino_backbone`` is shared, unmodified, between the live and frozen
    instances of this adapter -- it is already frozen and permanently in
    ``eval()`` mode, so it needs no "frozen clone" of its own (unlike
    ``dinowm``, whose *trainable* parameters do need one for imagination).

    State keys: ``tokens (B, [T,] context, P, visual_dim)``, ``proprio (B,
    [T,] context, proprio_dim)``, ``actions_out (B, [T,] context - 1,
    action_dim_embed)``, and optional exact ``task_id`` -- see :class:`DinoWM`.
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

    def encode(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode visual state and normalize proprioception separately.

        ``data`` values are ``(B, T, ...)``.
        """
        encoded = {
            "tokens": self.dino_backbone(data[self.dinowm.image_key]),
            "proprio": self.dinowm.encode_proprio(data),
        }
        if self.dinowm.task_key is not None:
            encoded[self.dinowm.task_key] = data[self.dinowm.task_key].float()
        return encoded

    def encode_for_act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``encode`` needs a ``(B, T, ...)`` sequence; rollout sees one frame ``(B, ...)``."""
        obs_seq = {k: (v.unsqueeze(1) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
        encoded_seq = self.encode(obs_seq)
        return {key: value.squeeze(1) for key, value in encoded_seq.items()}

    def observe_step(
        self,
        prev_state: dict[str, torch.Tensor],
        encoded: dict[str, torch.Tensor],
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
        proprio = torch.where(rpad(is_first, prev_state["proprio"].dim() - is_first.dim()), 0.0, prev_state["proprio"])
        actions_out = torch.where(
            rpad(is_first, prev_state["actions_out"].dim() - is_first.dim()), 0.0, prev_state["actions_out"]
        )
        prev_action_e = dinowm.action_embed(prev_action)
        prev_action_e = torch.where(rpad(is_first, prev_action_e.dim() - is_first.dim()), 0.0, prev_action_e)
        new_actions_out = torch.cat([actions_out[:, 1:], prev_action_e.unsqueeze(1)], dim=1)
        new_window = torch.cat([tokens[:, 1:], encoded["tokens"].unsqueeze(1)], dim=1)
        new_proprio = torch.cat([proprio[:, 1:], encoded["proprio"].unsqueeze(1)], dim=1)
        next_state = {"tokens": new_window, "proprio": new_proprio, "actions_out": new_actions_out}
        if dinowm.task_key is not None:
            next_state[dinowm.task_key] = encoded[dinowm.task_key]
        return next_state

    def loss(
        self, data: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        encoded = self.encode(data)
        task_id = encoded.get(self.dinowm.task_key) if self.dinowm.task_key is not None else None
        state, visual_loss, proprio_loss = self.dinowm.loss(
            encoded["tokens"], encoded["proprio"], data["action"], task_id
        )
        # Equal modality weight is intentional.  Averaging all dimensions in
        # one tensor would make 64*384 visual values drown out 10 proprio
        # values and recreate the state-collapse bug this separation fixes.
        pred_loss = visual_loss.mean() + proprio_loss.mean()
        metrics = {
            "pred/visual_mse": visual_loss.mean().detach(),
            "pred/proprio_mse": proprio_loss.mean().detach(),
        }
        return state, {"pred": pred_loss}, metrics
