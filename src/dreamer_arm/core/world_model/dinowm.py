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
  fixed ``context``-frame window. Dense one-step teacher forcing is augmented
  with bounded open-loop overshooting on sampled replay contexts so training
  covers the autoregressive path used by imagination. Exact task identity
  conditions the transition without becoming a prediction target.
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


def _skill_vs_baseline(error: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Bounded relative skill: -1 worse, 0 tied, +1 better than baseline."""
    return (baseline - error) / (baseline + error).clamp_min(1e-8)


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


class ConstraintPredictor(nn.Module):
    """Predict controller projection from explicit robot state + command."""

    def __init__(self, inp_dim: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(inp_dim, hidden),
            nn.RMSNorm(hidden, eps=1e-4, dtype=torch.float32),
            nn.SiLU(),
        )
        self.clamp = nn.Linear(hidden, 3)
        self.retained = nn.Linear(hidden, 3)
        self.achieved = nn.Linear(hidden, 3)
        self.apply(weight_init_)
        with torch.no_grad():
            self.clamp.weight.zero_()
            self.clamp.bias.fill_(-4.59511985013459)  # logit(0.01)
            self.retained.weight.zero_()
            self.retained.bias.zero_()
            self.achieved.weight.zero_()
            self.achieved.bias.zero_()

    def forward(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(condition)
        return {
            "clamp_logits": self.clamp(hidden),
            "retained_xyz": self.retained(hidden),
            "achieved_xyz": self.achieved(hidden),
        }


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
        self.act_dim = act_dim
        # Sliding windows per predictor call in `loss`; trades activation memory
        # for GPU occupancy.  1 measured fastest on MPS -- see the config.
        self.window_chunk = int(getattr(config, "window_chunk", 1))

        # One-step teacher forcing alone does not constrain the distribution
        # the actor actually sees: Dreamer recursively feeds this predictor's
        # own outputs back for `imag_horizon` steps. A small replay-context
        # subsample therefore receives an open-loop overshooting loss. Keep it
        # configurable because its cost is proportional to starts * horizon.
        rollout_cfg = getattr(config, "rollout", None)
        self.rollout_starts = int(getattr(rollout_cfg, "starts", 0)) if rollout_cfg is not None else 0
        self.rollout_horizons = self._validate_horizons(
            getattr(rollout_cfg, "horizons", (1, 3, 5, 10, 15)) if rollout_cfg is not None else ()
        )
        self.rollout_train_horizons = self._validate_horizons(
            getattr(rollout_cfg, "train_horizons", (3, 5, 10)) if rollout_cfg is not None else ()
        )
        self.rollout_motion_weight = (
            float(getattr(rollout_cfg, "motion_weight", 0.0)) if rollout_cfg is not None else 0.0
        )
        if self.rollout_starts < 0:
            raise ValueError(f"rollout.starts must be non-negative, got {self.rollout_starts}")
        if self.rollout_motion_weight < 0.0:
            raise ValueError(f"rollout.motion_weight must be non-negative, got {self.rollout_motion_weight}")
        if self.rollout_starts and not self.rollout_horizons:
            raise ValueError("rollout.horizons must be non-empty when rollout.starts is positive")
        if not set(self.rollout_train_horizons).issubset(self.rollout_horizons):
            raise ValueError("rollout.train_horizons must be a subset of rollout.horizons")

        # Keep this public alias for callers/tests: tokens now contain visual
        # DINO dimensions only; state/task/action are explicit conditioning.
        self.tok_dim = embed_dim
        predictor_dim = self.tok_dim + self.proprio_dim + self.task_dim + self.action_dim_embed
        self.predictor = CausalPredictor(config.predictor, predictor_dim, num_patches, self.context)
        self.residual = bool(getattr(config, "residual", False))
        self.prediction_head: nn.Linear | None = None
        if self.residual:
            self.prediction_head = nn.Linear(predictor_dim, self.tok_dim + self.proprio_dim)
            nn.init.zeros_(self.prediction_head.weight)
            nn.init.zeros_(self.prediction_head.bias)

        constraint_cfg = getattr(config, "constraint", None)
        self.constraint_enabled = bool(getattr(constraint_cfg, "enabled", False))
        self.constraint_motion_scale = float(getattr(constraint_cfg, "motion_scale", 100.0))
        self.constraint: ConstraintPredictor | None = None
        if self.constraint_enabled:
            if self.constraint_motion_scale <= 0.0:
                raise ValueError("constraint.motion_scale must be positive")
            constraint_dim = self.proprio_dim + self.task_dim + act_dim
            self.constraint = ConstraintPredictor(constraint_dim, int(getattr(constraint_cfg, "hidden", 128)))

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

    @staticmethod
    def _validate_horizons(values: Any) -> tuple[int, ...]:
        horizons = tuple(sorted({int(value) for value in values}))
        if any(value < 1 for value in horizons):
            raise ValueError(f"rollout horizons must be positive, got {horizons}")
        return horizons

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
        return self._img_step(state, action, checkpoint_predictor=False)

    def _img_step(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        *,
        checkpoint_predictor: bool,
    ) -> dict[str, torch.Tensor]:
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
        if checkpoint_predictor and torch.is_grad_enabled():
            pred_hidden = checkpoint(self.predictor, pred_in, use_reentrant=False)[:, -1]
        else:
            pred_hidden = self.predictor(pred_in)[:, -1]
        pred_tokens, pred_proprio = self._decode_prediction(pred_hidden, tokens[:, -1], proprio[:, -1])
        new_tokens = torch.cat([tokens[:, 1:], pred_tokens.unsqueeze(1)], dim=1)
        new_proprio = torch.cat([proprio[:, 1:], pred_proprio.unsqueeze(1)], dim=1)
        new_actions_out = ctx_actions[:, 1:]
        next_state = {"tokens": new_tokens, "proprio": new_proprio, "actions_out": new_actions_out}
        if self.task_key is not None:
            next_state[self.task_key] = state[self.task_key]
        return next_state

    def _decode_prediction(
        self,
        hidden: torch.Tensor,
        latest_tokens: torch.Tensor,
        latest_proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prediction_head is None:
            prediction = hidden[..., : self.tok_dim + self.proprio_dim]
        else:
            prediction = self.prediction_head(hidden)
        pred_tokens = prediction[..., : self.tok_dim]
        pred_proprio = prediction[..., self.tok_dim :].mean(dim=-2)
        if self.residual:
            pred_tokens = latest_tokens + pred_tokens
            pred_proprio = latest_proprio + pred_proprio
        return pred_tokens, pred_proprio

    def constraint_outputs(
        self, state: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        """Predict clamp events and retained/achieved XYZ for ``(state, action)``."""
        if self.constraint is None:
            return None
        condition = [state["proprio"][..., -1, :]]
        if self.task_key is not None:
            condition.append(state[self.task_key])
        condition.append(action)
        outputs = self.constraint(torch.cat(condition, dim=-1))
        # The regression heads train in centimetre-scale units for useful
        # gradients, but every public controller diagnostic remains SI.
        return {
            "clamp_logits": outputs["clamp_logits"],
            "retained_xyz": outputs["retained_xyz"] / self.constraint_motion_scale,
            "achieved_xyz": outputs["achieved_xyz"] / self.constraint_motion_scale,
        }

    def rollout_loss(
        self,
        state: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Train and diagnose the autoregressive path used by imagination.

        Loss is applied only at ``rollout_train_horizons``. Diagnostics
        continue through the largest requested horizon under ``no_grad`` after
        the last trained horizon. Persistence baselines keep a low DINO-token
        MSE honest on static scenes; action ablations expose ignored controls.
        """
        if self.rollout_starts == 0 or not self.rollout_horizons:
            return None, {}

        b, t = action.shape[:2]
        num_windows = state["tokens"].shape[1]
        active_horizons = tuple(horizon for horizon in self.rollout_horizons if horizon < num_windows)
        if not active_horizons:
            return None, {}
        active_train_horizons = tuple(horizon for horizon in self.rollout_train_horizons if horizon in active_horizons)
        max_horizon = active_horizons[-1]
        starts_per_row = num_windows - max_horizon
        assert starts_per_row > 0, (t, self.context, max_horizon)

        count = min(self.rollout_starts, b * starts_per_row)
        if count <= b:
            # MT replay is balanced by env slot; distinct slices tend to cover
            # more tasks than a flat sample that may repeat one slice.
            rows = torch.randperm(b, device=action.device)[:count]
            cols = torch.randint(starts_per_row, (count,), device=action.device)
        else:
            flat = torch.randperm(b * starts_per_row, device=action.device)[:count]
            rows, cols = flat // starts_per_row, flat % starts_per_row

        rollout_state = {key: value[rows, cols].detach() for key, value in state.items()}
        end = self.context + cols
        initial_tokens = rollout_state["tokens"][:, -1].detach()
        initial_proprio = rollout_state["proprio"][:, -1].detach()

        # Counterfactual controls are diagnostic only. Comparing their target
        # error with the real-action error catches both ignored and wrongly
        # signed action conditioning.
        first_action = action[rows, end + 1]
        with torch.no_grad():
            zero_state = self.img_step(rollout_state, torch.zeros_like(first_action))
            shuffled_state = self.img_step(rollout_state, first_action.roll(1, dims=0))

        visual_train_losses: list[torch.Tensor] = []
        proprio_train_losses: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] = {
            "rollout/start_count": torch.as_tensor(float(count), device=action.device),
            "rollout/max_horizon": torch.as_tensor(float(max_horizon), device=action.device),
        }
        max_train_horizon = max(active_train_horizons, default=0)
        detached_rollout = False

        for horizon in range(1, max_horizon + 1):
            rollout_action = action[rows, end + horizon]
            if horizon <= max_train_horizon:
                rollout_state = self._img_step(rollout_state, rollout_action, checkpoint_predictor=True)
            else:
                if not detached_rollout:
                    rollout_state = {key: value.detach() for key, value in rollout_state.items()}
                    detached_rollout = True
                with torch.no_grad():
                    rollout_state = self.img_step(rollout_state, rollout_action)

            target_tokens = tokens[rows, end + horizon].detach()
            target_proprio = proprio[rows, end + horizon].detach()
            visual_patch_mse = F.mse_loss(rollout_state["tokens"][:, -1], target_tokens, reduction="none").mean(dim=-1)
            visual_mse = visual_patch_mse.mean()
            proprio_mse = F.mse_loss(rollout_state["proprio"][:, -1], target_proprio)
            visual_motion = F.mse_loss(initial_tokens, target_tokens, reduction="none").mean(dim=-1)
            relative_motion = visual_motion / visual_motion.mean(dim=-1, keepdim=True).clamp_min(1e-8)
            motion_weights = 1.0 + self.rollout_motion_weight * relative_motion
            visual_motion_weighted_mse = (visual_patch_mse * motion_weights).sum() / motion_weights.sum()

            if horizon in active_train_horizons:
                visual_train_losses.append(visual_motion_weighted_mse)
                proprio_train_losses.append(proprio_mse)

            if horizon in active_horizons:
                visual_persistence = F.mse_loss(initial_tokens, target_tokens)
                proprio_persistence = F.mse_loss(initial_proprio, target_proprio)
                metrics[f"rollout/visual_mse_h{horizon}"] = visual_mse.detach()
                metrics[f"rollout/visual_motion_weighted_mse_h{horizon}"] = visual_motion_weighted_mse.detach()
                metrics[f"rollout/proprio_mse_h{horizon}"] = proprio_mse.detach()
                metrics[f"rollout/visual_persistence_mse_h{horizon}"] = visual_persistence.detach()
                metrics[f"rollout/proprio_persistence_mse_h{horizon}"] = proprio_persistence.detach()
                metrics[f"rollout/visual_skill_h{horizon}"] = _skill_vs_baseline(
                    visual_mse.detach(), visual_persistence
                )
                metrics[f"rollout/proprio_skill_h{horizon}"] = _skill_vs_baseline(
                    proprio_mse.detach(), proprio_persistence
                )

            if horizon == 1:
                real_combined = visual_mse.detach() + proprio_mse.detach()
                zero_combined = F.mse_loss(zero_state["tokens"][:, -1], target_tokens) + F.mse_loss(
                    zero_state["proprio"][:, -1], target_proprio
                )
                shuffled_combined = F.mse_loss(shuffled_state["tokens"][:, -1], target_tokens) + F.mse_loss(
                    shuffled_state["proprio"][:, -1], target_proprio
                )
                metrics["rollout/action_zero_excess_mse"] = (zero_combined - real_combined).detach()
                metrics["rollout/action_shuffled_excess_mse"] = (shuffled_combined - real_combined).detach()
                metrics["rollout/action_zero_effect_mse"] = (
                    F.mse_loss(zero_state["tokens"][:, -1], rollout_state["tokens"][:, -1].detach())
                    + F.mse_loss(zero_state["proprio"][:, -1], rollout_state["proprio"][:, -1].detach())
                ).detach()

        if not visual_train_losses:
            return None, metrics
        overshoot = torch.stack(visual_train_losses).mean() + torch.stack(proprio_train_losses).mean()
        return overshoot, metrics

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
            pred_hidden = checkpoint(self.predictor, pred_in, use_reentrant=False)[:, -1]
            flat_tokens = ctx_tokens.reshape(b * w, h, p, self.tok_dim)
            flat_proprio = ctx_proprio.reshape(b * w, h, self.proprio_dim)
            pred_tokens, pred_proprio = self._decode_prediction(pred_hidden, flat_tokens[:, -1], flat_proprio[:, -1])
            token_preds.append(pred_tokens.reshape(b, w, p, self.tok_dim))
            proprio_preds.append(pred_proprio.reshape(b, w, self.proprio_dim))

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

    def predict_constraints(
        self, state: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        return self.dinowm.constraint_outputs(state, action)

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
        visual_persistence = F.mse_loss(
            encoded["tokens"][:, self.dinowm.context - 1 : -1],
            encoded["tokens"][:, self.dinowm.context :],
        )
        proprio_persistence = F.mse_loss(
            encoded["proprio"][:, self.dinowm.context - 1 : -1],
            encoded["proprio"][:, self.dinowm.context :],
        )
        metrics = {
            "pred/visual_mse": visual_loss.mean().detach(),
            "pred/proprio_mse": proprio_loss.mean().detach(),
            "pred/visual_persistence_mse": visual_persistence.detach(),
            "pred/proprio_persistence_mse": proprio_persistence.detach(),
            "pred/visual_skill_vs_persistence": _skill_vs_baseline(visual_loss.mean().detach(), visual_persistence),
            "pred/proprio_skill_vs_persistence": _skill_vs_baseline(proprio_loss.mean().detach(), proprio_persistence),
        }
        losses = {"pred": pred_loss}
        overshoot, rollout_metrics = self.dinowm.rollout_loss(
            state,
            encoded["tokens"],
            encoded["proprio"],
            data["action"],
        )
        if overshoot is not None:
            losses["overshoot"] = overshoot
        constraint_loss, constraint_metrics = self._constraint_loss(state, data)
        if constraint_loss is not None:
            losses["constraint"] = constraint_loss
        metrics.update(constraint_metrics)
        metrics.update(rollout_metrics)
        return state, losses, metrics

    def _constraint_loss(
        self, state: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        required = {
            "action",
            "ctrl_valid",
            "ctrl_clamp",
            "ctrl_retained_xyz",
            "ctrl_achieved_xyz",
        }
        if self.dinowm.constraint is None or not required.issubset(data):
            return None, {}

        count = state["tokens"].shape[1]
        if count < 2:
            return None, {}
        depart = {key: value[:, :-1] for key, value in state.items()}
        outputs = self.dinowm.constraint_outputs(depart, data["action"][:, -count + 1 :])
        assert outputs is not None
        valid = data["ctrl_valid"][:, -count + 1 :].float()
        clamp_target = data["ctrl_clamp"][:, -count + 1 :].float()
        retained_target = data["ctrl_retained_xyz"][:, -count + 1 :].float()
        achieved_target = data["ctrl_achieved_xyz"][:, -count + 1 :].float()
        valid_count = valid.sum().clamp_min(1.0)

        logits = outputs["clamp_logits"]
        positives = (clamp_target * valid).sum(dim=(0, 1))
        negatives = ((1.0 - clamp_target) * valid).sum(dim=(0, 1))
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        clamp_element = F.binary_cross_entropy_with_logits(
            logits, clamp_target, pos_weight=pos_weight, reduction="none"
        )
        clamp_loss = (clamp_element * valid).sum() / (valid_count * clamp_target.shape[-1])

        scale = self.dinowm.constraint_motion_scale
        retained_element = F.smooth_l1_loss(outputs["retained_xyz"] * scale, retained_target * scale, reduction="none")
        achieved_element = F.smooth_l1_loss(outputs["achieved_xyz"] * scale, achieved_target * scale, reduction="none")
        retained_loss = (retained_element * valid).sum() / (valid_count * 3.0)
        achieved_loss = (achieved_element * valid).sum() / (valid_count * 3.0)
        total = clamp_loss + retained_loss + achieved_loss

        probability = logits.sigmoid()
        predicted = probability >= 0.5
        target_bool = clamp_target >= 0.5
        mask = valid.bool().expand_as(target_bool)
        tp = (predicted & target_bool & mask).sum(dim=(0, 1)).float()
        fp = (predicted & ~target_bool & mask).sum(dim=(0, 1)).float()
        fn = (~predicted & target_bool & mask).sum(dim=(0, 1)).float()
        labels = ("workspace", "lag", "joint_limit")
        metrics: dict[str, torch.Tensor] = {
            "constraint/clamp_loss": clamp_loss.detach(),
            "constraint/retained_xyz_loss": retained_loss.detach(),
            "constraint/achieved_xyz_loss": achieved_loss.detach(),
            "constraint/valid_fraction": valid.mean().detach(),
            "constraint/retained_xyz_mae_m": ((outputs["retained_xyz"] - retained_target).abs() * valid).sum().detach()
            / (valid_count * 3.0),
            "constraint/achieved_xyz_mae_m": ((outputs["achieved_xyz"] - achieved_target).abs() * valid).sum().detach()
            / (valid_count * 3.0),
        }
        valid_flat = valid.squeeze(-1)
        for index, label in enumerate(labels):
            metrics[f"constraint/{label}_prob"] = (probability[..., index] * valid_flat).sum().detach() / valid_count
            metrics[f"constraint/{label}_rate"] = (clamp_target[..., index] * valid_flat).sum().detach() / valid_count
            metrics[f"constraint/{label}_precision"] = (tp[index] / (tp[index] + fp[index]).clamp_min(1.0)).detach()
            metrics[f"constraint/{label}_recall"] = (tp[index] / (tp[index] + fn[index]).clamp_min(1.0)).detach()
        return total, metrics
