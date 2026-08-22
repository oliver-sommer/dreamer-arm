"""Shape + gradient tests for DINO-WM.

Uses `pretrained=False` so no network access or weight download is needed --
the architecture (patch size, embed dim, register tokens) is still the real
one timm registers locally, just with random weights.
"""

from __future__ import annotations

import torch

from dreamer_arm.core.world_model.dinowm import DinoEncoder, DinoWM, generate_mask_matrix


def _shapes() -> dict[str, tuple[int, ...]]:
    return {"scene": (32, 32, 3), "proprio": (5,)}


def test_dino_encoder_shapes_and_frozen(tiny_dino_encoder_cfg) -> None:  # type: ignore[no-untyped-def]
    enc = DinoEncoder(tiny_dino_encoder_cfg)
    assert enc.patch_size == 16
    assert enc.num_patches == 4  # 32 / 16 = 2 -> 2*2
    assert enc.embed_dim == 384
    assert enc.num_prefix_tokens == 5  # 1 CLS + 4 register tokens

    for p in enc.parameters():
        assert not p.requires_grad

    image = torch.rand(2, 3, 32, 32, 3)  # (B, T, H, W, C), float [0, 1]
    tokens = enc(image)
    assert tokens.shape == (2, 3, 4, 384)
    assert not tokens.requires_grad  # frozen forward runs under no_grad


def test_generate_mask_matrix_is_block_causal() -> None:
    mask = generate_mask_matrix(num_frames=2, tokens_per_frame=3)
    assert mask.shape == (6, 6)
    # Frame 0 tokens (0-2) may only see frame 0.
    assert mask[0].tolist() == [True, True, True, False, False, False]
    # Frame 1 tokens (3-5) may see both frames.
    assert mask[3].tolist() == [True, True, True, True, True, True]


def test_dinowm_loss_shapes_and_gradient(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    b, t = 2, 6
    tokens = torch.randn(b, t, 4, dinowm.tok_dim)
    action = torch.randn(b, t, 4)

    state, pred_loss = dinowm.loss(tokens, action)
    num_windows = t - dinowm.context
    assert pred_loss.shape == (b, num_windows)
    assert state["tokens"].shape == (b, num_windows, dinowm.context, 4, dinowm.tok_dim)
    assert state["actions_out"].shape == (b, num_windows, dinowm.context - 1, dinowm.action_dim_embed)

    pred_loss.mean().backward()
    grads = [p.grad for p in dinowm.predictor.parameters()]
    assert grads, "predictor has no parameters"
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_dinowm_img_step_gradient_flows_to_action(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    n = 2
    state = dinowm.initial(n, torch.device("cpu"))
    action = torch.randn(n, 4, requires_grad=True)

    next_state = dinowm.img_step(state, action)
    assert next_state["tokens"].shape == state["tokens"].shape
    assert next_state["actions_out"].shape == state["actions_out"].shape

    feat = dinowm.get_feat(next_state)
    assert feat.shape == (n, dinowm.feat_size)
    feat.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()


def test_feat_pool_flatten_matches_expected_size(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    tiny_dinowm_cfg.feat_pool = "flatten"
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    assert dinowm.feat_size == dinowm.tok_dim * 4
    state = dinowm.initial(2, torch.device("cpu"))
    feat = dinowm.get_feat(state)
    assert feat.shape == (2, dinowm.feat_size)
