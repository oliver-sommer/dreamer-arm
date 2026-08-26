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
    proprio = torch.randn(b, t, dinowm.proprio_dim)
    action = torch.randn(b, t, 4)

    state, visual_loss, proprio_loss = dinowm.loss(tokens, proprio, action)
    num_windows = t - dinowm.context
    assert visual_loss.shape == (b, num_windows)
    assert proprio_loss.shape == (b, num_windows)
    assert state["tokens"].shape == (b, num_windows, dinowm.context, 4, dinowm.tok_dim)
    assert state["proprio"].shape == (b, num_windows, dinowm.context, dinowm.proprio_dim)
    assert state["actions_out"].shape == (b, num_windows, dinowm.context - 1, dinowm.action_dim_embed)

    (visual_loss.mean() + proprio_loss.mean()).backward()
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
    assert next_state["proprio"].shape == state["proprio"].shape
    assert next_state["actions_out"].shape == state["actions_out"].shape

    feat = dinowm.get_feat(next_state)
    assert feat.shape == (n, dinowm.feat_size)
    feat.sum().backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()


def test_feat_pool_flatten_matches_expected_size(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    tiny_dinowm_cfg.feat_pool = "flatten"
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    assert dinowm.feat_size == dinowm.tok_dim * 4 + dinowm.proprio_dim
    state = dinowm.initial(2, torch.device("cpu"))
    feat = dinowm.get_feat(state)
    assert feat.shape == (2, dinowm.feat_size)


def test_task_attention_pool_is_task_conditioned(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    tiny_dinowm_cfg.mlp_keys = "proprio|task_id"
    shapes = {"scene": (32, 32, 3), "proprio": (5,), "task_id": (2,)}
    dinowm = DinoWM(tiny_dinowm_cfg, shapes, act_dim=4, num_patches=4, embed_dim=384)
    state = dinowm.initial(2, torch.device("cpu"))
    # Hold visual state and proprioception exactly fixed: only task identity
    # may account for a feature difference.
    tokens = torch.randn(1, dinowm.context, 4, dinowm.tok_dim)
    state["tokens"] = tokens.expand(2, -1, -1, -1).clone()
    state["task_id"] = torch.eye(2)

    feat = dinowm.get_feat(state)

    assert feat.shape == (2, dinowm.feat_size)
    assert not torch.allclose(feat[0, : dinowm.tok_dim], feat[1, : dinowm.tok_dim])
    assert torch.equal(feat[:, -2:], torch.eye(2))


def test_task_attention_pool_gradients_reach_query(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    state = dinowm.initial(2, torch.device("cpu"))
    state["tokens"] = torch.randn_like(state["tokens"])
    state["proprio"] = torch.randn_like(state["proprio"])

    dinowm.get_feat(state).square().mean().backward()

    assert dinowm.task_pool is not None
    assert dinowm.task_pool.query.weight.grad is not None
    assert torch.isfinite(dinowm.task_pool.query.weight.grad).all()


def test_dinowm_loss_is_chunk_invariant(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    """Batching windows into the predictor must not change what is computed.

    `window_chunk` only trades activation memory for GPU occupancy, so
    chunk=1 (one window per call, the original loop) and a chunk covering
    every window must agree on the loss, the imagination states, and the
    gradients.
    """
    torch.manual_seed(0)
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    b, t = 2, 8
    tokens = torch.randn(b, t, 4, dinowm.tok_dim)
    proprio = torch.randn(b, t, dinowm.proprio_dim)
    action = torch.randn(b, t, 4)

    results = []
    for chunk in (1, 64):
        dinowm.zero_grad(set_to_none=True)
        dinowm.window_chunk = chunk
        state, visual_loss, proprio_loss = dinowm.loss(tokens, proprio, action)
        pred_loss = visual_loss + proprio_loss
        pred_loss.mean().backward()
        grad = torch.cat([p.grad.reshape(-1) for p in dinowm.predictor.parameters() if p.grad is not None])
        results.append((pred_loss.detach(), state["tokens"], state["proprio"], state["actions_out"], grad.clone()))

    (loss_a, tok_a, prop_a, act_a, grad_a), (loss_b, tok_b, prop_b, act_b, grad_b) = results
    assert torch.allclose(loss_a, loss_b, atol=1e-6), (loss_a, loss_b)
    assert torch.equal(tok_a, tok_b)
    assert torch.equal(prop_a, prop_b)
    assert torch.equal(act_a, act_b)
    assert torch.allclose(grad_a, grad_b, atol=1e-5)


def test_task_id_is_exact_and_persistent_through_imagination(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    """Task context must not be reconstructed by the learned dynamics."""
    tiny_dinowm_cfg.mlp_keys = "proprio|task_id"
    shapes = {"scene": (32, 32, 3), "proprio": (5,), "task_id": (3,)}
    dinowm = DinoWM(tiny_dinowm_cfg, shapes, act_dim=4, num_patches=4, embed_dim=384)
    b, t = 2, 6
    task_id = torch.zeros(b, t, 3)
    task_id[0, :, 0] = 1.0
    task_id[1, :, 2] = 1.0
    tokens = torch.randn(b, t, 4, dinowm.tok_dim)
    proprio = torch.randn(b, t, dinowm.proprio_dim)

    state, _, _ = dinowm.loss(tokens, proprio, torch.randn(b, t, 4), task_id)
    expected = task_id[:, dinowm.context :]
    assert torch.equal(state["task_id"], expected)

    start = {key: value[:, 0] for key, value in state.items()}
    next_state = dinowm.img_step(start, torch.randn(b, 4))
    assert torch.equal(next_state["task_id"], start["task_id"])
    # Even with identical predicted tokens, exact task context reaches every
    # actor/reward/value feature unchanged.
    same_tokens = next_state["tokens"][:1].expand(b, *next_state["tokens"].shape[1:])
    conditioned = {**next_state, "tokens": same_tokens}
    feat = dinowm.get_feat(conditioned)
    assert torch.equal(feat[:, -3:], start["task_id"])
    assert not torch.equal(feat[0], feat[1])


def test_proprio_is_explicit_in_actor_feature(tiny_dinowm_cfg) -> None:  # type: ignore[no-untyped-def]
    """Robot state cannot collapse behind a learned auxiliary embedder."""
    dinowm = DinoWM(tiny_dinowm_cfg, _shapes(), act_dim=4, num_patches=4, embed_dim=384)
    state = dinowm.initial(2, torch.device("cpu"))
    state["tokens"][1] = state["tokens"][0]
    state["proprio"][1, -1] = 1.0

    feat = dinowm.get_feat(state)
    assert torch.equal(feat[0, : dinowm.tok_dim], feat[1, : dinowm.tok_dim])
    assert torch.equal(feat[:, -dinowm.proprio_dim :], state["proprio"][:, -1])
    assert not torch.equal(feat[0], feat[1])
