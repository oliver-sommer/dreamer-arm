"""Shape + gradient tests for `ActorCritic`'s imagination and replay-value losses.

Uses a minimal stand-in world model (state is a single `feat` tensor, `img_step`
perturbs it slightly) rather than a real RSSM/DINO-WM -- `ActorCritic` only
ever calls `get_feat`/`img_step` on whatever `WorldModel` it's given, so this
is enough to exercise the loss formula without pulling in a full agent.
"""

from __future__ import annotations

import torch

from dreamer_arm.core.actor_critic import ActorCritic, sanitize_action


class _TinyWorldModel:
    """`WorldModel`-shaped stand-in: state = {"feat": (..., feat_size)}."""

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"feat": state["feat"] + 0.01 * torch.randn_like(state["feat"])}

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["feat"]


def test_actor_critic_loss_shapes_and_gradient(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    feat_size, act_dim, b, t = 8, 3, 2, 5
    ac = ActorCritic(
        tiny_actor_critic_cfg, feat_size, (act_dim,), act_discrete=False, imag_starts=None, device=torch.device("cpu")
    )
    wm = _TinyWorldModel()

    feat = torch.randn(b, t, feat_size, requires_grad=True)
    state = {"feat": feat.detach()}
    data = {
        "action": torch.randn(b, t, act_dim),
        "reward": torch.randn(b, t, 1),
        "is_terminal": torch.zeros(b, t),
        "is_last": torch.zeros(b, t),
    }

    losses, metrics = ac.loss(feat, data, wm, state)

    for key in ("rew", "con", "policy", "value", "repval"):
        assert key in losses
        assert losses[key].shape == ()
        assert torch.isfinite(losses[key])
    assert "ret" in metrics and "action_mean" in metrics

    sum(losses.values()).backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert any(p.grad is not None for p in ac.value.parameters())
    assert any(p.grad is not None for p in ac.actor.parameters())


def test_imag_starts_none_keeps_every_start(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    ac = ActorCritic(tiny_actor_critic_cfg, 4, (2,), act_discrete=False, imag_starts=None, device=torch.device("cpu"))
    state = {"feat": torch.randn(2, 5, 4)}

    torch.manual_seed(0)
    out = ac._gather_imag_starts(state)
    after = torch.rand(1)

    assert torch.equal(out["feat"], state["feat"].reshape(10, 4))
    torch.manual_seed(0)
    assert torch.equal(after, torch.rand(1)), "imag_starts=None must not consume RNG"


def test_imag_starts_subsamples_to_requested_count(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    ac = ActorCritic(tiny_actor_critic_cfg, 4, (2,), act_discrete=False, imag_starts=3, device=torch.device("cpu"))
    sub = ac._gather_imag_starts({"feat": torch.randn(2, 5, 4)})
    assert sub["feat"].shape == (3, 4)


def test_imag_starts_pair_indexing_matches_flattening(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    """(b, t) indexing must pick the same starts a flatten-then-index would.

    The state can be a strided view (DINO-WM's sliding windows), so the pair is
    indexed instead of reshaped -- that must not change *which* starts are used.
    """
    ac = ActorCritic(tiny_actor_critic_cfg, 4, (2,), act_discrete=False, imag_starts=3, device=torch.device("cpu"))
    b, t = 2, 5
    feat = torch.arange(b * t * 4, dtype=torch.float32).reshape(b, t, 4)

    torch.manual_seed(0)
    got = ac._gather_imag_starts({"feat": feat})["feat"]
    torch.manual_seed(0)
    expected = feat.reshape(-1, 4)[torch.randperm(b * t)[:3]]
    assert torch.equal(got, expected)


def test_imag_starts_accepts_a_non_contiguous_view(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    """DINO-WM hands over `unfold` views; gathering must not require contiguity."""
    ac = ActorCritic(tiny_actor_critic_cfg, 4, (2,), act_discrete=False, imag_starts=3, device=torch.device("cpu"))
    windows = torch.randn(2, 8, 4).unfold(1, 3, 1).permute(0, 1, 3, 2)  # (2, 6, 3, 4), a view
    assert not windows.is_contiguous()

    assert ac._gather_imag_starts({"feat": windows})["feat"].shape == (3, 3, 4)


def test_refresh_frozen_picks_up_new_weights(tiny_actor_critic_cfg) -> None:  # type: ignore[no-untyped-def]
    ac = ActorCritic(tiny_actor_critic_cfg, 4, (2,), act_discrete=False, imag_starts=None, device=torch.device("cpu"))
    with torch.no_grad():
        for p in ac.value.parameters():
            p.fill_(0.123)
    ac.refresh_frozen()
    assert all(torch.equal(p, torch.full_like(p, 0.123)) for p in ac._frozen_value.parameters())


def test_sanitize_action_scrubs_nan_and_clamps_continuous() -> None:
    x = torch.tensor([float("nan"), 5.0, -5.0, 0.3])
    out = sanitize_action(x, discrete=False)
    assert torch.isfinite(out).all()
    assert out.max() <= 1.0
    assert out.min() >= -1.0


def test_sanitize_action_passes_discrete_through_unchanged() -> None:
    d = torch.tensor([0.0, 1.0, 0.0])
    assert torch.equal(sanitize_action(d, discrete=True), d)
