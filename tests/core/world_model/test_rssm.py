"""Shape + gradient tests for the RSSM.

We don't validate the *learning* dynamics here — just that the tensor
plumbing of ``observe`` / ``img_step`` matches what the agent assumes.
"""

from __future__ import annotations

import torch

from dreamer_arm.core.world_model.rssm import RSSM


def test_rssm_observe_shapes(tiny_rssm_cfg) -> None:  # type: ignore[no-untyped-def]
    embed_dim, act_dim = 16, 3
    rssm = RSSM(tiny_rssm_cfg, embed_dim, act_dim)
    b, t = 2, 4
    embed = torch.randn(b, t, embed_dim)
    action = torch.randn(b, t, act_dim)
    is_first = torch.zeros(b, t, dtype=torch.bool)
    is_first[:, 0] = True

    stoch0, deter0 = rssm.initial(b)
    stochs, deters, _logits = rssm.observe(embed, action, (stoch0, deter0), is_first)

    # posterior should have (B, T, stoch, discrete) / (B, T, deter).
    assert stochs.shape[0:2] == (b, t)
    assert deters.shape[0:2] == (b, t)


def test_rssm_img_step_gradient_flows(tiny_rssm_cfg) -> None:  # type: ignore[no-untyped-def]
    embed_dim, act_dim = 16, 3
    rssm = RSSM(tiny_rssm_cfg, embed_dim, act_dim)
    b = 2
    stoch, deter = rssm.initial(b)
    stoch = stoch.clone().requires_grad_(True)
    deter = deter.clone().requires_grad_(True)
    action = torch.randn(b, act_dim, requires_grad=True)
    out = rssm.img_step(stoch, deter, action)
    # Sum to scalar to check backward works end-to-end.
    loss = out[0].sum() + out[1].sum()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
