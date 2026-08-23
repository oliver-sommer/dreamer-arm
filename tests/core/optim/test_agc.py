import torch

from dreamer_arm.core.optim.agc import adaptive_grad_clip


def test_adaptive_grad_clip_bounds_gradient_norm() -> None:
    parameter = torch.tensor([3.0, 4.0], requires_grad=True)
    parameter.grad = torch.tensor([60.0, 80.0])
    adaptive_grad_clip(parameter, clip=0.1, pmin=1e-3)
    assert torch.linalg.norm(parameter.grad).item() <= 0.5 + 1e-6


def test_adaptive_grad_clip_skips_missing_gradient() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    adaptive_grad_clip([parameter], clip=0.1, pmin=1e-3)
    assert parameter.grad is None
