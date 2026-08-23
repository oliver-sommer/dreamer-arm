import pytest
import torch

from dreamer_arm.core.optim.laprop import LaProp


def test_laprop_reduces_quadratic_loss() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = LaProp([parameter], lr=0.05)
    initial = parameter.square().item()
    for _ in range(10):
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        optimizer.step()
    assert parameter.square().item() < initial


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"lr": -1.0}, "learning rate"), ({"eps": -1.0}, "epsilon"), ({"betas": (1.0, 0.9)}, "beta1")],
)
def test_laprop_validates_hyperparameters(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LaProp([torch.nn.Parameter(torch.ones(1))], **kwargs)
