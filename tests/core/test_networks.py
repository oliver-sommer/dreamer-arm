"""Shape-only tests for the conv encoder / decoder.

These tests catch the common failure mode where conv strides or paddings
get tweaked and break round-trip shape compatibility between the
:class:`ConvEncoder` and :class:`ConvDecoder`.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dreamer_arm.core.networks import ConvEncoder


def test_conv_encoder_shape() -> None:
    cfg = SimpleNamespace(
        act="SiLU",
        norm=True,
        kernel_size=5,
        minres=4,
        depth=4,
        mults=[1, 1, 1, 1],
    )
    enc = ConvEncoder(cfg, input_shape=(64, 64, 3))
    x = torch.randn(2, 64, 64, 3)
    y = enc(x)
    assert y.ndim == 2
    assert y.shape[0] == 2
    assert y.shape[1] == enc.out_dim


def test_multi_decoder_accepts_readonly_config() -> None:
    """``dispatch()`` marks the run config read-only before the entrypoint runs,
    so the decoder must not write its derived ``shape`` into the shared node --
    that made ``core/model=dreamerv3`` fail at construction.
    """
    from omegaconf import OmegaConf

    from dreamer_arm.core.networks import MultiDecoder

    cfg = OmegaConf.create(
        {
            "mlp_keys": "proprio",
            "cnn_keys": "^$",  # no image branch: keeps the test cheap
            "mlp_dist": {"name": "symlog_mse"},
            "cnn_dist": {"name": "mse"},
            "mlp": {
                "shape": None,
                "layers": 1,
                "units": 8,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "identity"},
                "outscale": 1.0,
                "symlog_inputs": False,
                "name": "mlp_decoder",
            },
        }
    )
    OmegaConf.set_readonly(cfg, True)

    MultiDecoder(cfg, deter=16, flat_stoch=8, shapes={"proprio": (4,)})

    assert cfg.mlp.shape is None, "decoder mutated the shared config"
