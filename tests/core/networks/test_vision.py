from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from dreamer_arm.core.networks import ConvEncoder, MultiDecoder


def test_conv_encoder_shape() -> None:
    cfg = SimpleNamespace(act="SiLU", norm=True, kernel_size=5, minres=4, depth=4, mults=[1, 1, 1, 1])
    encoder = ConvEncoder(cfg, input_shape=(64, 64, 3))
    output = encoder(torch.randn(2, 64, 64, 3))
    assert output.shape == (2, encoder.out_dim)


def test_multi_decoder_accepts_readonly_config() -> None:
    cfg = OmegaConf.create(
        {
            "mlp_keys": "proprio",
            "cnn_keys": "^$",
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
    assert cfg.mlp.shape is None
