from __future__ import annotations

import numpy as np
import pytest

from ant.backbone.mlx import MlxBackbone
from ant.backbone.qwen import build_quantization_config
from ant.config import ANTConfig


def test_mlx_quantization_requires_mlx_backend():
    with pytest.raises(ValueError, match="backbone_backend"):
        ANTConfig(quantization="mlx")


def test_mlx_quantization_config_roundtrips():
    config = ANTConfig(
        backbone_backend="mlx",
        quantization="mlx",
        quantization_bits=4,
        quantization_group_size=32,
    )
    restored = ANTConfig.from_dict(config.to_dict())
    assert restored == config


def test_metal_quantization_is_not_accepted_by_mlx_backend():
    with pytest.raises(ValueError, match="mlx"):
        ANTConfig(backbone_backend="mlx", quantization="metal")


def test_qwen_quantization_builder_rejects_mlx():
    with pytest.raises(ValueError, match="ant.backbone.mlx"):
        build_quantization_config("mlx", 8, 64)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, truncation=True, max_length=None):
        return {"a": [1, 2, 3], "b": [4]}[text]


def test_mlx_token_batch_pads_to_bucket():
    backbone = object.__new__(MlxBackbone)
    backbone.tokenizer = FakeTokenizer()
    ids, mask = backbone._token_batch(["a", "b"], cap=1024)
    assert ids.shape == mask.shape == (2, 64)
    assert ids[1, 1:].tolist() == [0] * 63
    assert mask[0].sum() == 3
    assert mask[1].sum() == 1
    assert ids.dtype == np.int32


def test_mlx_quantized_checkpoint_mismatch_is_rejected():
    backbone = object.__new__(MlxBackbone)
    backbone.model_id = "fake"
    backbone.quantization_bits = 8
    backbone.quantization_group_size = 64
    with pytest.raises(RuntimeError, match="4-bit"):
        backbone._require_requested_quantization(
            {"quantization": {"bits": 4, "group_size": 64}}
        )
