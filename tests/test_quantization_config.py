from __future__ import annotations

import pytest
import torch

from ant.backbone.qwen import apply_linear_attention_chunk_size, build_quantization_config
from ant.config import ANTConfig


def test_config_defaults_are_unquantized():
    config = ANTConfig()
    assert config.quantization is None
    assert config.quantization_bits == 8
    assert config.quantization_group_size == 64
    assert config.linear_attention_chunk_size is None


def test_quantized_config_roundtrips_through_dict():
    config = ANTConfig(
        quantization="metal",
        quantization_bits=4,
        quantization_group_size=32,
        linear_attention_chunk_size=2048,
    )
    restored = ANTConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.quantization == "metal"
    assert restored.quantization_bits == 4
    assert restored.quantization_group_size == 32
    assert restored.linear_attention_chunk_size == 2048


def test_metal_with_eight_bits_is_accepted():
    config = ANTConfig(quantization="metal", quantization_bits=8)
    assert config.quantization == "metal"
    assert config.quantization_bits == 8


def test_three_bit_quantization_is_rejected():
    with pytest.raises(ValueError):
        ANTConfig(quantization="metal", quantization_bits=3)


def test_cuda_quantization_is_rejected():
    with pytest.raises(ValueError):
        ANTConfig(quantization="cuda")


def test_zero_group_size_is_rejected():
    with pytest.raises(ValueError):
        ANTConfig(quantization="metal", quantization_group_size=0)


def test_zero_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        ANTConfig(linear_attention_chunk_size=0)


def test_build_quantization_config_none_is_none():
    assert build_quantization_config(None, 8, 64) is None


def test_build_quantization_config_metal_reports_missing_runtime(monkeypatch):
    """A missing runtime must fail with an actionable message, not a traceback."""
    import transformers.utils as transformers_utils

    monkeypatch.setattr(transformers_utils, "is_kernels_available", lambda: False)
    with pytest.raises(ImportError, match="kernels"):
        build_quantization_config("metal", 8, 64)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        build_quantization_config("metal", 8, 64)


def test_build_quantization_config_metal_builds_when_runtime_is_present(monkeypatch):
    import transformers.utils as transformers_utils

    monkeypatch.setattr(transformers_utils, "is_kernels_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    config = build_quantization_config("metal", 8, 128)
    assert (config.bits, config.group_size, config.quant_method) == (8, 128, "metal")


def test_build_quantization_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="allowed values"):
        build_quantization_config("cuda", 8, 64)


def test_apply_linear_attention_chunk_size_none_is_harmless():
    # restoring the default must not raise, and must be safe to repeat
    apply_linear_attention_chunk_size(None)
    apply_linear_attention_chunk_size(None)
