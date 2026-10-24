from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BACKBONE = "Qwen/Qwen3.5-4B"
DEFAULT_BASE_BACKBONE = "Qwen/Qwen3.5-4B-Base"

_POOLING_STRATEGIES = ("mean", "last_token", "max", "attention")
_SCORERS = ("dot", "cosine", "bilinear", "mlp")
_QUANTIZATIONS = (None, "metal", "mlx")
_QUANTIZATION_BITS = (2, 4, 8)
_BACKENDS = ("torch", "mlx")


@dataclass
class ANTConfig:
    """Full configuration of an ANT model. Backbone dimensions are read from the
    checkpoint at load time and never hard-coded here."""

    backbone_model: str = DEFAULT_BACKBONE
    backbone_backend: str = "torch"  # torch|mlx
    torch_dtype: str = "auto"  # auto|bfloat16|float16|float32
    device: str = "auto"  # auto|cuda|mps|cpu|index
    attn_implementation: str | None = None

    # Backbone weight quantization. "metal" uses the native transformers Metal
    # quantizer on MPS, which fuses dequantization into the matmul so the frozen
    # 4B backbone stays resident in unified memory instead of spilling to swap.
    quantization: str | None = None  # None or "metal"
    quantization_bits: int = 8  # 2, 4 or 8
    quantization_group_size: int = 64  # positive
    # Chunk width of the pure-PyTorch GatedDeltaNet fallback. None keeps the
    # transformers default (64); larger chunks cut Python loop overhead.
    linear_attention_chunk_size: int | None = None  # None = transformers default (64)

    max_context_tokens: int = 1024
    max_candidate_tokens: int = 128
    encode_batch_size: int = 8

    scoring_dim: int = 512
    scorer: str = "dot"
    context_pooling: str = "mean"
    candidate_pooling: str = "mean"
    dropout: float = 0.0
    context_template: str | None = None
    candidate_template: str | None = None

    temperature: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.scorer not in _SCORERS:
            raise ValueError(f"scorer must be one of {_SCORERS}, got {self.scorer!r}")
        if self.context_pooling not in _POOLING_STRATEGIES:
            raise ValueError(f"context_pooling must be one of {_POOLING_STRATEGIES}")
        if self.candidate_pooling not in _POOLING_STRATEGIES:
            raise ValueError(f"candidate_pooling must be one of {_POOLING_STRATEGIES}")
        if self.scoring_dim <= 0:
            raise ValueError("scoring_dim must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.backbone_backend not in _BACKENDS:
            raise ValueError(
                f"backbone_backend must be one of {_BACKENDS}, got {self.backbone_backend!r}"
            )
        if self.quantization not in _QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_QUANTIZATIONS}, got {self.quantization!r}"
            )
        if self.backbone_backend == "mlx" and self.quantization == "metal":
            raise ValueError("quantization='metal' belongs to the torch backend; use 'mlx'")
        if self.backbone_backend == "torch" and self.quantization == "mlx":
            raise ValueError("quantization='mlx' requires backbone_backend='mlx'")
        if self.quantization in ("metal", "mlx"):
            if self.quantization_bits not in _QUANTIZATION_BITS:
                raise ValueError(
                    f"quantization_bits must be one of {_QUANTIZATION_BITS} when "
                    f"quantization={self.quantization!r}, got {self.quantization_bits!r}"
                )
            if self.quantization_group_size <= 0:
                raise ValueError("quantization_group_size must be positive")
        if self.linear_attention_chunk_size is not None and (
            not isinstance(self.linear_attention_chunk_size, int)
            or isinstance(self.linear_attention_chunk_size, bool)
            or self.linear_attention_chunk_size <= 0
        ):
            raise ValueError("linear_attention_chunk_size must be a positive int or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ANTConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(data).items() if k in valid})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ANTConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
