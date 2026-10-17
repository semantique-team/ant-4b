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


@dataclass
class ANTConfig:
    """Full configuration of an ANT model. Backbone dimensions are read from the
    checkpoint at load time and never hard-coded here."""

    backbone_model: str = DEFAULT_BACKBONE
    torch_dtype: str = "auto"  # auto|bfloat16|float16|float32
    device: str = "auto"  # auto|cuda|mps|cpu|index
    attn_implementation: str | None = None

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
