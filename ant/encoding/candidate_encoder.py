from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput
from ant.encoding.pooling import build_pooling
from ant.encoding.templates import DEFAULT_CANDIDATE_TEMPLATE


class CandidateEncoder(nn.Module):
    """Encodes each candidate independently into the shared scoring space.

    Candidates are semantic objects: short labels, sentences, or serialized
    structures are all valid inputs. No positional or set-level information is
    injected, which is what makes ANT invariant to candidate ordering.
    """

    def __init__(
        self,
        backbone,
        *,
        pooling: str = "mean",
        proj_dim: int = 512,
        dropout: float = 0.0,
        template: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = build_pooling(pooling, backbone.hidden_size)
        self.input_norm = nn.LayerNorm(backbone.hidden_size)
        self.projection = nn.Linear(backbone.hidden_size, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.template = template or DEFAULT_CANDIDATE_TEMPLATE

    def format(self, candidate: str) -> str:
        return self.template.format(candidate=candidate.strip())

    def pool_hidden(self, output: BackboneOutput) -> torch.Tensor:
        """Mask-aware pooling over the backbone output, in float32 (as the context
        encoder does) so the head stays fp32 under a fp16/bf16 backbone."""
        hidden = output.hidden_states.to(torch.float32)
        return self.pooling(hidden, output.attention_mask)

    def forward(
        self,
        candidates: Sequence[str],
        *,
        batch_size: int = 16,
        max_length: int | None = None,
    ) -> torch.Tensor:
        texts = [self.format(candidate) for candidate in candidates]
        output = self.backbone.encode(texts, batch_size=batch_size, max_length=max_length)
        pooled = self.pool_hidden(output)
        # the head stays fp32 under a fp16/bf16 backbone
        return self.dropout(self.projection(self.input_norm(pooled.float())))


__all__ = ["CandidateEncoder"]
