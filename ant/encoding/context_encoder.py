from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput
from ant.encoding.pooling import build_pooling
from ant.encoding.templates import DEFAULT_CONTEXT_TEMPLATE


class ContextEncoder(nn.Module):
    """Encodes (state, question) into a context embedding, independent of the
    candidate set. The backbone is shared by reference, not owned."""

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
        self.template = template or DEFAULT_CONTEXT_TEMPLATE

    def format(self, state: str, question: str) -> str:
        return self.template.format(state=state.strip(), question=question.strip())

    def pool_hidden(self, output: BackboneOutput) -> torch.Tensor:
        hidden = output.hidden_states.to(torch.float32)
        return self.pooling(hidden, output.attention_mask)

    def forward(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if len(states) != len(questions):
            raise ValueError("states and questions must have the same length")
        texts = [self.format(state, question) for state, question in zip(states, questions)]
        output = self.backbone.encode(texts, batch_size=batch_size, max_length=max_length)
        pooled = self.pool_hidden(output)
        # the head stays fp32 under a fp16/bf16 backbone
        return self.dropout(self.projection(self.input_norm(pooled.float())))
