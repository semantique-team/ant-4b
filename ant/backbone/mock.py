from __future__ import annotations

import zlib
from typing import Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput, concat_outputs
from ant.utils import chunked


class MockBackbone(nn.Module):
    """Deterministic, differentiable stand-in for the Qwen backbone.

    Hashes whitespace tokens into a fixed vocabulary and mixes them with small
    per-token layers, mirroring the decoder structure (layers ModuleList + final
    norm) so that training modes like partial unfreezing behave the same way.
    Not a language model: it only reproduces the Backbone interface.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        vocab_size: int = 4096,
        max_tokens: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

    def _tokenize(self, text: str) -> list[int]:
        tokens = text.lower().split() or ["<empty>"]
        return [zlib.crc32(token.encode("utf-8")) % self.vocab_size for token in tokens][
            : self.max_tokens
        ]

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_length: int | None = None,
    ) -> BackboneOutput:
        limit = max_length or self.max_tokens
        device = self.embedding.weight.device
        outputs: list[BackboneOutput] = []
        for chunk in chunked(list(texts), batch_size):
            ids = [self._tokenize(t)[:limit] for t in chunk]
            width = max(len(seq) for seq in ids)
            batch_ids = torch.zeros(len(ids), width, dtype=torch.long)
            mask = torch.zeros(len(ids), width, dtype=torch.long)
            for i, seq in enumerate(ids):
                batch_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[i, : len(seq)] = 1
            hidden = self.embedding(batch_ids.to(device))
            for layer in self.layers:
                hidden = layer(hidden)
            hidden = self.norm(hidden)
            outputs.append(BackboneOutput(hidden_states=hidden, attention_mask=mask.to(device)))
        return concat_outputs(outputs)
