from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F


@dataclass
class BackboneOutput:
    """Last hidden states of a text batch plus the padding mask.

    hidden_states: [N, L, H], attention_mask: [N, L] with 1 for real tokens.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


def concat_outputs(outputs: list[BackboneOutput]) -> BackboneOutput:
    """Concatenate batched outputs, right-padding shorter chunks to the widest."""
    if len(outputs) == 1:
        return outputs[0]
    width = max(output.hidden_states.size(1) for output in outputs)

    def pad_hidden(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, 0, 0, width - tensor.size(1)))

    def pad_mask(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, width - tensor.size(1)))

    return BackboneOutput(
        hidden_states=torch.cat([pad_hidden(o.hidden_states) for o in outputs], dim=0),
        attention_mask=torch.cat([pad_mask(o.attention_mask) for o in outputs], dim=0),
    )


class Backbone(Protocol):
    """Anything that can turn strings into per-token hidden states.

    Gradients flow through ``encode`` only when the backbone parameters are
    unfrozen (head-only training keeps them frozen, LoRA/partial/full do not).
    """

    hidden_size: int
    num_layers: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> BackboneOutput: ...
