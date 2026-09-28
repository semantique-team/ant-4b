from __future__ import annotations

import torch
from torch import nn

from ant.scoring.base import CandidateScorer


class BilinearScorer(CandidateScorer):
    """score = z_contextᵀ W z_candidate, W initialized near identity so training
    starts from the dot-product baseline."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))

    def forward(self, z_context: torch.Tensor, z_candidates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,de,bne->bn", z_context, self.weight, z_candidates)
