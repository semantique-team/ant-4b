from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ant.scoring.base import CandidateScorer


class CosineScorer(CandidateScorer):
    """Cosine similarity rescaled by a learnable positive factor: raw cosine is
    confined to [-1, 1], which is too narrow a logit range for a useful softmax."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(2.77))  # e^2.77 ≈ 16

    @property
    def scale(self) -> float:
        return float(torch.exp(self.log_scale.detach()))

    def forward(self, z_context: torch.Tensor, z_candidates: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(z_context.unsqueeze(1), z_candidates, dim=-1)
        return similarity * torch.exp(self.log_scale)
