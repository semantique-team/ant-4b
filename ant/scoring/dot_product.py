from __future__ import annotations

import torch

from ant.scoring.base import CandidateScorer


class DotProductScorer(CandidateScorer):
    """score_i = z_context · z_candidate_i. The v0.1 baseline."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()

    def forward(self, z_context: torch.Tensor, z_candidates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bnd->bn", z_context, z_candidates)
