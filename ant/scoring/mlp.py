from __future__ import annotations

import torch
from torch import nn

from ant.scoring.base import CandidateScorer


class MLPInteractionScorer(CandidateScorer):
    """Interaction MLP over [z_context, z_candidate, z_context * z_candidate,
    |z_context - z_candidate|]."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(4 * dim, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, 1),
        )

    def forward(self, z_context: torch.Tensor, z_candidates: torch.Tensor) -> torch.Tensor:
        batch, n_candidates, _ = z_candidates.shape
        expanded = z_context.unsqueeze(1).expand(batch, n_candidates, -1)
        features = torch.cat(
            [expanded, z_candidates, expanded * z_candidates, (expanded - z_candidates).abs()],
            dim=-1,
        )
        return self.net(features).squeeze(-1)
