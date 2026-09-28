from __future__ import annotations

from torch import nn


class CandidateScorer(nn.Module):
    """Scores a context embedding against per-candidate embeddings.

    Input: z_context [B, d] and z_candidates [B, N, d]. Output: scores [B, N].
    Implementations must depend only on (z_context, z_candidate_i) pairs, never
    on candidate position or on the rest of the set: that is what guarantees
    permutation invariance and a variable number of candidates.
    """

    def forward(self, z_context, z_candidates):
        raise NotImplementedError
