from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, float]


def cross_entropy_loss(
    scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """L = -log softmax(s)_correct. Padded slots are already at MASK_SCORE, so
    they contribute ~0 probability."""
    return F.cross_entropy(scores.float(), labels)


def pairwise_margin_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge on score(correct) > score(incorrect) + margin, averaged over the
    incorrect candidates."""
    correct = scores.gather(1, labels.unsqueeze(1))
    violations = (margin - (correct - scores)).clamp_min(0)
    is_valid = mask & (torch.arange(scores.size(1), device=scores.device).unsqueeze(0)
                       != labels.unsqueeze(1))
    denominator = is_valid.sum().clamp_min(1)
    return (violations * is_valid.to(scores.dtype)).sum() / denominator


def combined_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    ce_weight: float = 1.0,
    rank_weight: float = 0.0,
    margin: float = 1.0,
) -> LossOutput:
    losses: dict[str, torch.Tensor] = {}
    if ce_weight > 0:
        losses["cross_entropy"] = cross_entropy_loss(scores, labels, mask)
    if rank_weight > 0:
        losses["pairwise_margin"] = pairwise_margin_loss(scores, labels, mask, margin)
    if not losses:
        raise ValueError("all loss weights are zero")
    total = sum(weight * losses[name] for name, weight in
                (("cross_entropy", ce_weight), ("pairwise_margin", rank_weight))
                if name in losses)
    return LossOutput(
        total=total,
        components={name: float(value.detach()) for name, value in losses.items()},
    )
