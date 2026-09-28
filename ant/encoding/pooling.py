from __future__ import annotations

import torch
from torch import nn


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).clamp_min(1)
    index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, index).squeeze(1)


def max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filler = torch.finfo(hidden.dtype).min
    masked = hidden.masked_fill(mask.unsqueeze(-1) == 0, filler)
    return masked.max(dim=1).values


class MeanPooling(nn.Module):
    def __init__(self, hidden_size: int | None = None) -> None:
        super().__init__()

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(hidden, mask)


class LastTokenPooling(nn.Module):
    def __init__(self, hidden_size: int | None = None) -> None:
        super().__init__()

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return last_token_pool(hidden, mask)


class MaxPooling(nn.Module):
    def __init__(self, hidden_size: int | None = None) -> None:
        super().__init__()

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return max_pool(hidden, mask)


class AttentionPooling(nn.Module):
    """Single learned query attending over real tokens."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size) / hidden_size**0.5)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("h,nlh->nl", self.query, hidden) / hidden.size(-1) ** 0.5
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


POOLING_REGISTRY: dict[str, type[nn.Module]] = {
    "mean": MeanPooling,
    "last_token": LastTokenPooling,
    "max": MaxPooling,
    "attention": AttentionPooling,
}


def build_pooling(name: str, hidden_size: int | None = None) -> nn.Module:
    try:
        cls = POOLING_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown pooling {name!r}; available: {sorted(POOLING_REGISTRY)}") from None
    if name == "attention":
        if not hidden_size:
            raise ValueError("attention pooling requires hidden_size")
        return cls(hidden_size)
    return cls()
