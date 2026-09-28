from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _prepare(scores: Sequence[torch.Tensor] | torch.Tensor, labels: Sequence[int] | torch.Tensor):
    scores = torch.as_tensor(scores, dtype=torch.float32).detach()
    labels = torch.as_tensor(labels, dtype=torch.long)
    if scores.dim() != 2 or labels.shape[0] != scores.shape[0]:
        raise ValueError(f"expected scores [M, N] and labels [M], got {scores.shape} / {labels.shape}")
    if scores.shape[1] < 2:
        raise ValueError("calibration needs at least two candidates per example")
    return scores, labels


class IdentityCalibrator(nn.Module):
    name = "identity"

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return scores

    def fit(self, scores, labels) -> dict:
        scores, labels = _prepare(scores, labels)
        return {"calibrator": self.name, "n": int(labels.numel())}


class TemperatureCalibrator(nn.Module):
    """Single scalar T fitted on a held-out split by NLL minimization. The only
    post-hoc method that is well-defined when the candidate set varies."""

    name = "temperature"

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).detach())

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return scores / torch.exp(self.log_temperature).to(scores.dtype)

    def fit(self, scores, labels, max_iter: int = 100) -> dict:
        scores, labels = _prepare(scores, labels)
        before = F.cross_entropy(scores, labels).item()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(scores / torch.exp(self.log_temperature), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.log_temperature.requires_grad_(False)
        after = F.cross_entropy(self.forward(scores), labels).item()
        return {
            "calibrator": self.name,
            "n": int(labels.numel()),
            "temperature": self.temperature,
            "nll_before": before,
            "nll_after": after,
        }


class VectorScalingCalibrator(nn.Module):
    """Per-logit affine transform s' = a·s + b. With a variable candidate set a
    constant bias cancels in the softmax, so only the scale is identifiable;
    kept for completeness as the softmax-invariant form of vector scaling."""

    name = "vector_scaling"

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_scale).to(scores.dtype) * scores + self.bias.to(scores.dtype)

    def fit(self, scores, labels, max_iter: int = 100) -> dict:
        scores, labels = _prepare(scores, labels)
        before = F.cross_entropy(scores, labels).item()
        optimizer = torch.optim.LBFGS([self.log_scale, self.bias], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(scores), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        for p in (self.log_scale, self.bias):
            p.requires_grad_(False)
        after = F.cross_entropy(self.forward(scores), labels).item()
        return {
            "calibrator": self.name,
            "n": int(labels.numel()),
            "scale": float(torch.exp(self.log_scale)),
            "bias": float(self.bias),
            "nll_before": before,
            "nll_after": after,
        }


class MLPCalibrator(nn.Module):
    """Learned monotone-in-practice head g(s) applied per candidate. Shared
    across candidates, so permutation invariance and variable N are preserved;
    monotonicity is not enforced, so validate that it beats temperature."""

    name = "mlp"

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        flat = scores.reshape(-1, 1)
        return self.net(flat).reshape(scores.shape)

    def fit(self, scores, labels, steps: int = 500, lr: float = 1e-2) -> dict:
        scores, labels = _prepare(scores, labels)
        before = F.cross_entropy(scores, labels).item()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(scores), labels)
            loss.backward()
            optimizer.step()
        for p in self.parameters():
            p.requires_grad_(False)
        after = F.cross_entropy(self.forward(scores), labels).item()
        return {
            "calibrator": self.name,
            "n": int(labels.numel()),
            "nll_before": before,
            "nll_after": after,
        }


def build_calibrator(name: str) -> nn.Module:
    from ant.calibration import CALIBRATOR_REGISTRY

    try:
        cls = CALIBRATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown calibrator {name!r}; available: {sorted(CALIBRATOR_REGISTRY)}"
        ) from None
    return cls()
