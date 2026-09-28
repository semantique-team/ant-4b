from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


def expected_calibration_error(
    confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 15
) -> float:
    """ECE over top-1 confidence: |accuracy - confidence| weighted by bin mass."""
    confidences = np.asarray(confidences, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = confidences.size
    if total == 0:
        return 0.0
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper) if lower > 0 else (
            (confidences >= lower) & (confidences <= upper)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(float(corrects[in_bin].mean()) - float(confidences[in_bin].mean()))
        ece += (count / total) * gap
    return float(ece)


def maximum_calibration_error(
    confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 15
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper) if lower > 0 else (
            (confidences >= lower) & (confidences <= upper)
        )
        if in_bin.sum() == 0:
            continue
        worst = max(worst, abs(float(corrects[in_bin].mean()) - float(confidences[in_bin].mean())))
    return float(worst)


def reliability_curve(
    confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 15
) -> list[ReliabilityBin]:
    confidences = np.asarray(confidences, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper) if lower > 0 else (
            (confidences >= lower) & (confidences <= upper)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                avg_confidence=float(confidences[in_bin].mean()),
                accuracy=float(corrects[in_bin].mean()),
            )
        )
    return bins


def plot_reliability(
    confidences: Sequence[float], corrects: Sequence[bool], path: str, n_bins: int = 15
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    bins = reliability_curve(confidences, corrects, n_bins)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    axis.plot(
        [b.avg_confidence for b in bins],
        [b.accuracy for b in bins],
        "o-",
        label="model",
    )
    axis.set_xlabel("top-1 confidence")
    axis.set_ylabel("empirical accuracy")
    axis.set_title("Reliability diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True
