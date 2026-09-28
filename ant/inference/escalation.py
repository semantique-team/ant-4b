from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

ACTION_EXECUTE = "execute"
ACTION_ESCALATE = "escalate"

# metrics where escalating happens when the value is BELOW the threshold
LOWER_IS_WORSE = {"confidence", "margin"}
# metrics where escalating happens when the value is ABOVE the threshold
HIGHER_IS_WORSE = {"entropy", "normalized_entropy"}


@dataclass(frozen=True)
class ConfidenceMetrics:
    n_candidates: int
    top1_probability: float
    top2_probability: float | None
    confidence: float
    margin: float | None
    entropy: float | None
    normalized_entropy: float | None
    choice_index: int


def compute_confidence_metrics(probabilities: Sequence[float]) -> ConfidenceMetrics:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0 or not np.isclose(probs.sum(), 1.0, atol=1e-5):
        raise ValueError("probabilities must be a non-empty distribution summing to 1")
    order = np.argsort(-probs, kind="stable")
    choice = int(order[0])
    top1 = float(probs[choice])
    top2 = float(probs[order[1]]) if probs.size > 1 else None
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    normalized = entropy / math.log(probs.size) if probs.size > 1 else 0.0
    return ConfidenceMetrics(
        n_candidates=int(probs.size),
        top1_probability=top1,
        top2_probability=top2,
        confidence=top1,
        margin=(top1 - top2) if top2 is not None else None,
        entropy=entropy,
        normalized_entropy=normalized,
        choice_index=choice,
    )


class EscalationPolicy(ABC):
    """Decides execute vs escalate from confidence metrics only.

    Thresholds are policy configuration, never architectural constants: derive
    them from validation data (see RiskCoveragePolicy.fit) or from an explicit
    risk appetite.
    """

    name = "policy"

    @abstractmethod
    def decide(self, metrics: ConfidenceMetrics) -> str:
        raise NotImplementedError

    def to_config(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_config(cls, data: dict) -> "EscalationPolicy":
        name = data.get("name", "always_execute")
        if name == "always_execute":
            return AlwaysExecute()
        if name == "threshold":
            return ThresholdPolicy(
                metric=data["metric"],
                threshold=data["threshold"],
            )
        if name == "risk_coverage":
            policy = RiskCoveragePolicy(
                metric=data.get("metric", "confidence"),
                target_coverage=data.get("target_coverage"),
                target_risk=data.get("target_risk"),
            )
            policy.threshold = data.get("threshold", RiskCoveragePolicy.DEFAULT_THRESHOLD)
            policy.stats = data.get("stats", {})
            return policy
        raise ValueError(f"unknown escalation policy {name!r}")


class AlwaysExecute(EscalationPolicy):
    """Default policy: expose the metrics and leave routing to the external
    System 1/System 2 router (ANT itself does not invoke System 2)."""

    name = "always_execute"

    def decide(self, metrics: ConfidenceMetrics) -> str:
        return ACTION_EXECUTE


class ThresholdPolicy(EscalationPolicy):
    name = "threshold"

    def __init__(self, metric: str = "confidence", threshold: float = 0.8) -> None:
        valid = LOWER_IS_WORSE | HIGHER_IS_WORSE
        if metric not in valid:
            raise ValueError(f"metric must be one of {sorted(valid)}, got {metric!r}")
        self.metric = metric
        self.threshold = float(threshold)

    def decide(self, metrics: ConfidenceMetrics) -> str:
        value = getattr(metrics, self.metric, None)
        if value is None:
            return ACTION_ESCALATE
        if self.metric in LOWER_IS_WORSE:
            return ACTION_EXECUTE if value >= self.threshold else ACTION_ESCALATE
        return ACTION_EXECUTE if value <= self.threshold else ACTION_ESCALATE

    def to_config(self) -> dict:
        return {"name": self.name, "metric": self.metric, "threshold": self.threshold}


class RiskCoveragePolicy(ThresholdPolicy):
    """Threshold derived from validation risk/coverage analysis.

    Fit on (confidence, correct) pairs from a held-out split, then choose the
    smallest threshold satisfying the constraints, which maximizes coverage
    (automation) within the stated risk budget.
    """

    name = "risk_coverage"
    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        metric: str = "confidence",
        *,
        target_coverage: float | None = None,
        target_risk: float | None = None,
    ) -> None:
        if target_coverage is None and target_risk is None:
            raise ValueError("provide target_coverage and/or target_risk")
        super().__init__(metric=metric, threshold=self.DEFAULT_THRESHOLD)
        self.target_coverage = target_coverage
        self.target_risk = target_risk
        self.stats: dict = {}

    def fit(self, values: Sequence[float], corrects: Sequence[bool]) -> "RiskCoveragePolicy":
        values = np.asarray(values, dtype=np.float64)
        corrects = np.asarray(corrects, dtype=bool)
        if values.shape[0] != corrects.shape[0] or values.size == 0:
            raise ValueError("values and corrects must be non-empty and aligned")
        # escalate items below threshold; evaluate on the retained fraction
        thresholds = np.unique(np.concatenate([[0.0], values, [np.nextafter(1.0, 2.0)]]))
        best = None
        for threshold in thresholds:
            keep = values >= threshold
            coverage = float(keep.mean())
            if coverage == 0.0:
                continue
            risk = float((~corrects[keep]).mean())
            accuracy = float(corrects[keep].mean())
            feasible = True
            if self.target_coverage is not None and coverage < self.target_coverage:
                feasible = False
            if self.target_risk is not None and risk > self.target_risk:
                feasible = False
            if feasible and (best is None or threshold < best[0]):
                best = (float(threshold), coverage, risk, accuracy)
        if best is None:
            raise ValueError(
                "no threshold satisfies the constraints; relax target_coverage/target_risk"
            )
        self.threshold, coverage, risk, accuracy = best
        self.stats = {
            "threshold": self.threshold,
            "coverage": coverage,
            "risk": risk,
            "automation_accuracy": accuracy,
            "n": int(values.size),
        }
        return self

    def to_config(self) -> dict:
        return {
            "name": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "target_coverage": self.target_coverage,
            "target_risk": self.target_risk,
            "stats": self.stats,
        }
