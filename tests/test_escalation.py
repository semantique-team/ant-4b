from __future__ import annotations

import math

import pytest

from ant.inference.escalation import (
    AlwaysExecute,
    ConfidenceMetrics,
    RiskCoveragePolicy,
    ThresholdPolicy,
    compute_confidence_metrics,
)


def _metrics(probs):
    return compute_confidence_metrics(probs)


def test_confidence_metrics_uniform():
    metrics = _metrics([0.25, 0.25, 0.25, 0.25])
    assert abs(metrics.confidence - 0.25) < 1e-9
    assert abs(metrics.entropy - math.log(4)) < 1e-9
    assert abs(metrics.normalized_entropy - 1.0) < 1e-9
    assert metrics.margin is not None and abs(metrics.margin) < 1e-9


def test_confidence_metrics_one_hot():
    metrics = _metrics([0.91, 0.06, 0.03])
    assert metrics.choice_index == 0
    assert abs(metrics.confidence - 0.91) < 1e-9
    assert abs(metrics.margin - 0.85) < 1e-9
    assert abs(metrics.normalized_entropy) < 0.5


def test_confidence_metrics_rejects_invalid():
    with pytest.raises(ValueError):
        _metrics([0.5, 0.4])


def test_always_execute():
    policy = AlwaysExecute()
    assert policy.decide(_metrics([0.3, 0.3, 0.4])) == "execute"


def test_threshold_policy_confidence():
    policy = ThresholdPolicy(metric="confidence", threshold=0.61)
    assert policy.decide(_metrics([0.61, 0.39])) == "execute"
    assert policy.decide(_metrics([0.60, 0.40])) == "escalate"


def test_threshold_policy_entropy():
    policy = ThresholdPolicy(metric="normalized_entropy", threshold=0.5)
    assert policy.decide(_metrics([0.9, 0.05, 0.05])) == "execute"
    assert policy.decide(_metrics([0.4, 0.3, 0.3])) == "escalate"


def test_risk_coverage_policy_fit():
    confidences = [0.95, 0.9, 0.85, 0.7, 0.6, 0.4]
    corrects = [True, True, True, True, False, False]
    policy = RiskCoveragePolicy(target_risk=0.25).fit(confidences, corrects)
    kept = [c for c in confidences if c >= policy.threshold]
    assert kept
    stats = policy.stats
    assert stats["risk"] <= 0.25
    assert stats["coverage"] > 0


def test_risk_coverage_policy_requires_target():
    with pytest.raises(ValueError):
        RiskCoveragePolicy()
    with pytest.raises(ValueError):
        RiskCoveragePolicy().fit([0.5], [True])


def test_policy_serialization_roundtrip():
    from ant.inference.escalation import EscalationPolicy

    policy = ThresholdPolicy(metric="margin", threshold=0.3)
    again = EscalationPolicy.from_config(policy.to_config())
    assert isinstance(again, ThresholdPolicy)
    assert again.threshold == 0.3 and again.metric == "margin"

    fitted = RiskCoveragePolicy(target_risk=0.2).fit([0.9, 0.5], [True, False])
    clone = EscalationPolicy.from_config(fitted.to_config())
    assert isinstance(clone, RiskCoveragePolicy)
    assert clone.threshold == fitted.threshold
