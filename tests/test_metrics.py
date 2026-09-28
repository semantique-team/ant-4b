from __future__ import annotations

import numpy as np
import pytest

from ant.data.synthetic import generate_dataset
from ant.evaluation.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)
from ant.evaluation.metrics import (
    collect_predictions,
    evaluate_model,
    prediction_metrics,
    risk_coverage,
    score_matrix,
)


def test_ece_perfect_calibration():
    confidences = np.repeat(0.7, 100)
    corrects = np.random.default_rng(0).random(100) < 0.7
    # with the exact 70% accuracy the ECE is ~sampling noise only
    corrects = np.zeros(100, dtype=bool)
    corrects[:70] = True
    assert expected_calibration_error(confidences, corrects) < 0.05


def test_ece_detects_overconfidence():
    confidences = np.ones(100)
    corrects = np.zeros(100, dtype=bool)
    corrects[:50] = True
    assert expected_calibration_error(confidences, corrects) == pytest.approx(0.5, abs=0.01)
    assert maximum_calibration_error(confidences, corrects) >= expected_calibration_error(
        confidences, corrects
    )


def test_reliability_curve_bins():
    curve = reliability_curve([0.05, 0.95, 0.96], [False, True, True], n_bins=10)
    assert len(curve) == 2
    assert curve[0].count == 1 and curve[1].count == 2
    assert curve[1].accuracy == 1.0


def test_prediction_metrics_values():
    from ant.evaluation.metrics import PredictionSet

    predictions = PredictionSet(
        probabilities=[np.array([0.8, 0.2]), np.array([0.3, 0.7])],
        labels=[0, 0],
        corrects=[True, False],
        confidences=[0.8, 0.7],
        candidates=[["a", "b"], ["a", "b"]],
        states=["s", "s"],
        questions=["q", "q"],
    )
    metrics = prediction_metrics(predictions)
    assert metrics["n"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] == pytest.approx(-(np.log(0.8) + np.log(0.3)) / 2, rel=1e-6)
    assert metrics["brier"] == pytest.approx(
        ((0.8 - 1) ** 2 + 0.2**2 + (0.3 - 1) ** 2 + 0.7**2) / 2, rel=1e-6
    )


def test_risk_coverage_monotone_accuracy():
    confidences = [0.9, 0.8, 0.7, 0.6]
    corrects = [True, True, True, False]
    result = risk_coverage(confidences, corrects)
    assert result.coverage[0] == 0.25
    assert result.accuracy[0] == 1.0
    assert result.aurc >= 0
    assert result.accuracy_at(1.0) == 0.75


def test_end_to_end_evaluation(model):
    splits = generate_dataset(n_train=10, n_val=6, n_test=6, n_ood=4)
    scores, labels, mask = score_matrix(model, splits.val)
    assert scores.shape[0] == 6 and labels.shape == (6,)
    assert (scores[~mask] <= -1e8).all()
    assert (np.exp(scores[mask]) > 0).all()

    report = evaluate_model(model, splits.val, name="val")
    assert 0 <= report.accuracy <= 1
    assert report.n > 0

    predictions = collect_predictions(model, splits.val)
    assert len(predictions.probabilities) == 6
    for probs in predictions.probabilities:
        assert abs(probs.sum() - 1.0) < 1e-5
