from __future__ import annotations

import numpy as np
import pytest
import torch

from ant.calibration.calibrator import (
    IdentityCalibrator,
    TemperatureCalibrator,
    VectorScalingCalibrator,
    build_calibrator,
)
from ant.evaluation.metrics import expected_calibration_error
from ant.utils import softmax_np


def _overconfident_data(n=400, temperature=0.1, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 4)) * 3.0
    labels = rng.integers(0, 4, size=n)
    logits[np.arange(n), labels] += 2.0
    scores = logits / temperature  # overconfident
    return torch.tensor(scores, dtype=torch.float32), torch.tensor(labels)


def _nll(scores, labels):
    probs = torch.softmax(scores, dim=1)
    return float(-torch.log(probs[torch.arange(len(labels)), labels]).mean())


def test_identity_is_noop():
    calibrator = IdentityCalibrator()
    scores = torch.randn(5, 3)
    assert torch.equal(calibrator(scores), scores)


def test_temperature_scaling_reduces_nll_on_overconfident_scores():
    scores, labels = _overconfident_data()
    calibrator = TemperatureCalibrator()
    before = _nll(scores, labels)
    info = calibrator.fit(scores, labels)
    after = _nll(calibrator(scores), labels)
    assert info["temperature"] > 1.0
    assert after < before


def test_temperature_scaling_improves_ece():
    scores, labels = _overconfident_data()
    probs_before = softmax_np(scores.numpy(), axis=1)
    correct_before = probs_before.argmax(1) == labels.numpy()
    conf_before = probs_before.max(1)

    calibrator = TemperatureCalibrator()
    calibrator.fit(scores, labels)
    probs_after = softmax_np(calibrator(scores).numpy(), axis=1)
    conf_after = probs_after.max(1)
    correct_after = probs_after.argmax(1) == labels.numpy()

    assert expected_calibration_error(conf_after, correct_after) < expected_calibration_error(
        conf_before, correct_before
    )


def test_vector_scaling_reduces_to_temperature():
    scores, labels = _overconfident_data(n=200)
    calibrator = VectorScalingCalibrator()
    info = calibrator.fit(scores, labels)
    before = _nll(scores, labels)
    after = _nll(calibrator(scores), labels)
    assert info["scale"] < 1.0  # overconfident scores need shrinking, i.e. T > 1
    assert abs(info["bias"]) < 1e-2  # a constant shift is softmax-invariant
    assert after < before


def test_mlp_calibrator_fits_fixed_width():
    scores, labels = _overconfident_data(n=200)
    calibrator = build_calibrator("mlp")
    before = _nll(scores, labels)
    calibrator.fit(scores, labels, steps=100)
    assert _nll(calibrator(scores), labels) < before + 0.5  # converges loosely


def test_build_calibrator_registry():
    for name in ("identity", "temperature", "vector_scaling", "mlp"):
        assert build_calibrator(name) is not None
    with pytest.raises(ValueError):
        build_calibrator("bogus")


def test_calibrator_state_roundtrip():
    scores, labels = _overconfident_data(n=100)
    calibrator = TemperatureCalibrator()
    calibrator.fit(scores, labels)
    restored = TemperatureCalibrator()
    restored.load_state_dict(calibrator.state_dict())
    assert torch.allclose(calibrator(scores), restored(scores))
