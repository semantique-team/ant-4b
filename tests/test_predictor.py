from __future__ import annotations

import numpy as np
import pytest
import torch

from ant.inference.cache import CandidateEmbeddingCache
from ant.inference.escalation import ThresholdPolicy
from ant.inference.predictor import ANTPredictor
from ant.calibration.calibrator import TemperatureCalibrator


@pytest.fixture
def ant(model):
    return ANTPredictor(model)


def test_predict_structure(ant, examples):
    example = examples[0]
    result = ant.predict(example.state, example.question, example.candidates)
    data = result.to_dict()
    assert data["choice"] in example.candidates
    assert set(data["probabilities"]) == set(example.candidates)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["action"] in ("execute", "escalate")
    assert data["n_candidates"] == 4
    assert data["scores"] is not None


def test_predict_with_threshold_policy(model, examples):
    ant = ANTPredictor(model, escalation=ThresholdPolicy(metric="confidence", threshold=1.1))
    result = ant.predict(examples[0].state, examples[0].question, examples[0].candidates)
    assert result.action == "escalate"


def test_predict_with_calibrator(model, examples):
    ant = ANTPredictor(model, calibrator=TemperatureCalibrator(), temperature=2.0)
    result = ant.predict(examples[0].state, examples[0].question, examples[0].candidates)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5


def test_predict_batch_matches_single(ant, examples):
    states = [e.state for e in examples]
    questions = [e.question for e in examples]
    shared = list(examples[0].candidates)
    batch = ant.predict_batch(states, questions, shared)
    singles = [ant.predict(s, q, shared) for s, q in zip(states, questions)]
    for single, item in zip(singles, batch):
        assert single.choice == item.choice
        for candidate, probability in single.probabilities.items():
            assert abs(item.probabilities[candidate] - probability) < 1e-5


def test_predict_batch_ragged(model, examples):
    ant = ANTPredictor(model)
    rows = [list(examples[0].candidates), list(examples[1].candidates)]
    results = ant.predict_batch(
        [e.state for e in examples], [e.question for e in examples], rows
    )
    assert len(results) == 2
    assert set(results[1].probabilities) == set(rows[1])


def test_candidate_cache_avoids_reencoding(model, examples):
    cache = CandidateEmbeddingCache()
    ant = ANTPredictor(model, candidate_cache=cache)
    example = examples[0]
    ant.predict(example.state, example.question, example.candidates)
    assert len(cache) == 4

    calls = {"n": 0}
    original = model.encode_candidates

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.encode_candidates = counting  # type: ignore[method-assign]
    try:
        ant.predict(example.state, example.question, example.candidates)
    finally:
        model.encode_candidates = original  # type: ignore[method-assign]
    assert calls["n"] == 0  # fully cached

    ant.predict(example.state, example.question, list(example.candidates) + ["Security"])
    assert len(cache) == 5


def test_score_candidates_low_level(ant, examples):
    scores = ant.score_candidates(examples[0].state, examples[0].question, examples[0].candidates)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (4,)


def test_save_load_roundtrip(model, examples, tmp_path):
    ant = ANTPredictor(
        model,
        calibrator=TemperatureCalibrator(),
        escalation=ThresholdPolicy(metric="margin", threshold=0.25),
    )
    ant.save(tmp_path / "ckpt")
    restored = ANTPredictor.load(tmp_path / "ckpt", backbone=model.backbone)
    assert isinstance(restored.escalation, ThresholdPolicy)
    assert restored.escalation.threshold == 0.25
    assert isinstance(restored.calibrator, TemperatureCalibrator)
    a = ant.predict(examples[0].state, examples[0].question, examples[0].candidates)
    b = restored.predict(examples[0].state, examples[0].question, examples[0].candidates)
    assert a.choice == b.choice
    for candidate, probability in a.probabilities.items():
        assert abs(b.probabilities[candidate] - probability) < 1e-5
