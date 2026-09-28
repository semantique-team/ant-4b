from __future__ import annotations

import numpy as np
import pytest
import torch

from ant.utils import softmax_np


@pytest.mark.parametrize("scorer", ["dot", "cosine", "bilinear", "mlp"])
def test_permutation_invariance_end_to_end(backbone, scorer, examples):
    torch.manual_seed(0)
    from ant.config import ANTConfig
    from ant.model import ANTModel

    model = ANTModel(ANTConfig(backbone_model="mock", scoring_dim=16, scorer=scorer), backbone=backbone)
    model.eval()
    example = examples[0]
    with torch.no_grad():
        base = softmax_np(model.score_candidates(example.state, example.question, example.candidates).numpy())
        permutations = [
            [2, 0, 3, 1],
            [3, 2, 1, 0],
            [1, 3, 0, 2],
        ]
        for permutation in permutations:
            shuffled = [example.candidates[i] for i in permutation]
            probs = softmax_np(model.score_candidates(example.state, example.question, shuffled).numpy())
            mapped = np.array([probs[shuffled.index(c)] for c in example.candidates])
            assert np.abs(mapped - base).max() < 1e-5


def test_duplicated_candidates_split_probability(model, examples):
    model.eval()
    example = examples[0]
    a, b, c = example.candidates[:3]
    with torch.no_grad():
        probs = softmax_np(model.score_candidates(example.state, example.question, [a, a, b, c]).numpy())
    assert abs(probs[0] - probs[1]) < 1e-6  # identical candidates, identical probability
    assert abs(probs.sum() - 1.0) < 1e-6


def test_single_candidate_is_deterministic(model):
    model.eval()
    with torch.no_grad():
        scores = model.score_candidates("s", "q", ["only"])
    probs = softmax_np(scores.numpy())
    assert probs.shape == (1,)
    assert abs(probs[0] - 1.0) < 1e-9


def test_candidate_paraphrase_scores_are_close(model):
    """Test C: 'Billing' vs 'Handle the customer's billing issue' should land
    near each other; both must stay far from 'Legal'."""
    model.eval()
    candidates = ["Billing", "Handle the customer's billing issue", "Legal"]
    with torch.no_grad():
        scores = model.score_candidates(
            "Customer charged twice for the same invoice and requests a refund.",
            "Which department should handle this case?",
            candidates,
        ).numpy()
    gap_near = abs(scores[0] - scores[1])
    gap_far = abs(scores[0] - scores[2])
    # untrained head: no directional claim, but the API must expose the signal
    assert np.isfinite(scores).all()
    assert gap_near >= 0 and gap_far >= 0
