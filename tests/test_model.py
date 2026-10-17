from __future__ import annotations

import pytest
import torch

from ant.config import ANTConfig
from ant.model import ANTModel, MASK_SCORE


def test_forward_shapes_and_mask(model, examples):
    scores, mask = model.forward(
        [e.state for e in examples], [e.question for e in examples], [list(e.candidates) for e in examples]
    )
    assert scores.shape == (2, 4)
    assert mask.dtype == torch.bool
    assert mask[1, 3].item() is False  # second row has 3 candidates
    assert scores[1, 3].item() <= MASK_SCORE / 2


def test_forward_variable_candidate_count(model):
    state, question = "s", "q"
    rows = [["a", "b"], ["a", "b", "c", "d", "e"], ["z"]]
    scores, mask = model.forward([state] * 3, [question] * 3, rows)
    assert scores.shape == (3, 5)
    assert mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
        [True, False, False, False, False],
    ]


def test_score_candidates_preserves_order(model):
    candidates = ["Sales", "Billing", "Legal"]
    scores = model.score_candidates("state", "question", candidates)
    assert scores.shape == (3,)
    reversed_scores = model.score_candidates("state", "question", list(reversed(candidates)))
    assert torch.allclose(scores.flip(0), reversed_scores, atol=1e-6)


def test_loss_finite_and_differentiable(model, examples):
    loss = model.loss(examples)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.context_encoder.projection.weight.grad is not None


def test_head_state_dict_excludes_backbone(model):
    state = model.head_state_dict()
    assert state
    assert all("backbone" not in key for key in state)


def test_save_load_roundtrip(model, examples, tmp_path):
    path = model.save_head(tmp_path / "ckpt")
    restored = ANTModel.load(path, backbone=model.backbone)
    scores_original = model.score_candidates(examples[0].state, examples[0].question, examples[0].candidates)
    scores_restored = restored.score_candidates(
        examples[0].state, examples[0].question, examples[0].candidates
    )
    assert torch.allclose(scores_original, scores_restored, atol=1e-6)


def test_all_scorers_run_through_model(backbone):
    for scorer in ("dot", "cosine", "bilinear", "mlp"):
        torch.manual_seed(0)
        config = ANTConfig(backbone_model="mock", scoring_dim=16, scorer=scorer)
        variant = ANTModel(config, backbone=backbone)
        scores = variant.score_candidates("a state", "a question", ["x", "y", "z"])
        assert scores.shape == (3,)


def test_load_ignores_the_device_saved_by_the_training_machine(tmp_path):
    """A checkpoint trained with device='cuda' must load on a machine without CUDA."""
    from dataclasses import replace

    from ant.model import ANTModel

    model = ANTModel.tiny()
    model.config = replace(model.config, device="cuda")
    model.save_head(tmp_path / "run")
    assert ANTModel.load(tmp_path / "run").config.device == "auto"
