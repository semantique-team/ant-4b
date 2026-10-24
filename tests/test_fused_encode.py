from __future__ import annotations

import pytest
import torch

STATE = "The customer was charged twice for the same invoice."
QUESTION = "Which department should handle this case?"
CANDIDATES = ["Billing", "Technical Support", "Sales", "Legal"]


def test_returns_context_and_candidate_embeddings(model):
    z_context, z_candidates = model.encode_state_and_candidates(STATE, QUESTION, CANDIDATES)
    assert z_context.shape == (1, model.config.scoring_dim)
    assert z_candidates.shape == (len(CANDIDATES), model.config.scoring_dim)


def test_matches_the_two_pass_encoding(model):
    with torch.no_grad():
        z_fused_context, z_fused_candidates = model.encode_state_and_candidates(
            STATE, QUESTION, CANDIDATES
        )
        z_context = model.encode_contexts([STATE], [QUESTION])
        z_candidates = model.encode_candidates(CANDIDATES)
    assert torch.allclose(z_fused_context, z_context, atol=1e-5)
    assert torch.allclose(z_fused_candidates, z_candidates, atol=1e-5)


def test_empty_candidate_list_is_rejected(model):
    with pytest.raises(ValueError):
        model.encode_state_and_candidates(STATE, QUESTION, [])


def test_uses_a_single_backbone_encode_call(model, monkeypatch):
    calls = {"count": 0}
    original = model.backbone.encode

    def counting_encode(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backbone, "encode", counting_encode)
    with torch.no_grad():
        model.encode_state_and_candidates(STATE, QUESTION, CANDIDATES)
    assert calls["count"] == 1


@pytest.mark.parametrize("order", [[3, 0, 2, 1], [1, 2, 3, 0]])
def test_candidate_order_is_preserved(model, order):
    ordered = [CANDIDATES[index] for index in order]
    with torch.no_grad():
        _, z_fused = model.encode_state_and_candidates(STATE, QUESTION, ordered)
        for row, candidate in zip(z_fused, ordered):
            single = model.encode_candidates([candidate])[0]
            assert torch.allclose(row, single, atol=1e-5)
