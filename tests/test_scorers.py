from __future__ import annotations

import pytest
import torch

from ant.scoring import SCORER_REGISTRY, build_scorer


@pytest.fixture(scope="module")
def embeddings():
    torch.manual_seed(1)
    z_context = torch.randn(3, 16)
    z_candidates = torch.randn(3, 5, 16)
    return z_context, z_candidates


@pytest.mark.parametrize("name", sorted(SCORER_REGISTRY))
def test_scorer_shapes(name, embeddings):
    z_context, z_candidates = embeddings
    scorer = build_scorer(name, 16)
    scores = scorer(z_context, z_candidates)
    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("name", sorted(SCORER_REGISTRY))
def test_scorer_permutation_invariance(name, embeddings):
    z_context, z_candidates = embeddings
    scorer = build_scorer(name, 16)
    scores = scorer(z_context, z_candidates)
    permutation = torch.tensor([4, 0, 3, 1, 2])
    permuted = scorer(z_context, z_candidates[:, permutation])
    assert torch.allclose(scores[:, permutation], permuted, atol=1e-6)


def test_dot_product_matches_einsum(embeddings):
    z_context, z_candidates = embeddings
    scores = build_scorer("dot", 16)(z_context, z_candidates)
    manual = (z_context.unsqueeze(1) * z_candidates).sum(-1)
    assert torch.allclose(scores, manual)


def test_bilinear_starts_near_dot_product(embeddings):
    z_context, z_candidates = embeddings
    bilinear = build_scorer("bilinear", 16)(z_context, z_candidates)
    dot = build_scorer("dot", 16)(z_context, z_candidates)
    assert torch.allclose(bilinear, dot, atol=0.5)


def test_cosine_bounded_and_scaled(embeddings):
    z_context, z_candidates = embeddings
    scorer = build_scorer("cosine", 16)
    scores = scorer(z_context, z_candidates)
    reference = torch.cosine_similarity(
        z_context.unsqueeze(1).expand_as(z_candidates), z_candidates, dim=-1
    )
    assert torch.allclose(scores / scorer.scale, reference, atol=1e-5)
    assert (scores.abs() <= scorer.scale + 1e-5).all()
