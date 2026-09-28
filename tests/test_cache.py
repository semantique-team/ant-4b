from __future__ import annotations

import pytest
import torch

from ant.inference.cache import CandidateEmbeddingCache


def test_cache_encodes_once(model):
    cache = CandidateEmbeddingCache()
    first = cache.embeddings(model, ["a", "b", "c"])
    assert first.shape == (3, model.config.scoring_dim)

    calls = {"n": 0}
    original = model.encode_candidates

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.encode_candidates = counting  # type: ignore[method-assign]
    try:
        second = cache.embeddings(model, ["c", "a"])
        cache.embeddings(model, ["a", "a", "b"])
    finally:
        model.encode_candidates = original  # type: ignore[method-assign]
    assert calls["n"] == 0
    assert torch.allclose(first[0], second[1])
    assert torch.allclose(first[2], second[0])


def test_cache_preserves_input_order_with_duplicates(model):
    cache = CandidateEmbeddingCache()
    embeddings = cache.embeddings(model, ["x", "y", "x"])
    reference = cache.embeddings(model, ["x", "y"])
    assert torch.allclose(embeddings[0], embeddings[2])
    assert torch.allclose(embeddings[0], reference[0])
    assert torch.allclose(embeddings[1], reference[1])


def test_cache_eviction_respects_request(model):
    with pytest.raises(ValueError):
        CandidateEmbeddingCache(max_size=2).embeddings(model, ["a", "b", "c"])
    small = CandidateEmbeddingCache(max_size=3)
    small.embeddings(model, ["a", "b", "c"])
    small.embeddings(model, ["d"])  # evicts oldest non-requested
    assert len(small) == 3
    assert small.contains("d")
    assert not small.contains("a")


def test_cache_save_load(model, tmp_path):
    cache = CandidateEmbeddingCache()
    embeddings = cache.embeddings(model, ["a", "b"])
    cache.save(tmp_path / "cache.pt")
    fresh = CandidateEmbeddingCache()
    fresh.load(tmp_path / "cache.pt")
    restored = fresh.embeddings(model, ["a", "b"])
    assert torch.allclose(embeddings, restored, atol=1e-6)
