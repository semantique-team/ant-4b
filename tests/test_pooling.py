from __future__ import annotations

import torch

from ant.encoding.pooling import build_pooling, masked_mean


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


def test_masked_mean_batch():
    hidden = torch.ones(2, 4, 3)
    hidden[1] *= 2
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    pooled = masked_mean(hidden, mask)
    assert torch.allclose(pooled[0], torch.ones(3))
    assert torch.allclose(pooled[1], torch.full((3,), 2.0))


def test_last_token_pool_picks_last_real_token():
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.tensor([[1, 1, 0]])  # right padding: last real token is index 1
    pooled = build_pooling("last_token")(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0]]))


def test_attention_pooling_is_learned_and_masked():
    torch.manual_seed(0)
    pooling = build_pooling("attention", hidden_size=8)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    pooled = pooling(hidden, mask)
    assert pooled.shape == (2, 8)
    perturbed = hidden.clone()
    perturbed[1, 1:] = 1000.0  # padding positions must not matter
    assert torch.allclose(pooling(perturbed, mask)[1], pooled[1])


def test_unknown_pooling_raises():
    import pytest

    with pytest.raises(ValueError):
        build_pooling("bogus")
