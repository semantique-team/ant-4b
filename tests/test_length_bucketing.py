from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ant.backbone.qwen import QwenBackbone
from ant.utils import bucket_max_length


def test_bucket_rounds_up_to_the_next_fixed_width():
    assert bucket_max_length([1]) == 64
    assert bucket_max_length([64]) == 64
    assert bucket_max_length([65]) == 128
    assert bucket_max_length([128, 37]) == 128
    assert bucket_max_length([129]) == 256
    assert bucket_max_length([1000]) == 1024


def test_bucket_never_truncates_beyond_the_largest_width():
    assert bucket_max_length([1025]) == 1025
    assert bucket_max_length([4096]) == 4096


def test_bucket_rejects_empty_or_non_positive_lengths():
    with pytest.raises(ValueError, match="at least one length"):
        bucket_max_length([])
    with pytest.raises(ValueError, match="positive"):
        bucket_max_length([0])


def test_bucket_accepts_a_custom_ladder():
    assert bucket_max_length([40], buckets=(16, 48, 64)) == 48


def _pad(input_ids: torch.Tensor, mask: torch.Tensor, cap: int, pad_token_id: int = 0):
    """Call the backbone helper without building a backbone (it only needs the tokenizer)."""
    stub = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=pad_token_id))
    return QwenBackbone._pad_to_bucket(stub, input_ids, mask, cap=cap)


def test_pad_to_bucket_widens_short_batches_and_extends_the_mask():
    input_ids = torch.arange(1, 38).unsqueeze(0)  # 37 tokens -> bucket 64
    mask = torch.ones_like(input_ids)
    padded_ids, padded_mask = _pad(input_ids, mask, cap=1024)
    assert padded_ids.shape == (1, 64)
    assert padded_mask.shape == (1, 64)
    assert torch.equal(padded_ids[:, :37], input_ids)
    assert padded_ids[0, 37:].eq(0).all()
    assert padded_mask[:, :37].eq(1).all()
    assert padded_mask[:, 37:].eq(0).all()


def test_pad_to_bucket_is_a_no_op_on_an_exact_bucket_or_above_the_cap():
    exact_ids = torch.arange(1, 65).unsqueeze(0)
    exact_mask = torch.ones_like(exact_ids)
    same_ids, same_mask = _pad(exact_ids, exact_mask, cap=1024)
    assert same_ids.shape == (1, 64)
    assert torch.equal(same_ids, exact_ids) and torch.equal(same_mask, exact_mask)

    # a 37-token batch whose cap is 40 cannot be widened to 64 without changing
    # the truncation length the caller asked for
    short_ids = torch.arange(1, 38).unsqueeze(0)
    short_mask = torch.ones_like(short_ids)
    capped_ids, _ = _pad(short_ids, short_mask, cap=40)
    assert capped_ids.shape == (1, 37)


def test_pad_to_bucket_uses_the_tokenizer_pad_id():
    input_ids = torch.arange(1, 38).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    padded_ids, _ = _pad(input_ids, mask, cap=1024, pad_token_id=151643)
    assert padded_ids[0, 37:].eq(151643).all()


def test_pad_to_bucket_skips_a_tokenizer_without_a_pad_token():
    input_ids = torch.arange(1, 38).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    padded_ids, _ = _pad(input_ids, mask, cap=1024, pad_token_id=None)
    assert padded_ids.shape == (1, 37)
