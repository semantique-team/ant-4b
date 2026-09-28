"""Integration tests against a real Qwen3.5 checkpoint.

Skipped unless ANT_TEST_QWEN points to a model id or a local download:
    ANT_TEST_QWEN=models/Qwen3.5-4B .venv/bin/pytest tests/test_qwen_integration.py -m integration
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("ANT_TEST_QWEN"),
        reason="set ANT_TEST_QWEN to a model id or local path to run integration tests",
    ),
]


@pytest.fixture(scope="module")
def qwen_backbone():
    from ant.backbone.qwen import QwenBackbone

    backbone = QwenBackbone(os.environ["ANT_TEST_QWEN"], torch_dtype="auto", device="auto")
    assert backbone.hidden_size == 2560, "expected the official Qwen3.5-4B text config"
    return backbone


def test_hidden_states(qwen_backbone):
    output = qwen_backbone.encode(["a short text", "another text", "x"])
    assert output.hidden_states.shape == (3, output.hidden_states.shape[1], 2560)
    assert torch.isfinite(output.hidden_states.float()).all()


def test_model_forward_and_invariance(qwen_backbone):
    from ant.config import ANTConfig
    from ant.model import ANTModel
    from ant.utils import softmax_np

    model = ANTModel(ANTConfig(backbone_model="mock", scoring_dim=64), backbone=qwen_backbone)
    candidates = ["Billing", "Technical Support", "Sales", "Legal"]
    with torch.no_grad():
        scores = model.score_candidates(
            "Customer charged twice for the same invoice.",
            "Which department should handle this case?",
            candidates,
        )
        shuffled = list(reversed(candidates))
        permuted = model.score_candidates(
            "Customer charged twice for the same invoice.",
            "Which department should handle this case?",
            shuffled,
        )
    probs = softmax_np(scores.numpy())
    mapped = softmax_np(permuted.numpy())[::-1]
    assert abs((probs - mapped).max()) < 1e-4
