from __future__ import annotations

import pytest
import torch

from ant.backbone.mock import MockBackbone
from ant.config import ANTConfig
from ant.model import ANTModel
from ant.types import DecisionExample


@pytest.fixture(scope="session")
def backbone():
    torch.manual_seed(0)
    return MockBackbone(hidden_size=64, num_layers=2)


@pytest.fixture
def model(backbone):
    torch.manual_seed(0)
    return ANTModel(ANTConfig(backbone_model="mock", scoring_dim=32), backbone=backbone)


@pytest.fixture
def examples():
    return [
        DecisionExample(
            state="The customer was charged twice for the same invoice.",
            question="Which department should handle this case?",
            candidates=["Billing", "Technical Support", "Sales", "Legal"],
            correct="Billing",
        ),
        DecisionExample(
            state="The mobile app crashes on the reports page.",
            question="Who should take care of this request?",
            candidates=["Technical Support", "Billing", "Legal"],
            correct="Technical Support",
        ),
    ]
