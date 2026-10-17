from __future__ import annotations

import pytest
import torch

from ant.data.synthetic import generate_dataset
from ant.training.data import collate_decisions, decision_loader
from ant.training.losses import combined_loss, cross_entropy_loss, pairwise_margin_loss
from ant.training.trainer import Trainer, TrainingConfig


def test_collate_pads_ragged_rows(examples):
    batch = collate_decisions(examples)
    assert batch.mask.shape == (2, 4)
    assert batch.labels.tolist() == [0, 0]
    assert batch.mask[1, 3].item() is False


def test_loader_yields_batches(examples):
    from ant.training.data import DecisionDataset

    loader = decision_loader(DecisionDataset(examples), batch_size=1, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0].states == [examples[0].state]


def test_cross_entropy_matches_manual():
    scores = torch.tensor([[2.0, 0.0, -1.0]])
    labels = torch.tensor([0])
    loss = cross_entropy_loss(scores, labels, None)
    manual = -(torch.log_softmax(scores, dim=1)[0, 0])
    assert torch.allclose(loss, manual)


def test_pairwise_margin_zero_when_confident():
    scores = torch.tensor([[10.0, 0.0, -5.0]])
    labels = torch.tensor([0])
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert pairwise_margin_loss(scores, labels, mask, margin=1.0).item() == 0.0


def test_pairwise_margin_positive_when_violated():
    scores = torch.tensor([[0.0, 0.5]])
    labels = torch.tensor([0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = pairwise_margin_loss(scores, labels, mask, margin=1.0)
    assert loss.item() == pytest.approx(1.5)


def test_combined_loss_components():
    scores = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    output = combined_loss(scores, labels, mask, ce_weight=1.0, rank_weight=1.0, margin=0.5)
    assert set(output.components) == {"cross_entropy", "pairwise_margin"}
    assert torch.isfinite(output.total)


def test_trainer_runs_and_checkpoint_roundtrip(model, tmp_path):
    torch.manual_seed(0)
    splits = generate_dataset(n_train=24, n_val=8, n_test=8, n_ood=4)
    trainer = Trainer(
        model,
        TrainingConfig(
            mode="head_only", epochs=2, batch_size=8, learning_rate=5e-3,
            checkpoint_dir=str(tmp_path / "run"), log_every=0,
        ),
    )
    report = trainer.fit(splits.train, splits.val)
    assert len(report.history) == 2
    assert report.history[-1]["val_loss"] == report.history[-1]["val_loss"]  # finite
    assert (tmp_path / "run" / "ant_head.pt").exists()
    assert (tmp_path / "run" / "training_report.json").exists()

    from ant.model import ANTModel

    restored = ANTModel.load(tmp_path / "run", backbone=model.backbone)
    scores_a = model.score_candidates("s", "q", ["a", "b"])
    scores_b = restored.score_candidates("s", "q", ["a", "b"])
    assert torch.allclose(scores_a, scores_b, atol=1e-6)


def test_trainer_restores_the_best_epoch(model):
    """After fit, the in-memory head is the best-validation epoch, not the last one."""
    torch.manual_seed(0)
    splits = generate_dataset(n_train=24, n_val=8, n_test=8, n_ood=4)
    trainer = Trainer(
        model,
        TrainingConfig(mode="head_only", epochs=3, batch_size=8, learning_rate=5e-1, log_every=0),
    )
    report = trainer.fit(splits.train, splits.val)
    val_loss, _ = trainer.evaluate(splits.val)
    assert val_loss == pytest.approx(report.best_val_loss, abs=1e-5)
