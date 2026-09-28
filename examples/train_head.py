"""Minimal training example: frozen backbone + trainable head on the synthetic
benchmark, with temperature calibration and a risk-coverage escalation policy."""

from __future__ import annotations

import torch

from ant.calibration.calibrator import build_calibrator
from ant.data.synthetic import generate_dataset
from ant.evaluation.metrics import collect_predictions, evaluate_model, score_matrix
from ant.inference.escalation import RiskCoveragePolicy
from ant.model import ANTModel
from ant.training.trainer import Trainer, TrainingConfig


def main() -> int:
    model = ANTModel.tiny()  # swap with ANTModel(ANTConfig(backbone_model="Qwen/Qwen3.5-4B")) on a GPU box

    splits = generate_dataset(n_train=400, n_val=100, n_test=100)
    report = Trainer(
        model,
        TrainingConfig(
            mode="head_only",
            epochs=8,
            batch_size=8,
            learning_rate=1e-2,  # mock backbone; use ~2e-4 with Qwen
            checkpoint_dir="runs/example",
        ),
    ).fit(splits.train, splits.val)

    scores, labels, _ = score_matrix(model, splits.val)
    calibrator = build_calibrator("temperature")
    print("calibration:", calibrator.fit(torch.from_numpy(scores), torch.from_numpy(labels)))

    predictions = collect_predictions(model, splits.val, calibrator=calibrator)
    escalation = RiskCoveragePolicy(target_risk=0.10).fit(predictions.confidences, predictions.corrects)
    print("escalation:", escalation.stats)

    print(evaluate_model(model, splits.test, calibrator=calibrator, name="test").as_row())
    print(evaluate_model(model, splits.ood, calibrator=calibrator, name="ood").as_row())
    print(f"\nbest epoch {report.best_epoch} | val accuracy {report.best_val_accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
