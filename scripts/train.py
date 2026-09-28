"""Train an ANT head (mode 1 by default: frozen Qwen + trainable head).

CPU end-to-end run:
    python scripts/train.py --backbone mock --out runs/mock-dot

Real backbone:
    python scripts/train.py --backbone Qwen/Qwen3.5-4B --device cuda --out runs/qwen-dot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ant.calibration.calibrator import build_calibrator
from ant.data.synthetic import generate_dataset
from ant.evaluation.metrics import (
    collect_predictions,
    evaluate_model,
    score_matrix,
)
from ant.inference.escalation import RiskCoveragePolicy
from ant.inference.predictor import ANTPredictor
from ant.model import ANTModel
from ant.training.trainer import Trainer, TrainingConfig


def build_model(args) -> ANTModel:
    if args.backbone == "mock":
        return ANTModel.tiny(scorer=args.scorer, seed=args.seed)
    from ant.config import ANTConfig

    config = ANTConfig(
        backbone_model=args.backbone,
        device=args.device,
        torch_dtype=args.dtype,
        scorer=args.scorer,
        encode_batch_size=args.encode_batch_size,
    )
    return ANTModel(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mock", help="'mock' or an HF model id / local path")
    parser.add_argument("--scorer", default="dot", choices=["dot", "cosine", "bilinear", "mlp"])
    parser.add_argument("--mode", default="head_only", choices=["head_only", "lora", "partial", "full"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--target-risk", type=float, default=0.10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = build_model(args)
    print(f"backbone: {args.backbone} | scorer: {args.scorer} | mode: {args.mode}")

    splits = generate_dataset(
        n_train=args.train_size, n_val=args.val_size, n_test=args.test_size, seed=args.seed
    )
    out_dir = Path(args.out or f"runs/{args.backbone.replace('/', '_')}-{args.scorer}")
    trainer = Trainer(
        model,
        TrainingConfig(
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            checkpoint_dir=str(out_dir),
        ),
    )
    report = trainer.fit(splits.train, splits.val)

    calibrator = None
    escalation = None
    if not args.no_calibration:
        scores, labels, _ = score_matrix(model, splits.val)
        calibrator = build_calibrator("temperature")
        fit_info = calibrator.fit(torch.from_numpy(scores), torch.from_numpy(labels))
        print(f"calibration: {fit_info}")

        predictions = collect_predictions(model, splits.val, calibrator=calibrator)
        escalation = RiskCoveragePolicy(target_risk=args.target_risk)
        escalation.fit(predictions.confidences, predictions.corrects)
        print(f"escalation: {escalation.stats}")

    test_report = evaluate_model(model, splits.test, calibrator=calibrator, name="test")
    ood_report = evaluate_model(model, splits.ood, calibrator=calibrator, name="ood")
    print(json.dumps(test_report.as_row(), indent=2))
    print(json.dumps(ood_report.as_row(), indent=2))

    predictor = ANTPredictor(model, calibrator=calibrator, escalation=escalation)
    predictor.save(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test": test_report.as_row(),
                "ood": ood_report.as_row(),
                "training": {
                    "best_epoch": report.best_epoch,
                    "best_val_loss": report.best_val_loss,
                    "best_val_accuracy": report.best_val_accuracy,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"checkpoint saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
