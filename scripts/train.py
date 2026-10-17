"""Train an ANT head (mode 1 by default: frozen Qwen + trainable head).

CPU end-to-end run:
    python scripts/train.py --backbone mock --out runs/mock-dot

Real backbone on the multi-use-case dataset (python scripts/build_dataset.py):
    python scripts/train.py --backbone Qwen/Qwen3.5-4B --device cuda \\
        --data data/ant-multitask --name "ANT multitask" --out runs/ant-multitask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ant.calibration.calibrator import build_calibrator
from ant.data.jsonl import example_to_row, load_splits
from ant.data.synthetic import generate_dataset
from ant.evaluation.metrics import (
    accuracy_by_use_case,
    collect_predictions,
    evaluate_model,
    score_matrix,
)
from ant.inference.escalation import RiskCoveragePolicy
from ant.inference.predictor import ANTPredictor
from ant.model import ANTModel
from ant.training.data import DecisionDataset
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


def write_examples(path: Path, dataset: DecisionDataset, per_use_case: int) -> int:
    """Playground examples: the first ``per_use_case`` held-out rows of each use case."""
    taken: dict[str, int] = {}
    rows = []
    for example in dataset:
        tag = (example.metadata or {}).get("use_case", "untagged")
        if taken.get(tag, 0) >= per_use_case:
            continue
        taken[tag] = taken.get(tag, 0) + 1
        rows.append(example_to_row(example))
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return len(rows)


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
    parser.add_argument(
        "--data",
        default=None,
        help="directory with train.jsonl, val.jsonl and optional test.jsonl / ood.jsonl "
        "(default: the synthetic triage benchmark)",
    )
    parser.add_argument("--name", default=None, help="display name shown in the playground")
    parser.add_argument("--examples-per-use-case", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = build_model(args)
    print(f"backbone: {args.backbone} | scorer: {args.scorer} | mode: {args.mode}")

    if args.data:
        loaded = load_splits(args.data)
        datasets = {name: DecisionDataset(rows) for name, rows in loaded.items()}
        print({name: len(ds) for name, ds in datasets.items()})
    else:
        synthetic = generate_dataset(
            n_train=args.train_size, n_val=args.val_size, n_test=args.test_size, seed=args.seed
        )
        datasets = {
            "train": synthetic.train,
            "val": synthetic.val,
            "test": synthetic.test,
            "ood": synthetic.ood,
        }
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
    report = trainer.fit(datasets["train"], datasets["val"])

    calibrator = None
    escalation = None
    if not args.no_calibration:
        scores, labels, _ = score_matrix(model, datasets["val"])
        calibrator = build_calibrator("temperature")
        fit_info = calibrator.fit(torch.from_numpy(scores), torch.from_numpy(labels))
        print(f"calibration: {fit_info}")

        predictions = collect_predictions(model, datasets["val"], calibrator=calibrator)
        escalation = RiskCoveragePolicy(target_risk=args.target_risk)
        try:
            escalation.fit(predictions.confidences, predictions.corrects)
            print(f"escalation: {escalation.stats}")
        except ValueError as error:
            # a weak model cannot reach the target risk at any threshold
            print(f"WARNING: no escalation policy saved ({error})")
            escalation = None

    metrics: dict = {}
    for split in ("test", "ood"):
        if split not in datasets:
            continue
        dataset = datasets[split]
        metrics[split] = evaluate_model(
            model, dataset, calibrator=calibrator, name=split
        ).as_row()
        metrics[split]["by_use_case"] = accuracy_by_use_case(
            collect_predictions(model, dataset, calibrator=calibrator), dataset
        )
        print(json.dumps(metrics[split], indent=2))

    predictor = ANTPredictor(model, calibrator=calibrator, escalation=escalation)
    predictor.save(out_dir)
    metrics["training"] = {
        "best_epoch": report.best_epoch,
        "best_val_loss": report.best_val_loss,
        "best_val_accuracy": report.best_val_accuracy,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # playground metadata: a display name, and real held-out rows to try
    card = {"display_name": args.name or out_dir.name}
    (out_dir / "card.json").write_text(json.dumps(card, indent=2) + "\n")
    examples_source = datasets.get("test")
    if args.data and examples_source is not None:
        n = write_examples(out_dir / "examples.json", examples_source, args.examples_per_use_case)
        print(f"{n} playground examples written")
    print(f"checkpoint saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
