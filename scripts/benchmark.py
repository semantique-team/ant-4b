"""Scorer and scaling benchmarks (spec §28, §19, §24).

Mock end-to-end run (CPU):
    python scripts/benchmark.py --backbone mock --out runs/bench-mock

Real backbone:
    python scripts/benchmark.py --backbone Qwen/Qwen3.5-4B --device cuda --out runs/bench-qwen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ant.calibration.calibrator import build_calibrator
from ant.config import ANTConfig
from ant.data.synthetic import generate_dataset
from ant.evaluation.benchmark import (
    BenchmarkRow,
    batch_latency_benchmark,
    candidate_count_benchmark,
    rows_to_markdown,
)
from ant.evaluation.metrics import evaluate_model, score_matrix
from ant.inference.predictor import ANTPredictor
from ant.model import ANTModel
from ant.training.trainer import Trainer, TrainingConfig
from ant.utils import format_mb, peak_memory_mb


def build_model(args, scorer: str = "dot", *, backbone=None) -> ANTModel:
    if args.backbone == "mock":
        if backbone is not None:
            return ANTModel(ANTConfig(scorer=scorer), backbone=backbone)
        return ANTModel.tiny(scorer=scorer)
    return ANTModel(
        ANTConfig(
            scorer=scorer,
            backbone_model=args.backbone,
            device=args.device,
            torch_dtype=args.dtype,
            encode_batch_size=args.encode_batch_size,
        ),
        backbone=backbone,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mock")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--scorers", nargs="+", default=["dot", "cosine", "bilinear", "mlp"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--scaling-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out or f"runs/bench-{args.backbone.replace('/', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = generate_dataset(
        n_train=args.train_size, n_val=args.val_size, n_test=args.test_size, seed=args.seed
    )

    backbone = build_model(args).backbone
    print(f"backbone: {args.backbone}")

    rows: list[BenchmarkRow] = []
    for scorer in args.scorers:
        torch.manual_seed(args.seed)
        model = build_model(args, scorer, backbone=backbone)
        trainer = Trainer(
            model,
            TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed,
            ),
        )
        started = time.perf_counter()
        report = trainer.fit(splits.train, splits.val)
        train_seconds = time.perf_counter() - started

        scores, labels, _ = score_matrix(model, splits.val)
        calibrator = build_calibrator("temperature")
        calibrator.fit(torch.from_numpy(scores), torch.from_numpy(labels))
        test_report = evaluate_model(
            model, splits.test, calibrator=calibrator, name=f"frozen+{scorer}"
        )
        metrics = test_report.as_row()
        metrics.pop("name", None)
        metrics["train_seconds"] = round(train_seconds, 1)
        metrics["best_val_acc"] = round(report.best_val_accuracy, 4)
        metrics["peak_vram_mb"] = format_mb(peak_memory_mb(next(backbone.parameters()).device))
        rows.append(BenchmarkRow(name=f"Frozen + {scorer}", metrics=metrics))
        print(rows_to_markdown([rows[-1]]))

    table = rows_to_markdown(
        rows, columns=["accuracy", "nll", "brier", "ece", "train_seconds", "best_val_acc"]
    )
    (out_dir / "scorers.md").write_text(table)
    (out_dir / "scorers.json").write_text(json.dumps([r.as_dict() for r in rows], indent=2))

    best_index = max(range(len(rows)), key=lambda i: rows[i].metrics.get("accuracy", 0))
    best_scorer = args.scorers[best_index]
    print(f"\nbest scorer: {best_scorer}")

    torch.manual_seed(args.seed)
    best_model = build_model(args, best_scorer, backbone=backbone)
    Trainer(
        best_model,
        TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed),
    ).fit(splits.train, splits.val)

    scaling_rows = candidate_count_benchmark(
        best_model, splits.test.examples[:50], sizes=tuple(args.scaling_sizes)
    )
    scaling_table = rows_to_markdown(scaling_rows)
    (out_dir / "candidate_scaling.md").write_text(scaling_table)
    print(scaling_table)

    predictor = ANTPredictor(best_model)
    examples = splits.test.examples[: max(args.batch_sizes)]
    batch_rows = batch_latency_benchmark(
        predictor,
        [e.state for e in examples],
        [e.question for e in examples],
        list(splits.test.examples[0].candidates),
        batch_sizes=tuple(args.batch_sizes),
    )
    batch_table = rows_to_markdown(batch_rows)
    (out_dir / "batch_latency.md").write_text(batch_table)
    print(batch_table)

    print(f"\nresults written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
