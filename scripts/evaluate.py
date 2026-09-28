"""Evaluate a trained ANT checkpoint on the synthetic test/OOD splits.

    python scripts/evaluate.py --checkpoint runs/mock-dot --split test
"""

from __future__ import annotations

import argparse
import json

from ant.data.synthetic import generate_dataset
from ant.evaluation.calibration import (
    maximum_calibration_error,
    plot_reliability,
    reliability_curve,
)
from ant.evaluation.metrics import collect_predictions, evaluate_model, risk_coverage
from ant.inference.predictor import ANTPredictor
from ant.evaluation.benchmark import rows_to_markdown, BenchmarkRow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "ood"])
    parser.add_argument("--backbone", default=None, help="reuse a loaded backbone (repo id/path)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--plot", default=None, help="path for the reliability diagram (needs matplotlib)")
    args = parser.parse_args()

    backbone = None
    if args.backbone:
        from ant.backbone.qwen import QwenBackbone

        backbone = QwenBackbone(args.backbone)
    predictor = ANTPredictor.load(args.checkpoint, backbone=backbone)

    splits = generate_dataset(n_test=args.size, seed=args.seed)
    dataset = {"val": splits.val, "test": splits.test, "ood": splits.ood}[args.split]

    report = evaluate_model(
        predictor.model,
        dataset,
        calibrator=predictor.calibrator,
        temperature=predictor.temperature,
        name=args.split,
    )
    predictions = collect_predictions(
        predictor.model, dataset, calibrator=predictor.calibrator, temperature=predictor.temperature
    )

    rc = risk_coverage(predictions.confidences, predictions.corrects)
    curve = reliability_curve(predictions.confidences, predictions.corrects)
    mce = maximum_calibration_error(predictions.confidences, predictions.corrects)

    print(f"\n== {args.split} (n={report.n}) ==")
    print(
        rows_to_markdown(
            [
                BenchmarkRow(
                    name=report.name,
                    metrics={
                        "accuracy": round(report.accuracy, 4),
                        "nll": round(report.nll, 4),
                        "brier": round(report.brier, 4),
                        "ece": round(report.ece, 4),
                        "mce": round(mce, 4),
                        "aurc": round(rc.aurc, 4),
                        "e-aurc": round(rc.e_aurc, 4),
                    },
                )
            ]
        )
    )
    print("reliability bins (confidence -> accuracy, count):")
    for b in curve:
        print(f"  [{b.lower:.2f}-{b.upper:.2f}] {b.avg_confidence:.3f} -> {b.accuracy:.3f}  ({b.count})")
    print(f"coverage 80% -> accuracy {rc.accuracy_at(0.8):.4f}")
    print(f"coverage 95% -> accuracy {rc.accuracy_at(0.95):.4f}")

    if args.plot:
        saved = plot_reliability(predictions.confidences, predictions.corrects, args.plot)
        print("reliability diagram" + (f" saved to {args.plot}" if saved else " skipped (matplotlib missing)"))

    print("\nescalation policy:", json.dumps(predictor.escalation.to_config()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
