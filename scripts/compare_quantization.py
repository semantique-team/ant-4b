r"""Does a quantized backbone still make the same decisions?

Both runs load the same head checkpoint and the same calibration; only the
backbone differs (unquantized reference vs quantized). The gate is
|Δaccuracy| <= 0.005 and ΔECE <= 0.02.

    python scripts/compare_quantization.py --checkpoint runs/ant-4b-2026-10-17 \
        --bits 8 --split test

MLX arm (two separately converted model folders):
    python scripts/compare_quantization.py --checkpoint runs/ant-08b \
        --backend mlx --quantization mlx \
        --backbone models/Qwen3.5-0.8B-mlx \
        --quantized-backbone models/Qwen3.5-0.8B-mlx-8bit --bits 8

Exit code: 0 when the gate passes, 1 when it fails or the platform cannot run
Metal quantization (the message then says what to install or enable).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ant.config import ANTConfig
from ant.data.jsonl import load_jsonl, load_splits
from ant.evaluation.metrics import PredictionSet, collect_predictions, prediction_metrics
from ant.inference.predictor import ANTPredictor
from ant.training.data import DecisionDataset
from ant.types import DecisionExample

ACCURACY_TOLERANCE = 0.005
ECE_TOLERANCE = 0.02
REFERENCE_DTYPES = ("fp16", "float16", "bf16", "bfloat16", "fp32", "float32")


# -- environment ---------------------------------------------------------------


def kernels_available() -> bool:
    """Match the check ``build_quantization_config`` makes, without importing it."""
    try:
        from transformers.utils import is_kernels_available
    except ImportError:
        try:
            import kernels  # noqa: F401
        except ImportError:
            return False
        return True
    return bool(is_kernels_available())


def dependency_error(args: argparse.Namespace) -> str | None:
    """Actionable message when the selected quantizer cannot run here, else None."""
    if args.quantization == "mlx":
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return (
                "the `mlx-lm` package is not installed, so MLX quantization cannot run.\n"
                "  install it with:  pip install -e '.[mlx]'\n"
                "  (Apple Silicon only; the unquantized reference needs neither)"
            )
        return None
    if not kernels_available():
        return (
            "the `kernels` package is not installed, so Metal quantization cannot run.\n"
            "  install it with:  pip install kernels\n"
            "  (Apple Silicon + MPS only; the unquantized reference needs neither)"
        )
    if not torch.backends.mps.is_available():
        return (
            "MPS is not available, so Metal quantization cannot run.\n"
            "  run this on an Apple Silicon Mac with MPS enabled"
        )
    return None


def release_memory() -> None:
    """Drop one backbone before the next one is loaded: two 4B runs do not fit twice."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -- data and scoring ----------------------------------------------------------


def load_rows(path: str, split: str, max_examples: int | None) -> list[DecisionExample]:
    location = Path(path)
    if location.is_file():
        rows = load_jsonl(location)
    else:
        splits = load_splits(location)
        if split not in splits:
            available = ", ".join(sorted(splits)) or "none"
            raise ValueError(f"split {split!r} not found in {path} (available: {available})")
        rows = splits[split]
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError(f"split {split!r} is empty")
    return rows


def build_predictor(
    checkpoint: Path, backbone_id: str, args: argparse.Namespace, *, quantized: bool
) -> ANTPredictor:
    if args.backend == "mlx" or args.quantization == "mlx":
        from ant.backbone.mlx import MlxBackbone

        extra = {}
        if quantized:
            extra = {
                "quantization": "mlx",
                "quantization_bits": args.bits,
                "quantization_group_size": args.group_size,
            }
        backbone = MlxBackbone(
            backbone_id,
            torch_dtype=args.reference,
            device="auto",
            **extra,
        )
        return ANTPredictor.load(checkpoint, backbone=backbone)

    from ant.backbone.qwen import QwenBackbone

    extra = {}
    if quantized:
        extra = {
            "quantization": args.quantization,
            "quantization_bits": args.bits,
            "quantization_group_size": args.group_size,
        }
    backbone = QwenBackbone(backbone_id, torch_dtype=args.reference, device="auto", **extra)
    return ANTPredictor.load(checkpoint, backbone=backbone)


def predict_rows(predictor: ANTPredictor, rows: Sequence[DecisionExample]) -> PredictionSet:
    """Score every row through the evaluation harness (same calibration as serving)."""
    return collect_predictions(
        predictor.model,
        DecisionDataset(rows),
        calibrator=predictor.calibrator,
        temperature=predictor.temperature,
    )


def summarise(predictions: PredictionSet) -> dict[str, float]:
    metrics = prediction_metrics(predictions)
    return {"accuracy": metrics["accuracy"], "ece": metrics["ece"]}


# -- report --------------------------------------------------------------------


def render_quality_table(
    n: int,
    reference: dict[str, float],
    quantized: dict[str, float],
    reference_label: str,
    quantized_label: str,
) -> str:
    delta_accuracy = quantized["accuracy"] - reference["accuracy"]
    delta_ece = quantized["ece"] - reference["ece"]
    rows = [
        ("n", f"{n}", f"{n}"),
        ("accuracy", f"{reference['accuracy']:.4f}", f"{quantized['accuracy']:.4f}"),
        ("Δaccuracy", "", f"{delta_accuracy:+.4f}"),
        ("ECE", f"{reference['ece']:.4f}", f"{quantized['ece']:.4f}"),
        ("ΔECE", "", f"{delta_ece:+.4f}"),
    ]
    width = max(len(row[0]) for row in rows)
    left = max(len(reference_label), len(quantized_label), 9)
    lines = [
        f"| {'metric':<{width}} | {reference_label:>{left}} | {quantized_label:>{left}} |",
        f"|{'-' * (width + 2)}|{'-' * (left + 2)}|{'-' * (left + 2)}|",
    ]
    lines += [
        f"| {name:<{width}} | {ref:>{left}} | {quant:>{left}} |"
        for name, ref, quant in rows
    ]
    return "\n".join(lines)


def render_delta_table(
    mean_abs_delta: float, max_abs_delta: float, flips: int, n: int
) -> str:
    return "\n".join(
        [
            "| distribution delta     | value        |",
            "|------------------------|--------------|",
            f"| mean per-row \\|Δp\\|    | {mean_abs_delta:.5f}      |",
            f"| max per-row \\|Δp\\|     | {max_abs_delta:.5f}      |",
            f"| argmax flips           | {flips} / {n} ({100.0 * flips / n:.1f}%) |",
        ]
    )


def gate_line(delta_accuracy: float, delta_ece: float) -> tuple[bool, str]:
    """|Δaccuracy| <= 0.005 and ΔECE <= 0.02 (an ECE improvement always passes)."""
    failures: list[str] = []
    if abs(delta_accuracy) > ACCURACY_TOLERANCE:
        failures.append(
            f"|Δaccuracy| = {abs(delta_accuracy):.4f} > {ACCURACY_TOLERANCE}"
        )
    if delta_ece > ECE_TOLERANCE:
        failures.append(f"ΔECE = {delta_ece:+.4f} > {ECE_TOLERANCE}")
    if failures:
        return False, "GATE: FAIL — " + "; ".join(failures)
    return True, (
        f"GATE: PASS — |Δaccuracy| = {abs(delta_accuracy):.4f} <= {ACCURACY_TOLERANCE} "
        f"and ΔECE = {delta_ece:+.4f} <= {ECE_TOLERANCE}"
    )


def write_json(path: str, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")


# -- CLI -----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reference",
        default="fp16",
        help="dtype of the unquantized reference backbone on the same model (default: fp16)",
    )
    parser.add_argument("--quantization", default="metal", choices=("metal", "mlx"))
    parser.add_argument("--backend", default=None, choices=("torch", "mlx"), help="default: infer from --quantization")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", dest="group_size", type=int, default=64)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "ood"])
    parser.add_argument("--data", default="data/ant-multitask")
    parser.add_argument("--max-examples", dest="max_examples", type=int, default=None)
    parser.add_argument("--backbone", default=None, help="override the checkpoint's backbone id/path")
    parser.add_argument(
        "--quantized-backbone",
        default=None,
        help="MLX only: the separately converted quantized model (bits/group must match)",
    )
    parser.add_argument("--out", default=None, help="also write the report as JSON here")
    args = parser.parse_args()
    args.backend = args.backend or ("mlx" if args.quantization == "mlx" else "torch")
    if args.quantization == "mlx" and args.backend != "mlx":
        parser.error("--quantization mlx requires --backend mlx")
    if args.quantization == "metal" and args.backend == "mlx":
        parser.error("--backend mlx requires --quantization mlx (or use the torch backend)")

    if args.reference not in REFERENCE_DTYPES:
        parser.error(f"--reference must be one of {REFERENCE_DTYPES}")
    if args.bits not in (2, 4, 8):
        parser.error("--bits must be 2, 4 or 8")
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not (checkpoint / "ant_head.pt").exists():
        print(f"error: {checkpoint / 'ant_head.pt'} is missing", file=sys.stderr)
        return 1
    config_file = checkpoint / "config.json"
    if not config_file.exists():
        print(f"error: {config_file} is missing", file=sys.stderr)
        return 1

    error = dependency_error(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 1

    config = ANTConfig.load(config_file)
    if config.backbone_model == "mock":
        print(
            f"error: {checkpoint} was trained on the mock backbone; a quantization "
            "comparison needs a real Qwen backbone",
            file=sys.stderr,
        )
        return 1
    backbone_id = args.backbone or config.backbone_model
    quantized_backbone_id = backbone_id
    if args.quantization == "mlx":
        if args.quantized_backbone is None:
            print(
                "error: --quantized-backbone is required for MLX quantization: "
                "the unquantized and quantized arms are separate MLX model folders",
                file=sys.stderr,
            )
            return 1
        quantized_backbone_id = args.quantized_backbone

    try:
        rows = load_rows(args.data, args.split, args.max_examples)
    except (FileNotFoundError, ValueError) as load_error:
        print(f"error: {load_error}", file=sys.stderr)
        return 1

    print(f"checkpoint : {checkpoint}")
    print(f"backbone   : {backbone_id}")
    print(f"data       : {args.data} [{args.split}]  n={len(rows)}")

    reference_predictor = build_predictor(checkpoint, backbone_id, args, quantized=False)
    reference_predictions = predict_rows(reference_predictor, rows)
    reference = summarise(reference_predictions)
    del reference_predictor
    release_memory()

    quantized_predictor = build_predictor(
        checkpoint, quantized_backbone_id, args, quantized=True
    )
    quantized_predictions = predict_rows(quantized_predictor, rows)
    del quantized_predictor
    if reference_predictions.labels != quantized_predictions.labels:
        print("error: the two runs scored different rows", file=sys.stderr)
        return 1
    quantized = summarise(quantized_predictions)
    release_memory()

    row_means: list[float] = []
    row_maxima: list[float] = []
    flips = 0
    for reference_row, quantized_row in zip(
        reference_predictions.probabilities, quantized_predictions.probabilities
    ):
        delta = np.abs(reference_row - quantized_row)
        row_means.append(float(delta.mean()))
        row_maxima.append(float(delta.max()))
        if np.argmax(reference_row) != np.argmax(quantized_row):
            flips += 1
    mean_abs_delta = float(np.mean(row_means))
    max_abs_delta = float(np.max(row_maxima))
    delta_accuracy = quantized["accuracy"] - reference["accuracy"]
    delta_ece = quantized["ece"] - reference["ece"]
    passed, gate = gate_line(delta_accuracy, delta_ece)

    reference_label = f"reference ({args.reference})"
    quantized_label = f"{args.quantization} {args.bits}-bit g{args.group_size}"
    print()
    print(render_quality_table(len(rows), reference, quantized, reference_label, quantized_label))
    print()
    print(render_delta_table(mean_abs_delta, max_abs_delta, flips, len(rows)))
    print()
    print(gate)

    if args.out:
        report = {
            "checkpoint": str(checkpoint),
            "backbone": backbone_id,
            "data": args.data,
            "split": args.split,
            "n": len(rows),
            "reference": {"dtype": args.reference, **reference},
            "quantized": {
                "quantization": args.quantization,
                "bits": args.bits,
                "group_size": args.group_size,
                **quantized,
            },
            "delta_accuracy": delta_accuracy,
            "delta_ece": delta_ece,
            "mean_abs_delta_p": mean_abs_delta,
            "max_abs_delta_p": max_abs_delta,
            "argmax_flips": flips,
            "argmax_flip_rate": flips / len(rows),
            "gate": {
                "passed": passed,
                "accuracy_tolerance": ACCURACY_TOLERANCE,
                "ece_tolerance": ECE_TOLERANCE,
            },
        }
        write_json(args.out, report)
        print(f"report written to {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
