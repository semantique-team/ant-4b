from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ant.evaluation.calibration import expected_calibration_error
from ant.training.data import DecisionDataset, decision_loader


@dataclass(frozen=True)
class PredictionSet:
    """Model outputs on a dataset, kept per example (candidate sets vary in size)."""

    probabilities: list[np.ndarray]
    labels: list[int]
    corrects: list[bool]
    confidences: list[float]
    candidates: list[list[str]]
    states: list[str]
    questions: list[str]


@dataclass
class EvalReport:
    name: str
    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float
    entropy_mean: float | None = None
    latency_ms_per_example: float | None = None
    peak_vram_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["entropy_mean"] = None if self.entropy_mean is None else round(self.entropy_mean, 4)
        for key in ("accuracy", "nll", "brier", "ece"):
            row[key] = round(row[key], 4)
        if self.latency_ms_per_example is not None:
            row["latency_ms_per_example"] = round(self.latency_ms_per_example, 2)
        if self.peak_vram_mb is not None:
            row["peak_vram_mb"] = round(self.peak_vram_mb, 1)
        return row


def collect_predictions(
    model,
    dataset: DecisionDataset,
    *,
    calibrator: torch.nn.Module | None = None,
    temperature: float = 1.0,
    batch_size: int | None = None,
) -> PredictionSet:
    model.eval()
    loader = decision_loader(dataset, batch_size=batch_size or model.config.encode_batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    corrects: list[bool] = []
    confidences: list[float] = []
    candidates: list[list[str]] = []
    states: list[str] = []
    questions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            scores, mask = model(batch.states, batch.questions, batch.candidates)
            if calibrator is not None:
                scores = calibrator(scores)
            probs = torch.softmax(scores.float() / temperature, dim=-1).cpu().numpy()
            for i, row in enumerate(batch.candidates):
                width = len(row)
                row_probs = probs[i, :width]
                row_probs = row_probs / row_probs.sum()
                probabilities.append(row_probs)
                labels.append(int(batch.labels[i]))
                corrects.append(bool(np.argmax(row_probs) == batch.labels[i]))
                confidences.append(float(row_probs.max()))
                candidates.append(row)
                states.append(batch.states[i])
                questions.append(batch.questions[i])
    return PredictionSet(
        probabilities=probabilities,
        labels=labels,
        corrects=corrects,
        confidences=confidences,
        candidates=candidates,
        states=states,
        questions=questions,
    )


def prediction_metrics(predictions: PredictionSet) -> dict[str, float]:
    n = len(predictions.labels)
    if n == 0:
        raise ValueError("empty prediction set")
    nll = float(
        np.mean([-np.log(np.clip(p[label], 1e-12, 1.0))
                 for p, label in zip(predictions.probabilities, predictions.labels)])
    )
    brier = float(
        np.mean([
            np.sum((p - np.eye(len(p))[label]) ** 2)
            for p, label in zip(predictions.probabilities, predictions.labels)
        ])
    )
    ece = expected_calibration_error(predictions.confidences, predictions.corrects)
    entropies = [float(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in predictions.probabilities]
    return {
        "n": n,
        "accuracy": float(np.mean(predictions.corrects)),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "entropy_mean": float(np.mean(entropies)),
    }


def accuracy_by_use_case(
    predictions: PredictionSet, dataset: DecisionDataset
) -> dict[str, dict[str, float]]:
    """Accuracy and count per ``use_case`` metadata tag (dataset order == prediction order)."""
    groups: dict[str, list[bool]] = {}
    for example, correct in zip(dataset.examples, predictions.corrects):
        tag = (example.metadata or {}).get("use_case", "untagged")
        groups.setdefault(tag, []).append(correct)
    return {
        tag: {"accuracy": float(np.mean(hits)), "n": len(hits)}
        for tag, hits in sorted(groups.items())
    }


def score_matrix(
    model,
    dataset: DecisionDataset,
    *,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (uncalibrated) scores as a padded [M, N_max] matrix plus labels [M]
    and mask [M, N_max]. Padding sits at -1e9, which is softmax-neutral, so the
    matrix can feed calibrators directly."""
    model.eval()
    loader = decision_loader(dataset, batch_size=batch_size or model.config.encode_batch_size, shuffle=False)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            scores, mask = model(batch.states, batch.questions, batch.candidates)
            blocks.append(scores.float().cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
            masks.append(mask.cpu().numpy())
    width = max(block.shape[1] for block in blocks)
    padded_scores = [
        np.pad(block, ((0, 0), (0, width - block.shape[1])), constant_values=-1.0e9)
        for block in blocks
    ]
    padded_masks = [
        np.pad(mask, ((0, 0), (0, width - mask.shape[1])), constant_values=False) for mask in masks
    ]
    return (
        np.concatenate(padded_scores, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(padded_masks, axis=0).astype(bool),
    )


def evaluate_model(
    model,
    dataset: DecisionDataset,
    *,
    calibrator: torch.nn.Module | None = None,
    name: str = "model",
    temperature: float = 1.0,
) -> EvalReport:
    predictions = collect_predictions(model, dataset, calibrator=calibrator, temperature=temperature)
    metrics = prediction_metrics(predictions)
    return EvalReport(name=name, **metrics)


@dataclass
class RiskCoverageResult:
    thresholds: np.ndarray
    coverage: np.ndarray
    risk: np.ndarray
    accuracy: np.ndarray
    aurc: float  # area under the risk-coverage curve (lower is better)
    e_aurc: float  # excessive AURC vs the oracle ranking

    def accuracy_at(self, coverage_target: float) -> float:
        indices = np.where(self.coverage >= coverage_target)[0]
        if indices.size == 0:
            return float("nan")
        return float(self.accuracy[indices[0]])

    def best_threshold(self, target_risk: float | None = None) -> float:
        feasible = np.where(self.risk <= target_risk)[0] if target_risk is not None else np.arange(len(self.thresholds))
        if feasible.size == 0:
            raise ValueError("no threshold meets the target risk")
        return float(self.thresholds[feasible[-1]])


def risk_coverage(confidences: Sequence[float], corrects: Sequence[bool]) -> RiskCoverageResult:
    confidences = np.asarray(confidences, dtype=np.float64)
    corrects = np.asarray(corrects, dtype=bool)
    order = np.argsort(-confidences, kind="stable")
    sorted_correct = corrects[order]
    n = len(order)
    coverage = np.arange(1, n + 1) / n
    errors = (~sorted_correct).cumsum()
    risk = errors / np.arange(1, n + 1)
    accuracy = 1.0 - risk
    aurc = float(risk.mean())
    oracle = np.sort(~corrects).cumsum() / np.arange(1, n + 1)
    e_aurc = float(aurc - oracle.mean())
    thresholds = confidences[order]
    return RiskCoverageResult(
        thresholds=thresholds,
        coverage=coverage,
        risk=risk,
        accuracy=accuracy,
        aurc=aurc,
        e_aurc=e_aurc,
    )
