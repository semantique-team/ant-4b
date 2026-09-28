from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ant.model import ANTModel
from ant.training.data import DecisionDataset
from ant.types import DecisionExample
from ant.utils import format_mb, peak_memory_mb, reset_peak_memory, synchronize


@dataclass
class BenchmarkRow:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.metrics}


def rows_to_markdown(rows: Sequence[BenchmarkRow], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    keys = list(columns) if columns else list(rows[0].metrics.keys())
    header = "| " + " | ".join(["model"] + keys) + " |"
    separator = "|" + "---|" * (len(keys) + 1)
    lines = [header, separator]
    for row in rows:
        values = [str(row.metrics.get(key, "n/a")) for key in keys]
        lines.append("| " + " | ".join([row.name] + values) + " |")
    return "\n".join(lines) + "\n"


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
    device: torch.device | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    seconds = (time.perf_counter() - start) / repeats
    return {"latency_ms": seconds * 1e3, "throughput_per_s": 1.0 / seconds if seconds > 0 else 0.0}


def _rebuild_with_scorer(model: ANTModel, scorer_name: str) -> ANTModel:
    from ant.config import ANTConfig
    from ant.scoring import build_scorer

    config = ANTConfig.from_dict({**model.config.to_dict(), "scorer": scorer_name})
    fresh = ANTModel.__new__(ANTModel)
    torch.nn.Module.__init__(fresh)
    fresh.config = config
    fresh.backbone = model.backbone
    from ant.encoding.candidate_encoder import CandidateEncoder
    from ant.encoding.context_encoder import ContextEncoder

    fresh.context_encoder = ContextEncoder(
        model.backbone,
        pooling=config.context_pooling,
        proj_dim=config.scoring_dim,
        dropout=config.dropout,
        template=config.context_template,
    )
    fresh.candidate_encoder = CandidateEncoder(
        model.backbone,
        pooling=config.candidate_pooling,
        proj_dim=config.scoring_dim,
        dropout=config.dropout,
        template=config.candidate_template,
    )
    fresh.scorer = build_scorer(scorer_name, config.scoring_dim)
    device = next(model.backbone.parameters()).device
    fresh.context_encoder.to(device)
    fresh.candidate_encoder.to(device)
    fresh.scorer.to(device)
    return fresh


def candidate_count_benchmark(
    model: ANTModel,
    examples: Sequence[DecisionExample],
    *,
    sizes: Sequence[int] = (2, 4, 8, 16, 32, 64, 128),
    fillers: Sequence[str] | None = None,
) -> list[BenchmarkRow]:
    """Subsample/extend candidate sets to fixed sizes (correct answer always
    kept) and measure accuracy and latency as N grows."""
    if fillers is None:
        fillers = [f"Generic team {i}" for i in range(200)]
    pool = list(fillers)
    rows: list[BenchmarkRow] = []
    for size in sizes:
        adjusted: list[DecisionExample] = []
        for position, example in enumerate(examples):
            needed = size - 1
            extras: list[str] = []
            sources = [c for c in example.candidates if c != example.correct] + pool
            for source in sources:
                if len(extras) >= needed:
                    break
                if source != example.correct and source not in extras:
                    extras.append(source)
            candidates = [example.correct] + extras
            rng = np.random.default_rng(position)
            rng.shuffle(candidates)
            adjusted.append(
                DecisionExample(
                    state=example.state,
                    question=example.question,
                    candidates=candidates,
                    correct=example.correct,
                    metadata=example.metadata,
                )
            )
        dataset = DecisionDataset(adjusted)

        def run(dataset=dataset):
            states = [e.state for e in dataset]
            questions = [e.question for e in dataset]
            candidate_rows = [list(e.candidates) for e in dataset]
            with torch.no_grad():
                model(states, questions, candidate_rows)

        timing = measure_latency(run, warmup=1, repeats=3, device=next(model.backbone.parameters()).device)
        scores_list = []
        labels = []
        with torch.no_grad():
            for start in range(0, len(dataset), model.config.encode_batch_size):
                chunk = dataset.examples[start : start + model.config.encode_batch_size]
                scores, _ = model(
                    [e.state for e in chunk],
                    [e.question for e in chunk],
                    [list(e.candidates) for e in chunk],
                )
                for i, example in enumerate(chunk):
                    width = len(example.candidates)
                    probs = torch.softmax(scores[i, :width].float(), dim=-1).cpu().numpy()
                    scores_list.append(probs)
                    labels.append(list(example.candidates).index(example.correct))
        accuracy = float(np.mean([np.argmax(p) == l for p, l in zip(scores_list, labels)]))
        rows.append(
            BenchmarkRow(
                name=f"N={size}",
                metrics={
                    "accuracy": round(accuracy, 4),
                    "latency_ms_batch": round(timing["latency_ms"], 1),
                    "throughput_per_s": round(timing["throughput_per_s"], 2),
                },
            )
        )
    return rows


def batch_latency_benchmark(
    predictor,
    states: Sequence[str],
    questions: Sequence[str],
    candidates: Sequence[str],
    *,
    batch_sizes: Sequence[int] = (1, 8, 32, 64, 128),
) -> list[BenchmarkRow]:
    """Many states, same candidate set: the batch path encodes candidates once."""
    rows: list[BenchmarkRow] = []
    total = len(states)
    for batch_size in batch_sizes:
        n = min(batch_size, total)
        subset_states = states[:n]
        subset_questions = questions[:n]

        def run(n=n, subset_states=subset_states, subset_questions=subset_questions):
            predictor.predict_batch(subset_states, subset_questions, candidates)

        timing = measure_latency(run, warmup=1, repeats=3)
        rows.append(
            BenchmarkRow(
                name=f"batch={n}",
                metrics={
                    "latency_ms": round(timing["latency_ms"], 1),
                    "per_example_ms": round(timing["latency_ms"] / n, 2),
                    "throughput_per_s": round(n / max(timing["latency_ms"] / 1e3, 1e-9), 2),
                },
            )
        )
    return rows


def peak_memory_probe(fn: Callable[[], Any], device: torch.device | None = None) -> dict[str, Any]:
    reset_peak_memory(device)
    fn()
    synchronize(device)
    return {"peak_vram_mb": format_mb(peak_memory_mb(device))}
