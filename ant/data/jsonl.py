"""Read and write decision examples as JSON Lines.

One decision per line::

    {"state": "...", "question": "...", "candidates": ["A", "B"], "correct": "B",
     "use_case": "support_routing"}

``use_case`` (or any other extra key) is optional and kept as metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ant.types import DecisionExample


def example_to_row(example: DecisionExample) -> dict:
    row = {
        "state": example.state,
        "question": example.question,
        "candidates": list(example.candidates),
        "correct": example.correct,
    }
    row.update(example.metadata or {})
    return row


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_row(example), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_splits(directory: str | Path) -> dict[str, list[DecisionExample]]:
    """Load ``train/val/test/ood.jsonl`` from a directory; train and val are required."""
    directory = Path(directory)
    splits: dict[str, list[DecisionExample]] = {}
    for name in ("train", "val", "test", "ood"):
        path = directory / f"{name}.jsonl"
        if path.exists():
            splits[name] = load_jsonl(path)
        elif name in ("train", "val"):
            raise FileNotFoundError(f"{path} is required")
    return splits


def load_jsonl(path: str | Path) -> list[DecisionExample]:
    """Load examples; a bad line raises ValueError naming the file and line."""
    path = Path(path)
    examples: list[DecisionExample] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                extra = {
                    key: value
                    for key, value in row.items()
                    if key not in {"state", "question", "candidates", "correct"}
                }
                examples.append(
                    DecisionExample(
                        state=row["state"],
                        question=row["question"],
                        candidates=tuple(row["candidates"]),
                        correct=row["correct"],
                        metadata=extra or None,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return examples
