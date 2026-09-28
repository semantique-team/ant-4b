from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ant.types import DecisionExample


class DecisionDataset(Dataset):
    def __init__(self, examples: Sequence[DecisionExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecisionExample:
        return self.examples[index]

    def __iter__(self) -> Iterator[DecisionExample]:
        return iter(self.examples)


@dataclass
class DecisionBatch:
    states: list[str]
    questions: list[str]
    candidates: list[list[str]]
    labels: torch.Tensor  # [B] index of the correct candidate in its row
    mask: torch.Tensor  # [B, N_max] True for real candidates


def collate_decisions(examples: Sequence[DecisionExample]) -> DecisionBatch:
    rows = [list(example.candidates) for example in examples]
    width = max(len(row) for row in rows)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    labels = torch.zeros(len(rows), dtype=torch.long)
    for i, (row, example) in enumerate(zip(rows, examples)):
        mask[i, : len(row)] = True
        labels[i] = row.index(example.correct)
    return DecisionBatch(
        states=[e.state for e in examples],
        questions=[e.question for e in examples],
        candidates=rows,
        labels=labels,
        mask=mask,
    )


def decision_loader(
    dataset: DecisionDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_decisions,
        generator=generator,
        num_workers=num_workers,
    )
