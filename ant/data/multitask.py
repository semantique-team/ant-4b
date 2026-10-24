"""Multi-use-case decision dataset built from the seed cases in ``usecases``.

Every seed situation is expanded into several examples that differ in
question wording, candidate subset and order, candidate alias text, and a light
context wrapper (ticket header, forwarded note, ...). The split is made on the
seed situation, so no paraphrase of a validation or test situation is ever seen
in training, and the held-out use cases form a separate OOD split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ant.data.usecases import USE_CASES, UseCase
from ant.types import DecisionExample

SPLITS = ("train", "val", "test", "ood")

_PREFIXES = (
    "",
    "",
    "",
    "New message: ",
    "Forwarded: ",
    "Channel: #inbox. ",
    "Context: ",
    "Update: ",
    "Received via web form. ",
)
_SUFFIXES = ("", "", "", " Thanks.", " (please advise)", " Sent from my phone.")


@dataclass(frozen=True)
class TaggedExample:
    example: DecisionExample
    use_case: str
    split: str


def _split_states(use_case: UseCase, label_index: int, n_states: int) -> list[str]:
    """Assign each seed state of one label to a split; at least two stay in train."""
    if use_case.held_out:
        return ["ood"] * n_states
    if n_states >= 5:
        return ["train"] * (n_states - 2) + ["val", "test"]
    if n_states == 4:
        return ["train", "train", "val", "test"]
    # 3 states: one held-out state, alternating val/test across labels
    return ["train", "train", "val" if label_index % 2 == 0 else "test"]


def _candidate_texts(use_case: UseCase, rng: random.Random) -> dict[str, str]:
    """Surface text per label for one example: canonical, or an alias half the time."""
    use_aliases = rng.random() < 0.5
    return {
        name: rng.choice(label.aliases) if use_aliases and label.aliases else name
        for name, label in use_case.labels.items()
    }


def _make_example(
    use_case: UseCase, gold: str, state: str, rng: random.Random
) -> DecisionExample:
    labels = list(use_case.labels)
    texts = _candidate_texts(use_case, rng)
    others = [name for name in labels if name != gold]
    # keep the gold plus 1..all distractors, so N varies from 2 up to the full set
    keep = rng.randint(1, len(others))
    chosen = rng.sample(others, keep) + [gold]
    rng.shuffle(chosen)
    if use_case.language == "en":
        state = rng.choice(_PREFIXES) + state + rng.choice(_SUFFIXES)
    return DecisionExample(
        state=state,
        question=rng.choice(use_case.questions),
        candidates=tuple(texts[name] for name in chosen),
        correct=texts[gold],
        metadata={"use_case": use_case.name},
    )


def build_multitask(
    *,
    variants: dict[str, int] | None = None,
    seed: int = 7,
    use_cases: tuple[UseCase, ...] = USE_CASES,
) -> list[TaggedExample]:
    """Expand the seed situations into tagged examples across the four splits."""
    variants = {"train": 24, "val": 8, "test": 8, "ood": 8, **(variants or {})}
    if any(count <= 0 for count in variants.values()):
        raise ValueError(f"variant counts must be positive, got {variants}")
    rng = random.Random(seed)
    rows: list[TaggedExample] = []
    for use_case in use_cases:
        for label_index, (gold, label) in enumerate(use_case.labels.items()):
            splits = _split_states(use_case, label_index, len(label.states))
            for state, split in zip(label.states, splits):
                for _ in range(variants[split]):
                    example = _make_example(use_case, gold, state, rng)
                    rows.append(TaggedExample(example, use_case.name, split))
    return rows
