from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Action = Literal["execute", "escalate"]


@dataclass(frozen=True)
class DecisionInput:
    """A single routing decision: context, question, and an unordered candidate set."""

    state: str
    question: str
    candidates: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state.strip():
            raise ValueError("state must be a non-empty string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("at least one candidate is required")
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("candidates must be non-empty strings")
        object.__setattr__(self, "candidates", candidates)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class DecisionExample(DecisionInput):
    """A supervised decision: the same input plus the correct candidate."""

    correct: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.correct:
            raise ValueError("correct candidate is required")
        if self.correct not in self.candidates:
            raise ValueError(
                f"correct candidate {self.correct!r} is not among candidates {list(self.candidates)}"
            )


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of a single prediction, exposing everything a System 1/2 router needs."""

    choice: str
    action: Action
    probabilities: dict[str, float]
    scores: dict[str, float] | None = None
    confidence: float = 0.0
    margin: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None
    n_candidates: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "action": self.action,
            "probabilities": dict(self.probabilities),
            "scores": dict(self.scores) if self.scores is not None else None,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "n_candidates": self.n_candidates,
        }
