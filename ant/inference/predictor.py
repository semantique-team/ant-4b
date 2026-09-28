from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ant.calibration.calibrator import IdentityCalibrator, build_calibrator
from ant.config import ANTConfig
from ant.inference.cache import CandidateEmbeddingCache
from ant.inference.escalation import (
    AlwaysExecute,
    ConfidenceMetrics,
    EscalationPolicy,
    compute_confidence_metrics,
)
from ant.model import ANTModel
from ant.types import PredictionResult


class ANTPredictor:
    """High-level System 1 API: predict / predict_batch / score_candidates.

    Composes the model with an optional calibrator, an escalation policy, and
    an optional candidate-embedding cache. Calibrator and policy are applied in
    that order; ``temperature`` is a final user knob (leave at 1.0 to preserve
    fitted calibration).
    """

    def __init__(
        self,
        model: ANTModel,
        *,
        calibrator: nn.Module | None = None,
        escalation: EscalationPolicy | None = None,
        temperature: float | None = None,
        candidate_cache: CandidateEmbeddingCache | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator or IdentityCalibrator()
        self.escalation = escalation or AlwaysExecute()
        self.temperature = temperature if temperature is not None else model.config.temperature
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.candidate_cache = candidate_cache

    @classmethod
    def from_config(
        cls, config: ANTConfig, *, backbone: nn.Module | None = None, **kwargs
    ) -> "ANTPredictor":
        return cls(ANTModel(config, backbone=backbone), **kwargs)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backbone: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "ANTPredictor":
        path = Path(path)
        model = ANTModel.load(path, backbone=backbone, map_location=map_location)
        calibrator = None
        if (path / "calibrator.pt").exists():
            blob = torch.load(path / "calibrator.pt", map_location=map_location, weights_only=False)
            calibrator = build_calibrator(blob["name"])
            calibrator.load_state_dict(blob["state_dict"])
        escalation = None
        policy_file = path / "escalation.json"
        if policy_file.exists():
            escalation = EscalationPolicy.from_config(json.loads(policy_file.read_text()))
        return cls(model, calibrator=calibrator, escalation=escalation)

    def save(self, path: str | Path) -> Path:
        path = self.model.save_head(path)
        if not isinstance(self.calibrator, IdentityCalibrator):
            torch.save(
                {
                    "name": getattr(self.calibrator, "name", "custom"),
                    "state_dict": self.calibrator.state_dict(),
                },
                path / "calibrator.pt",
            )
        if not isinstance(self.escalation, AlwaysExecute):
            (path / "escalation.json").write_text(
                json.dumps(self.escalation.to_config(), indent=2) + "\n"
            )
        return path

    # -- low level -----------------------------------------------------------

    def _candidate_embeddings(self, candidates: Sequence[str]) -> torch.Tensor:
        if self.candidate_cache is not None:
            return self.candidate_cache.embeddings(
                self.model, candidates, batch_size=self.model.config.encode_batch_size
            )
        return self.model.encode_candidates(
            candidates, batch_size=self.model.config.encode_batch_size
        )

    def _probabilities(self, scores: torch.Tensor) -> np.ndarray:
        logits = self.calibrator(scores).float().detach() / self.temperature
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def score_candidates(
        self, state: str, question: str, candidates: Sequence[str]
    ) -> np.ndarray:
        """Raw (uncalibrated) scores in input order."""
        scores = self.model.score_candidates(state, question, candidates)
        return scores.detach().float().cpu().numpy()

    # -- prediction ----------------------------------------------------------

    def _result_from_probs(
        self,
        probabilities: np.ndarray,
        candidates: Sequence[str],
        scores: np.ndarray | None = None,
    ) -> PredictionResult:
        metrics: ConfidenceMetrics = compute_confidence_metrics(probabilities)
        action = self.escalation.decide(metrics)
        probs_by_candidate: dict[str, float] = {}
        scores_by_candidate: dict[str, float] | None = {} if scores is not None else None
        for i, candidate in enumerate(candidates):
            probs_by_candidate[candidate] = probs_by_candidate.get(candidate, 0.0) + float(
                probabilities[i]
            )
            if scores_by_candidate is not None:
                scores_by_candidate[candidate] = float(scores[i])
        ordered = dict(sorted(probs_by_candidate.items(), key=lambda kv: -kv[1]))
        if scores_by_candidate is not None:
            scores_by_candidate = dict(
                sorted(scores_by_candidate.items(), key=lambda kv: -probs_by_candidate[kv[0]])
            )
        return PredictionResult(
            choice=candidates[metrics.choice_index],
            action=action,
            probabilities=ordered,
            scores=scores_by_candidate,
            confidence=metrics.confidence,
            margin=metrics.margin,
            entropy=metrics.entropy,
            normalized_entropy=metrics.normalized_entropy,
            n_candidates=metrics.n_candidates,
        )

    def predict(
        self, state: str, question: str, candidates: Sequence[str]
    ) -> PredictionResult:
        candidates = list(candidates)
        if not candidates:
            raise ValueError("at least one candidate is required")
        z_context = self.model.encode_contexts([state], [question])
        z_candidates = self._candidate_embeddings(candidates)
        scores = self.model.score_embeddings(z_context, z_candidates.unsqueeze(0))[0]
        probabilities = self._probabilities(scores)
        return self._result_from_probs(
            probabilities, candidates, scores=scores.detach().float().cpu().numpy()
        )

    def predict_batch(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        candidates: Sequence[str] | Sequence[Sequence[str]],
    ) -> list[PredictionResult]:
        """Optimized for the many-states / same-candidate-set case: candidates
        are embedded once (cache-aware); contexts are embedded in batches."""
        if len(states) != len(questions):
            raise ValueError("states and questions must have the same length")
        shared = isinstance(candidates[0], str)
        rows: list[list[str]] = [list(candidates)] * len(states) if shared else [
            list(row) for row in candidates  # type: ignore[arg-type]
        ]
        results: list[PredictionResult] = []
        batch = self.model.config.encode_batch_size
        for start in range(0, len(states), batch):
            chunk = range(start, min(start + batch, len(states)))
            z_context = self.model.encode_contexts(
                [states[i] for i in chunk], [questions[i] for i in chunk]
            )
            for i in chunk:
                z_candidates = self._candidate_embeddings(rows[i])
                scores = self.model.score_embeddings(
                    z_context[i - start].unsqueeze(0), z_candidates.unsqueeze(0)
                )[0]
                probabilities = self._probabilities(scores)
                results.append(
                    self._result_from_probs(
                        probabilities,
                        rows[i],
                        scores=scores.detach().float().cpu().numpy(),
                    )
                )
        return results
