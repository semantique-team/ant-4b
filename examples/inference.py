"""ANT-4B inference example.

With a trained checkpoint:
    python examples/inference.py --checkpoint runs/mock-dot

Without one (random head, output distribution is meaningless but the API is identical):
    python examples/inference.py
"""

from __future__ import annotations

import argparse

from ant.config import ANTConfig
from ant.inference.cache import CandidateEmbeddingCache
from ant.inference.escalation import RiskCoveragePolicy, ThresholdPolicy
from ant.inference.predictor import ANTPredictor
from ant.model import ANTModel

STATE = """
Customer has been charged twice for the same invoice.
The customer is requesting a refund.
"""
QUESTION = "Which department should handle this case?"
CANDIDATES = ["Billing", "Technical Support", "Sales", "Legal"]

STATIC_CANDIDATES = ["Billing", "Sales", "Technical Support", "Legal", "Customer Success", "Security"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default=None, help="backbone id/path when no checkpoint is given")
    args = parser.parse_args()

    if args.checkpoint:
        ant = ANTPredictor.load(args.checkpoint)
    elif args.model:
        ant = ANTPredictor.from_config(ANTConfig(backbone_model=args.model))
    else:
        print("no checkpoint given: using a randomly initialized head (probabilities are meaningless)\n")
        ant = ANTPredictor(ANTModel.tiny())

    ant.candidate_cache = CandidateEmbeddingCache()

    result = ant.predict(state=STATE, question=QUESTION, candidates=CANDIDATES)
    print(result.to_dict())

    ant.escalation = ThresholdPolicy(metric="confidence", threshold=0.61)
    result = ant.predict(state=STATE, question=QUESTION, candidates=CANDIDATES)
    print("\nwith threshold policy:", {"choice": result.choice, "action": result.action,
                                       "confidence": result.confidence})

    batch = ant.predict_batch(
        states=[STATE, "Our onboarding has stalled and nobody has contacted us in two weeks."],
        questions=[QUESTION] * 2,
        candidates=STATIC_CANDIDATES,
    )
    print("\nbatch (shared candidates, cached embeddings):")
    for item in batch:
        print(f"  {item.choice:20} confidence={item.confidence:.3f} action={item.action}")
    print(f"cache size after batch: {len(ant.candidate_cache)}")

    print("\nlow-level scores:")
    print(ant.score_candidates(STATE, QUESTION, CANDIDATES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
