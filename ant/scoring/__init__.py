from ant.scoring.base import CandidateScorer
from ant.scoring.bilinear import BilinearScorer
from ant.scoring.cosine import CosineScorer
from ant.scoring.dot_product import DotProductScorer
from ant.scoring.mlp import MLPInteractionScorer

SCORER_REGISTRY: dict[str, type[CandidateScorer]] = {
    "dot": DotProductScorer,
    "cosine": CosineScorer,
    "bilinear": BilinearScorer,
    "mlp": MLPInteractionScorer,
}


def build_scorer(name: str, dim: int) -> CandidateScorer:
    try:
        cls = SCORER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown scorer {name!r}; available: {sorted(SCORER_REGISTRY)}") from None
    return cls(dim)


__all__ = [
    "BilinearScorer",
    "CandidateScorer",
    "CosineScorer",
    "DotProductScorer",
    "MLPInteractionScorer",
    "SCORER_REGISTRY",
    "build_scorer",
]
