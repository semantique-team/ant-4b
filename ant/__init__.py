"""ANT-4B: a System 1 candidate-scoring decision engine built on Qwen3.5-4B.

The model never generates text. Given (state, question, candidates) it returns a
score per candidate and a calibrated probability distribution over them.
"""

from ant.config import ANTConfig
from ant.model import ANTModel
from ant.types import DecisionExample, DecisionInput, PredictionResult

__version__ = "0.1.0a0"

__all__ = [
    "ANTConfig",
    "ANTModel",
    "DecisionExample",
    "DecisionInput",
    "PredictionResult",
    "__version__",
]
