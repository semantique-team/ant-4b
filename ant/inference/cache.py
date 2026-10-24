from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch


class CandidateEmbeddingCache:
    """Precomputed candidate embeddings for static candidate sets.

    With a cached set, inference cost is one context encoding plus N dot
    products: the main practical advantage of candidate scoring.
    """

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def contains(self, candidate: str) -> bool:
        return candidate in self._store

    def store(self, candidates: Sequence[str], embeddings: torch.Tensor) -> None:
        """Insert already-computed embeddings, honouring ``max_size``.

        Kept separate from ``embeddings`` so a caller that encoded candidates in a
        fused backbone pass (rather than through ``model.encode_candidates``) can
        still populate the cache with the public API.
        """
        if len(candidates) != embeddings.shape[0]:
            raise ValueError(
                f"got {len(candidates)} candidates but {embeddings.shape[0]} embeddings"
            )
        stored = list(dict.fromkeys(candidates))
        if self.max_size is not None and len(stored) > self.max_size:
            raise ValueError(
                f"storing {len(stored)} unique candidates but cache max_size is {self.max_size}"
            )
        detached = embeddings.detach()
        by_candidate = {c: detached[i] for i, c in enumerate(candidates)}
        for candidate in stored:
            self._store[candidate] = by_candidate[candidate]
        if self.max_size is not None:
            for candidate in list(self._store):
                if len(self._store) <= self.max_size:
                    break
                if candidate not in stored:
                    self._store.pop(candidate)

    def embeddings(
        self,
        model,
        candidates: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return [N, d] embeddings in input order, encoding and caching only
        the candidates not seen before."""
        requested = list(dict.fromkeys(candidates))
        if self.max_size is not None and len(requested) > self.max_size:
            raise ValueError(
                f"requesting {len(requested)} unique candidates but cache max_size is {self.max_size}"
            )
        missing = [c for c in requested if c not in self._store]
        if missing:
            fresh = model.encode_candidates(missing, batch_size=batch_size).detach()
            self.store(missing, fresh)
        return torch.stack([self._store[c] for c in candidates])

    def save(self, path: str | Path) -> None:
        torch.save({k: v.cpu() for k, v in self._store.items()}, Path(path))

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> None:
        self._store.update(torch.load(Path(path), map_location=map_location, weights_only=False))
