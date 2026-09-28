from __future__ import annotations

import random
from itertools import islice
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")

_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    """BF16 on CUDA; FP32 elsewhere by default (linear-attention ops are not
    reliably BF16 on MPS/CPU and a 4B model in FP32 on CUDA is wasteful)."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {dtype!r}; use auto or one of {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def synchronize(device: torch.device | str | None = None) -> None:
    device = torch.device(device) if device is not None else None
    kind = (device or torch.device("cpu")).type if device is not None else "cpu"
    try:
        if kind == "cuda":
            torch.cuda.synchronize(device)
        elif kind == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def peak_memory_mb(device: torch.device | None = None) -> float | None:
    """Best-effort peak/allocated memory in MB; None when not measurable."""
    try:
        device = device or (torch.device("cuda") if torch.cuda.is_available() else None)
        if device is not None and device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / 1e6
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 1e6
    except Exception:
        pass
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def reset_peak_memory(device: torch.device | None = None) -> None:
    try:
        if device is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def format_mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def softmax_np(scores, temperature: float = 1.0, axis: int | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-9)
    values = values - values.max(axis=axis, keepdims=True)
    weights = np.exp(values)
    return weights / weights.sum(axis=axis, keepdims=True)
