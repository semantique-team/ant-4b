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


def supports_bf16(device: torch.device) -> bool:
    """Native bf16 needs CUDA compute capability >= 8 (Ampere+); a T4 is 7.5."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    """Default floating-point type per device when the caller passes ``auto``.

    CUDA uses BF16 on Ampere+ and FP16 on older cards (e.g. T4). MPS uses FP16:
    the frozen 4B backbone is ~9.3 GB of weights in FP16 but ~18.6 GB in FP32,
    which cannot fit in the unified memory of an Apple Silicon machine, so FP32
    is not a usable default there. CPU stays FP32 because half-precision CPU
    kernels are slow and incomplete. Precision-sensitive parts (the scoring
    head, calibration) are cast to FP32 separately, so the backbone dtype does
    not degrade their numerics.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        if device.type == "cuda":
            return torch.bfloat16 if supports_bf16(device) else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {dtype!r}; use auto or one of {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def bucket_max_length(
    lengths: Sequence[int], buckets: Sequence[int] = (64, 128, 256, 512, 1024)
) -> int:
    """Round the longest input in a batch up to a fixed padded-length bucket.

    MPS compiles Metal kernels per tensor shape, so a serving workload where every
    request has a slightly different context length pays a compilation cost on
    nearly every call. Measured on an M3 with a small Qwen3.5 text model, the first
    call at an unseen length costs 10-200 ms more than a repeat at the same length.
    Collapsing requests onto a handful of padded widths keeps the graphs hot.

    This only accepts real token counts: paddings and truncation caps are not
    interchangeable here. Rounding a cap (e.g. ``max_context_tokens``) down would
    truncate longer inputs, so callers must apply it after tokenization.

    Returns the smallest bucket >= max(lengths), or max(lengths) itself when it
    exceeds every bucket (never truncates).
    """
    if not lengths:
        raise ValueError("at least one length is required")
    longest = max(int(length) for length in lengths)
    if longest <= 0:
        raise ValueError("lengths must be positive")
    for bucket in sorted(int(bucket) for bucket in buckets):
        if bucket >= longest:
            return bucket
    return longest


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
