r"""Where the inference latency actually goes, phase by phase.

Mock backbone (offline, no checkpoint, no download):
    python scripts/profile_inference.py --iters 1 --warmup 0 --lengths 32 128

Real checkpoint, optionally quantized on Apple Metal:
    python scripts/profile_inference.py --checkpoint runs/ant-4b-2026-10-17 \
        --backbone models/Qwen3.5-4B --quantization metal --bits 8 \
        --group-size 64 --chunk-size 2048
"""

from __future__ import annotations

import argparse
import gc
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import torch

from ant.config import ANTConfig
from ant.inference.cache import CandidateEmbeddingCache
from ant.inference.predictor import ANTPredictor
from ant.model import ANTModel
from ant.utils import resolve_dtype, synchronize

T = TypeVar("T")

PHASES = (
    "tokenize",
    "context_only",
    "fused_encode",
    "candidate_encode_warm",
    "score_softmax",
)

QUESTION = "Which department should handle this case?"
CANDIDATES = ("Billing", "Technical Support", "Sales", "Legal", "Engineering", "Facilities")
FILLER = (
    "The customer reports that the application stopped responding after the latest update "
    "and asks for a written update on the status of the open ticket. "
)


# -- environment ---------------------------------------------------------------


def process_rss_gb() -> float:
    """Peak resident set size in GB (psutil is not a dependency)."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return max_rss / 1e9 if sys.platform == "darwin" else max_rss / 1024**2


def mps_memory() -> tuple[str, str]:
    """(recommended max, current allocated) in GB, or ("n/a", "n/a") off MPS."""
    try:
        if torch.backends.mps.is_available():
            return (
                f"{torch.mps.recommended_max_memory() / 1e9:.2f} GB",
                f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB",
            )
    except Exception:
        pass
    return "n/a", "n/a"


def _swap_to_gb(value: str, unit: str) -> float:
    return float(value) / 1024.0 if unit == "M" else float(value)


def swap_usage() -> str:
    """macOS swap usage from ``sysctl -n vm.swapusage``; "n/a" elsewhere."""
    if platform.system() != "Darwin":
        return "n/a"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "n/a"
    match = re.search(
        r"total = ([\d.,]+)([MG])\s+used = ([\d.,]+)([MG])", result.stdout.strip()
    )
    if match is None:
        return result.stdout.strip() or "n/a"
    total, total_unit, used, used_unit = match.groups()
    return (
        f"{_swap_to_gb(used.replace(',', '.'), used_unit):.2f} GB used / "
        f"{_swap_to_gb(total.replace(',', '.'), total_unit):.2f} GB total"
    )


def environment_lines(device: torch.device, dtype: torch.dtype) -> list[str]:
    recommended, allocated = mps_memory()
    return [
        f"torch               : {torch.__version__}",
        f"device              : {device}",
        f"resolved dtype      : {dtype}",
        f"MPS available       : {torch.backends.mps.is_available()}",
        f"MPS recommended max : {recommended}",
        f"MPS allocated       : {allocated}",
        f"process RSS (peak)  : {process_rss_gb():.2f} GB",
        f"macOS swap          : {swap_usage()}",
    ]


def model_lines(predictor: ANTPredictor) -> list[str]:
    model = predictor.model
    config = model.config
    backbone = model.backbone
    label = getattr(backbone, "model_id", None) or config.backbone_model
    if getattr(config, "backbone_backend", "torch") == "mlx":
        summary = backbone.summary()
        return [
            f"backbone_model  : {label}",
            f"backend         : mlx",
            f"hidden_size     : {summary['hidden_size']}",
            f"num_layers      : {summary['num_layers']}",
            f"quantization    : {summary.get('quantization') or 'off'}",
            f"torch head only : {sum(p.numel() for p in model.head_parameters()) / 1e6:.2f}M",
        ]
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head_parameters())
    dtypes = ", ".join(sorted({str(p.dtype) for p in model.parameters()}))
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    quantization = getattr(config, "quantization", None)
    if quantization:
        quantization_line = (
            f"{quantization} (bits={getattr(config, 'quantization_bits', '?')}, "
            f"group_size={getattr(config, 'quantization_group_size', '?')})"
        )
    else:
        quantization_line = "off"
    chunk_size = getattr(config, "linear_attention_chunk_size", None)

    return [
        f"backbone_model  : {label}",
        f"parameters      : {total / 1e6:.1f}M ({head / 1e6:.2f}M in the head)",
        f"parameter dtypes: {dtypes}",
        f"parameter bytes : {weight_bytes / 1e9:.3f} GB ({weight_bytes / 1e6:.1f} MB)",
        f"quantization    : {quantization_line}",
        f"chunk size      : {chunk_size if chunk_size is not None else 'off'}",
    ]


# -- model construction --------------------------------------------------------


def build_qwen_backbone(model_id: str, args: argparse.Namespace):
    if args.backend == "mlx":
        from ant.backbone.mlx import MlxBackbone

        return MlxBackbone(
            model_id,
            torch_dtype=args.dtype,
            device=args.device,
            quantization=args.quantization,
            quantization_bits=args.bits,
            quantization_group_size=args.group_size,
        )
    from ant.backbone.qwen import QwenBackbone

    return QwenBackbone(
        model_id,
        torch_dtype=args.dtype,
        device=args.device,
        quantization=args.quantization,
        quantization_bits=args.bits,
        quantization_group_size=args.group_size,
        linear_attention_chunk_size=args.chunk_size,
    )


def load_predictor(args: argparse.Namespace) -> ANTPredictor:
    """Load the checkpoint when given, otherwise fall back to the mock backbone."""
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if checkpoint is not None and not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint is not a directory: {checkpoint}")
    if checkpoint is not None and not (checkpoint / "ant_head.pt").exists():
        raise FileNotFoundError(f"{checkpoint / 'ant_head.pt'} is missing")

    saved_backbone = None
    saved_backend = None
    if checkpoint is not None and (checkpoint / "config.json").exists():
        saved_config = ANTConfig.load(checkpoint / "config.json")
        saved_backbone = saved_config.backbone_model
        saved_backend = saved_config.backbone_backend

    model_id = args.backbone or saved_backbone
    args.backend = args.backend or saved_backend or "torch"
    backbone = build_qwen_backbone(model_id, args) if model_id not in (None, "mock") else None

    if checkpoint is not None:
        predictor = ANTPredictor.load(checkpoint, backbone=backbone)
    elif backbone is not None:
        config = ANTConfig(
            backbone_model=model_id,
            backbone_backend=args.backend,
            torch_dtype=args.dtype,
            device=args.device,
            quantization=args.quantization,
            quantization_bits=args.bits,
            quantization_group_size=args.group_size,
            linear_attention_chunk_size=args.chunk_size,
        )
        predictor = ANTPredictor(ANTModel(config, backbone=backbone))
    else:
        predictor = ANTPredictor(ANTModel.tiny())

    if predictor.candidate_cache is None:
        predictor.candidate_cache = CandidateEmbeddingCache()

    mock = getattr(predictor.model.config, "backbone_model", "") == "mock"
    if mock and (args.quantization is not None or args.chunk_size is not None):
        print(
            "warning: --quantization/--chunk-size are ignored on the mock backbone",
            file=sys.stderr,
        )
    return predictor


# -- state synthesis -----------------------------------------------------------


def tokenize_once(text: str, tokenizer, max_length: int) -> None:
    """Mirror the tokenizer call QwenBackbone.encode makes for a batch of one."""
    if tokenizer is None:
        text.split()
    else:
        tokenizer(
            [text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def synthesize_state(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Build a state of roughly ``target_tokens`` tokens and report the real count."""
    if target_tokens <= 0:
        raise ValueError("target length must be positive")
    if tokenizer is None:
        filler = FILLER.split()
        words: list[str] = []
        while len(words) < target_tokens:
            words.extend(filler)
        text = " ".join(words[:target_tokens])
        return text, count_tokens(text, tokenizer)

    ids: list[int] = []
    while len(ids) < target_tokens:
        ids.extend(tokenizer(FILLER, add_special_tokens=False)["input_ids"])
    text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)
    return text, count_tokens(text, tokenizer)


# -- timing --------------------------------------------------------------------


def measure(device: torch.device, fn: Callable[[], T]) -> tuple[float, T]:
    """Wall time of ``fn`` in ms, with the device synchronised on both ends."""
    synchronize(device)
    started = time.perf_counter()
    value = fn()
    synchronize(device)
    return (time.perf_counter() - started) * 1e3, value


def profile_length(
    predictor: ANTPredictor,
    length: int,
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[int, dict[str, float], float]:
    """Median ms per phase for one context length: (actual tokens, phases, total)."""
    model = predictor.model
    cache = predictor.candidate_cache
    if cache is None:
        raise RuntimeError("profile_length needs a candidate cache")
    tokenizer = getattr(model.backbone, "tokenizer", None)
    max_length = model.config.max_context_tokens
    batch_size = model.config.encode_batch_size
    state, achieved = synthesize_state(length, tokenizer)
    candidates = list(CANDIDATES)

    samples: dict[str, list[float]] = {name: [] for name in PHASES}
    total_samples: list[float] = []

    # One untimed pass over every code path primes the allocator and any lazy
    # kernels, so --warmup 0 reports steady-state phases instead of first-touch
    # page faults on the very first candidate encode.
    with torch.inference_mode():
        model.encode_contexts([state], [QUESTION])
        model.encode_candidates(candidates)
        predictor.predict(state, QUESTION, candidates)
    cache.clear()

    # A cycle collection landing inside one phase would attribute its cost to
    # that phase; collect once up front and keep the collector out of the loop.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(warmup + iters):
            elapsed, _ = measure(
                device, lambda: tokenize_once(state, tokenizer, max_length)
            )
            samples["tokenize"].append(elapsed)

            with torch.inference_mode():
                elapsed, _ = measure(
                    device, lambda: model.encode_contexts([state], [QUESTION])
                )
            samples["context_only"].append(elapsed)

            # The serving path on a cache miss: one pass for the context and
            # every candidate together, instead of two passes over the weights.
            with torch.inference_mode():
                elapsed, (z_context, z_candidates) = measure(
                    device,
                    lambda: model.encode_state_and_candidates(state, QUESTION, candidates),
                )
            samples["fused_encode"].append(elapsed)

            cache.clear()
            cache.store(candidates, z_candidates)
            with torch.inference_mode():
                elapsed, _ = measure(
                    device, lambda: cache.embeddings(model, candidates, batch_size=batch_size)
                )
            samples["candidate_encode_warm"].append(elapsed)

            def score_and_softmax() -> torch.Tensor:
                scores = model.score_embeddings(z_context, z_candidates.unsqueeze(0))[0]
                return torch.softmax(scores.float() / predictor.temperature, dim=-1)

            with torch.inference_mode():
                elapsed, _ = measure(device, score_and_softmax)
            samples["score_softmax"].append(elapsed)

            with torch.inference_mode():
                elapsed, _ = measure(
                    device, lambda: predictor.predict(state, QUESTION, candidates)
                )
            total_samples.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    # the slices below drop the warm-up runs: first-touch allocation dominates them
    medians = {name: statistics.median(values[warmup:]) for name, values in samples.items()}
    return achieved, medians, statistics.median(total_samples[warmup:])


def render_timing(
    length: int, achieved: int, medians: dict[str, float], total: float
) -> list[str]:
    phase_total = sum(medians.values())
    lines = [
        f"== phase timing: target {length} tokens (actual {achieved}) ==",
        f"| {'phase':<22} | {'ms':>9} | {'share':>7} |",
        f"|{'-' * 24}|{'-' * 11}|{'-' * 9}|",
    ]
    for name in PHASES:
        milliseconds = medians[name]
        share = 100.0 * milliseconds / phase_total if phase_total > 0 else 0.0
        lines.append(f"| {name:<22} | {milliseconds:9.3f} | {share:6.1f}% |")
    lines.append(f"| {'total_predict':<22} | {total:9.3f} | {'n/a':>7} |")
    return lines


def verdict(medians: dict[str, float]) -> str:
    phase_total = sum(medians.values())
    name = max(PHASES, key=lambda phase: medians[phase])
    share = 100.0 * medians[name] / phase_total if phase_total > 0 else 0.0
    return f"VERDICT: dominant phase: {name} ({share:.0f}%)"


# -- CLI -----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=None, help="head checkpoint directory")
    parser.add_argument("--backbone", default=None, help="HF repo id or local backbone path")
    parser.add_argument("--backend", choices=("torch", "mlx"), default=None, help="default: checkpoint config, else torch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None, choices=(None, "none", "metal", "mlx"), help="default: off")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", dest="group_size", type=int, default=64)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 128, 512, 1024])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.quantization == "none":
        args.quantization = None

    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if any(length <= 0 for length in args.lengths):
        parser.error("--lengths values must be positive")
    if args.quantization is not None:
        if args.bits not in (2, 4, 8):
            parser.error("--bits must be 2, 4 or 8")
        if args.group_size <= 0:
            parser.error("--group-size must be positive")
    if args.chunk_size is not None and args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    try:
        predictor = load_predictor(args)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    model = predictor.model
    model.eval()
    device = next(model.parameters()).device
    dtype = resolve_dtype(args.dtype, device)

    print("== environment ==")
    for line in environment_lines(device, dtype):
        print(line)
    print("\n== model ==")
    for line in model_lines(predictor):
        print(line)
    if getattr(model.backbone, "max_tokens", None) is not None:
        print(
            f"\nnote: the mock backbone truncates at {model.backbone.max_tokens} tokens, "
            "so context cost flattens above that length"
        )

    print(
        "\n== phase timing (median ms over "
        f"{args.iters} iterations after {args.warmup} warm-up) =="
    )
    for length in args.lengths:
        achieved, medians, total = profile_length(
            predictor, length, iters=args.iters, warmup=args.warmup, device=device
        )
        print()
        for line in render_timing(length, achieved, medians, total):
            print(line)
        print(verdict(medians))
    print(
        "\nshare = phase median / sum of the phase medians; total_predict is the "
        "end-to-end call and has no share. context_only is the fully warm-cache "
        "path; fused_encode is the cache-miss path that predict() actually takes, "
        "encoding the context and every candidate in one pass."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
