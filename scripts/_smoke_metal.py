"""One-shot 4B run on Apple Metal: benchmark the 8-bit backbone, then save it.

The load/quantize step is the only memory-heavy part of the pipeline (the
full-precision shards are read before they can be packed); everything after it
runs at ~3.8 GB. Loading once here and saving the quantized weights means
future runs skip the risky peak entirely.

    python scripts/_smoke_metal.py --quantization metal --bits 8

`--no-save` skips writing models/Qwen3.5-4B-metal-8bit.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from ant.backbone.qwen import QwenBackbone
from ant.utils import synchronize

SAVE_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B-metal-8bit"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", default="metal")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", dest="group_size", type=int, default=64)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512, 1024])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_args()

    quantization = None if args.quantization in ("none", "off", "") else args.quantization
    device = torch.device("mps")
    started = time.perf_counter()
    backbone = QwenBackbone(
        "Qwen/Qwen3.5-4B",
        torch_dtype="float16",
        device="mps",
        quantization=quantization,
        quantization_bits=args.bits,
        quantization_group_size=args.group_size,
        linear_attention_chunk_size=args.chunk_size,
    )
    synchronize(device)
    log(f"load_seconds: {time.perf_counter() - started:.1f}")

    parameters = list(backbone.model.parameters())
    packed = sum(1 for m in backbone.model.modules() if type(m).__name__ == "MetalLinear")
    log(f"sample_param: dtype={parameters[0].dtype} device={parameters[0].device}")
    log(f"mps_allocated_gb: {torch.mps.current_allocated_memory() / 1e9:.2f}")
    log(f"metal_linear: {packed}")
    log(
        f"quantization: {quantization} bits={args.bits} group={args.group_size} "
        f"chunk={args.chunk_size}"
    )

    filler = "The customer reports that the application stopped responding after the latest update. "
    for target in args.lengths:
        text = (filler * 200)[: int(target * 3.4)]
        timings: list[float] = []
        for _ in range(args.iters + 1):
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                output = backbone.encode([text], batch_size=1)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
        tokens = int(output.attention_mask.sum())
        warm = timings[1:] or timings
        log(
            f"tokens={tokens:5d} cold_ms={timings[0]:8.0f} "
            f"warm_ms={statistics.median(warm):8.0f} "
            f"warm_min_ms={min(warm):8.0f} mps_gb={torch.mps.current_allocated_memory() / 1e9:.2f}"
        )

    if args.save:
        log(f"saving pre-quantized checkpoint to {SAVE_DIR} ...")
        save_started = time.perf_counter()
        backbone.model.save_pretrained(SAVE_DIR)
        backbone.tokenizer.save_pretrained(SAVE_DIR)
        log(f"saved in {time.perf_counter() - save_started:.1f}s")
        for path in sorted(SAVE_DIR.glob("*")):
            log(f"  {path.name}: {path.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
