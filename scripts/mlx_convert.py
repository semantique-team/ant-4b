"""Convert a Hugging Face Qwen checkpoint to an MLX model, optionally quantized.

The output directory is the value to pass to ``backbone_model`` when using
``backbone_backend="mlx"``. Quantized output must be loaded with
``quantization="mlx"`` and the same bits/group size.

Examples:
    python scripts/mlx_convert.py --hf-path models/Qwen3.5-0.8B \
        --mlx-path models/Qwen3.5-0.8B-mlx-8bit --bits 8 --group-size 64

    python scripts/mlx_convert.py --hf-path models/Qwen3.5-4B \
        --mlx-path models/Qwen3.5-4B-mlx-4bit --bits 4 --group-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Hugging Face model id or local HF folder")
    parser.add_argument("--mlx-path", required=True, help="destination MLX model folder")
    parser.add_argument("--bits", type=int, default=8, choices=(2, 4, 8))
    parser.add_argument("--group-size", dest="group_size", type=int, default=64)
    parser.add_argument("--dtype", default=None, choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="convert without weight quantization (use quantization=None when loading)",
    )
    args = parser.parse_args()

    output = Path(args.mlx_path)
    if output.exists():
        raise SystemExit(f"error: output path already exists: {output}")

    try:
        from mlx_lm import convert
    except ImportError as error:
        raise SystemExit(
            "error: mlx-lm is required; install the Apple Silicon extra with "
            "`pip install -e '.[mlx]'`"
        ) from error

    if args.group_size <= 0:
        raise SystemExit("error: --group-size must be positive")

    convert(
        hf_path=args.hf_path,
        mlx_path=str(output),
        quantize=not args.no_quantize,
        q_group_size=None if args.no_quantize else args.group_size,
        q_bits=None if args.no_quantize else args.bits,
        dtype=args.dtype,
    )
    print(f"MLX model written to {output}")
    if not args.no_quantize:
        print(
            "load it with backbone_backend='mlx', quantization='mlx', "
            f"quantization_bits={args.bits}, quantization_group_size={args.group_size}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
