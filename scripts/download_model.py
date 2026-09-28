"""Download a Qwen3.5-4B checkpoint from the Hugging Face Hub.

Example:
    python scripts/download_model.py --model Qwen/Qwen3.5-4B --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "*.txt",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="repo id or local name")
    parser.add_argument("--dest", default=None, help="local directory (default: models/<name>)")
    parser.add_argument("--revision", default=None, help="specific commit/branch")
    parser.add_argument("--verify", action="store_true", help="load config + tokenizer to confirm integrity")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest) if args.dest else Path("models") / args.model.split("/")[-1]
    path = snapshot_download(
        repo_id=args.model,
        local_dir=str(dest),
        revision=args.revision,
        allow_patterns=ALLOW_PATTERNS,
    )
    size_gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"downloaded {args.model} -> {path} ({size_gb:.2f} GB)")

    if args.verify:
        from ant.backbone.qwen import resolve_hidden_size, resolve_num_layers, resolve_text_config
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(str(path))
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        text_config = resolve_text_config(config)
        print(f"model_type        : {getattr(config, 'model_type', '?')}")
        print(f"architectures     : {getattr(config, 'architectures', '?')}")
        print(f"hidden_size       : {resolve_hidden_size(config)}")
        print(f"num_layers        : {resolve_num_layers(config)}")
        print(f"vocab_size        : {getattr(text_config, 'vocab_size', '?')}")
        print(f"max_positions     : {getattr(text_config, 'max_position_embeddings', '?')}")
        print(f"tokenizer entries : {len(tokenizer)}")
        print("verification OK")

    print(f"\nuse it with:  ANTConfig(backbone_model=\"{path}\")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
