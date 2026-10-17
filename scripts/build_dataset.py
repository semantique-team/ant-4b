"""Write the multi-use-case ANT dataset to data/ant-multitask/*.jsonl.

    python scripts/build_dataset.py [--out data/ant-multitask] [--seed 7]

Produces train/val/test (held-out *situations* of the training use cases) and
ood (held-out *use cases*), plus a manifest with the counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ant.data.jsonl import write_jsonl
from ant.data.multitask import SPLITS, build_multitask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/ant-multitask")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = build_multitask(seed=args.seed)
    out = Path(args.out)
    manifest: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        tagged = [row for row in rows if row.split == split]
        write_jsonl(out / f"{split}.jsonl", (row.example for row in tagged))
        manifest[split] = dict(Counter(row.use_case for row in tagged))
        print(f"{split}: {len(tagged)} examples, {len(manifest[split])} use cases")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
