"""ANT-4B smoke test: the full pre-training verification checklist.

Runs against a mock backbone by default (no download), or against a real
checkpoint:  python scripts/smoke_test.py --model models/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from ant.data.synthetic import generate_examples, SyntheticConfig
from ant.model import ANTModel
from ant.training.modes import set_training_mode
from ant.utils import softmax_np

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF repo id or local path; omit for mock backbone")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    if args.model:
        from ant.config import ANTConfig

        config = ANTConfig(backbone_model=args.model, device=args.device, torch_dtype=args.dtype)
        model = ANTModel(config)
    else:
        model = ANTModel.tiny()
    model.eval()

    backbone = model.backbone
    summary = backbone.summary() if hasattr(backbone, "summary") else {"hidden_size": backbone.hidden_size}
    print("== backbone ==")
    for key, value in summary.items():
        print(f"  {key:22}: {value}")
    record(
        "config read dynamically",
        summary.get("hidden_size", 0) > 0,
        f"hidden_size={summary.get('hidden_size')}",
    )

    examples = generate_examples(8, split="smoke", config=SyntheticConfig(seed=1))
    record("synthetic examples generated", len(examples) == 8)

    with torch.no_grad():
        output = backbone.encode([e.state for e in examples][:4])
    ok = (
        output.hidden_states.shape[0] == 4
        and output.hidden_states.shape[2] == backbone.hidden_size
        and bool(torch.isfinite(output.hidden_states.float()).all())
    )
    record(
        "hidden states extracted",
        ok,
        f"shape={tuple(output.hidden_states.shape)} (no LM head, no generate())",
    )

    loss = model.loss(examples)
    record("forward pass + cross-entropy", bool(torch.isfinite(loss)))

    set_training_mode(model, "head_only")
    loss = model.loss(examples[:4])
    loss.backward()
    head_grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    backbone_grads = [
        n
        for n, p in backbone.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    ]
    record(
        "gradients reach the head, not the frozen backbone",
        len(head_grads) > 0 and len(backbone_grads) == 0,
        f"{len(head_grads)} head tensors with grads",
    )

    example = examples[0]
    with torch.no_grad():
        base = model.score_candidates(example.state, example.question, example.candidates)
        base_probs = softmax_np(base.numpy())
        order = [3, 0, 2, 1] if len(example.candidates) == 4 else list(reversed(range(len(example.candidates))))
        shuffled = [example.candidates[i] for i in order]
        permuted = model.score_candidates(example.state, example.question, shuffled)
        permuted_probs = softmax_np(permuted.numpy())
    mapped = np.array([permuted_probs[shuffled.index(c)] for c in example.candidates])
    delta = float(np.abs(mapped - base_probs).max())
    record("candidate permutation invariance", delta < 1e-5, f"max |Δp| = {delta:.2e}")

    with torch.no_grad():
        for n in (2, 3, 5, 9):
            subset = list(example.candidates[:n])
            subset += [f"Team {i}" for i in range(n - len(subset))]
            scores = model.score_candidates(example.state, example.question, subset)
            assert scores.shape[0] == n
    record("variable candidate count", True, "N=2,3,5,9 OK")

    duplicated = [example.candidates[0], example.candidates[0], example.candidates[1]]
    with torch.no_grad():
        dup_probs = softmax_np(model.score_candidates(example.state, example.question, duplicated).numpy())
    identical = abs(float(dup_probs[0]) - float(dup_probs[1])) < 1e-6
    detail = (
        f"two copies of {example.candidates[0]!r} get identical probability "
        f"({dup_probs[0]:.3f} each), splitting the label mass"
    )
    record("candidate duplication documented", identical, detail)

    states = [e.state for e in examples]
    questions = [e.question for e in examples]
    backbone.eval()
    start = time.perf_counter()
    with torch.no_grad():
        model.encode_contexts(states, questions, batch_size=4)
    context_ms = (time.perf_counter() - start) / len(states) * 1e3
    start = time.perf_counter()
    with torch.no_grad():
        model.encode_candidates([e.correct for e in examples], batch_size=4)
    candidate_ms = (time.perf_counter() - start) / len(examples) * 1e3
    record("latency measured", True, f"context {context_ms:.1f} ms, candidate {candidate_ms:.1f} ms each")

    print()
    failed = [name for name, passed, _ in CHECKS if not passed]
    if failed:
        print(f"{len(failed)} check(s) failed: {failed}")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
