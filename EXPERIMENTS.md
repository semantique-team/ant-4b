# Research log

Every experiment records: hypothesis, model configuration, dataset, training parameters,
hardware, metrics, result, conclusion. Architecture changes are never made without an entry here.

---

## E-000 — Environment and backbone verification (spec §36)

- **Date**: 2026-09-28
- **Hypothesis**: the official `Qwen/Qwen3.5-4B` implementation exposes text hidden states
  through `AutoModel` without the LM head, with all dimensions readable from the config; ANT can
  therefore be built without hard-coding any backbone constant.
- **Model configuration**: none yet (backbone inspection only).
- **Procedure**: fetched `config.json` (revision `851bf6e`) from the Hub; instantiated a tiny
  random-init `Qwen3_5Config` model on transformers 5.17.0 / torch 2.14.0 and inspected the
  module tree and a text-only forward.
- **Findings**:
  - `model_type=qwen3_5`, multimodal wrapper; `text_config.hidden_size=2560`,
    `num_hidden_layers=32` (hybrid: 3× linear attention + 1× full attention per group),
    `vocab_size=248320`, `max_position_embeddings=262144`, tied embeddings, ~4.66B BF16 params,
    ~9.3 GB on disk, not gated.
  - `AutoModel` → `Qwen3_5Model` (children: `visual`, `language_model`); no LM head loaded.
  - `language_model(input_ids, attention_mask)` → `[B, L, 2560]` hidden states, text-only OK.
  - Without `causal-conv1d` / `flash-linear-attention`, linear-attention layers use a slow
    reference fallback (correct results).
- **Hardware**: Apple M3, 16 GB RAM, macOS (inspection only).
- **Metrics**: n/a.
- **Result**: `QwenBackbone` implemented with dynamic config resolution and decoder extraction;
  verified against the real module layout.
- **Conclusion**: hypothesis confirmed; implementation constraint §34 (no LM head, no generate)
  is satisfied by construction.

## E-001 — Smoke test, mock backbone (spec §36 checklist)

- **Date**: 2026-09-28
- **Hypothesis**: the full ANT forward/backward path works and satisfies permutation invariance,
  variable N, documented duplication, and head-only gradient flow before any training.
- **Model configuration**: `ANTModel.tiny()` (MockBackbone h=64, scoring_dim=32, dot scorer,
  mean pooling).
- **Dataset**: synthetic triage, 8 examples.
- **Training parameters**: none (single backward pass).
- **Hardware**: Apple M3, 16 GB, CPU.
- **Metrics**: all 9 checks PASS — permutation max |Δp| = 0.0; gradients reach 8 head tensors,
  0 backbone tensors; N = 2,3,5,9 OK; duplicated candidates get identical probability.
- **Result**: 84/84 unit tests pass; pipeline `train → calibrate → escalate → evaluate → reload`
  works end to end.
- **Conclusion**: v0.1-alpha scaffolding is sound; proceed to real-backbone training runs.

## E-002 — Mock training sanity run

- **Date**: 2026-09-28
- **Hypothesis**: the training pipeline learns on the synthetic benchmark even with a toy
  backbone, and calibration/escalation fitting works on real outputs.
- **Model configuration**: `ANTModel.tiny()`, dot scorer, head_only.
- **Dataset**: synthetic, 300 train / 100 val / 100 test / 100 OOD, seed 7.
- **Training parameters**: 8 epochs, batch 8, lr 1e-2, AdamW + cosine.
- **Hardware**: Apple M3, 16 GB, CPU.
- **Metrics**: test accuracy 0.61 (chance ≈ 0.18 at N≈5.5), OOD accuracy 0.13 with lower
  confidence (abstention signal present); temperature calibration fitted on val.
- **Result**: pipeline-level success; absolute numbers meaningless (mock backbone).
- **Conclusion**: proceed to `Qwen3.5-4B` runs on a CUDA box; record those as E-1xx below.

## Done

- **2026-10-20 — Inference latency & Metal 8-bit quantization (lab report)**: full write-up in
  [`docs/experiments/2026-10-20-inference-latency-and-metal-quantization.md`](docs/experiments/2026-10-20-inference-latency-and-metal-quantization.md)
  — working config: fp16 plain + fused encode, ≈8× on short inputs; Metal 8-bit OFF (accuracy
  gate FAIL, pre-quantized reload unusable on MPS).

---

## Planned experiments (templates to fill with real-backbone runs)

### E-101 — Scorer comparison (spec §28)

Command: `python scripts/benchmark.py --backbone models/Qwen3.5-4B --device cuda`

| Model | Accuracy | NLL | Brier | ECE | Latency | VRAM |
|---|---|---|---|---|---|---|
| Frozen Qwen + Dot | | | | | | |
| Frozen Qwen + Cosine | | | | | | |
| Frozen Qwen + Bilinear | | | | | | |
| Frozen Qwen + MLP | | | | | | |
| LoRA + best scorer | | | | | | |

- **Hypothesis**: (state before running) —
- **Training parameters**: —
- **Hardware**: —
- **Conclusion**: —

### E-102 — Candidate-count scaling (spec §19)

Command: `scripts/benchmark.py … --scaling-sizes 2 4 8 16 32 64 128`
Accuracy / latency / throughput / ECE vs N; compare slope against a fixed-classification-head
baseline. **Hypothesis**: scoring latency grows ~O(N) in cheap dot products, accuracy degrades
gracefully with generic fillers.

### E-103 — Batch inference with cached candidates (spec §24)

Command: `scripts/benchmark.py … --batch-sizes 1 8 32 64 128`
Per-example latency at fixed candidate set, cache on. **Hypothesis**: amortized cost tends to
one context encoding per state.

### E-104 — Joint encoding, Mode B (spec §7)

Train a `CONTEXT_JOINT_TEMPLATE` variant (context+candidate encoded together, candidate-
conditioned pooling) and compare against Mode A on latency / accuracy / ECE.
**Hypothesis**: Mode B may win accuracy on hard negatives at N× context-encoding cost, losing
the cache advantage.

### E-105 — Multilingual calibration (spec §21)

Evaluate the E-101 winner per language (en/it/es/fr/de) on the test split; check ECE drift
across languages. **Hypothesis**: Qwen's multilingual representations keep calibration
approximately stable, with mild degradation for it/de.
