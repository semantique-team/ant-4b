# ANT-4B — Adaptive Neural Triage

> The lightweight `Qwen3.5-0.8B` variant lives in the sibling repository `ant-08b`; this
> repository is the 4B reference. Both use MLX for local Apple Silicon serving and Colab for
> real-backbone training.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen](https://img.shields.io/badge/Backbone-Qwen3.5--4B-6E46AE)](https://huggingface.co/Qwen/Qwen3.5-4B)
[![License](https://img.shields.io/badge/License-Apache--2.0-3DA639)](LICENSE)

An open-source **System 1 decision model** built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).

ANT does not generate natural-language answers. It turns a transformer backbone into a
**candidate-scoring engine**:

```
state + question          candidates (any number, any text)
        ↓                              ↓
   ContextEncoder                 CandidateEncoder
        ↓                              ↓
     z_context                    z_candidate_i
        └────────────┬─────────────────┘
                     ↓
              scoring function  →  s_i = f(z_context, z_candidate_i)
                     ↓
              softmax(s / T)  →  P(candidate_i | state, question)
                     ↓
              confidence metrics → execute / escalate
```

![ANT-4B inference path: state, question and candidates encoded by two encoders sharing one frozen Qwen3.5-4B decoder, scored per candidate, calibrated, and routed to execute or escalate](docs/assets/ant-architecture.svg)

The primitive is `score(state, question, candidate) -> scalar`; the choice primitive is a
calibrated probability rule `P(candidate_i | state, question) = softmax(s / T)`, applied outside
the model by `ANTPredictor.predict(...)` together with confidence metrics and the
execute/escalate policy. There is no LM head and no `model.generate(...)` anywhere in the
inference path.

- **Variable candidate count**: N = 2 to 100+; candidates are semantic objects (labels, sentences,
  serialized JSON), not integer class IDs.
- **Permutation invariant**: reordering candidates provably cannot change their probabilities.
- **Cacheable**: static candidate sets are embedded once; inference then costs one context encoding
  plus N dot products.
- **Calibrated**: temperature/affine/MLP calibration fitted on a held-out split, with ECE, Brier,
  NLL and reliability diagrams as first-class metrics.
- **Escalation-first**: execute/escalate decisions come from a policy fitted on validation
  risk/coverage data, not a hard-coded threshold, exposing everything an external System 2 router
  needs.

## How it relates to other System One models

- [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model)) (Typesafe AI, proprietary,
  RLCD-trained) defined the category: state + typed questions in, typed answers with
  probabilities out, zero-shot. An independent evaluation
  ([arXiv:2609.37647](https://arxiv.org/abs/2609.37647)) confirms its choice probabilities
  are calibrated and unchanged under option rotation.
- [Laya](https://github.com/NandhaKishorM/laya) (ConvAI Innovations, open-source Apache-2.0,
  [model card](https://huggingface.co/convaiinnovations/laya)) is the open counterpart:
  zero-shot on typed-decisions sits at chance (0.36 vs 0.318 random) while the fine-tuned
  [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions)
  checkpoint reaches 0.766 vs Jev's published 0.727 — a base to specialise, not a
  zero-shot engine.
- [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
  publishes a recipe for fine-tuning Qwen3.5 models into decision models with LoRA —
  a Qwen3.5-4B decision model trains on 8 GB VRAM.

ANT takes a different route, documented in [docs/research-notes.md](docs/research-notes.md):
a bi-encoder scores candidates with one dot product each once embeddings are cached
(decoder-as-encoder after [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) and
[LLM2Vec](https://arxiv.org/abs/2404.05961)), and calibration plus the risk-coverage
escalation policy are fitted post-hoc ([Guo et al. 2017](https://arxiv.org/abs/1706.04599),
[Geifman & El-Yaniv 2017](https://arxiv.org/abs/1705.08500)) rather than baked into
training — keeping the permutation-invariance guarantee and the candidate cache by
construction, and exposing to a System 2 router ([FrugalGPT](https://arxiv.org/abs/2305.05176),
[RouteLLM](https://arxiv.org/abs/2406.18665)) exactly what it needs to route.

## Install

```bash
git clone <this-repo> && cd ant-4b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- Runtime: `torch`, `transformers>=4.57` (native `qwen3_5` support), `numpy`
- Extras: `train` (PEFT/LoRA, accelerate), `hub` (HF CLI + fast transfers), `viz` (matplotlib),
  `mlx` (Apple Silicon MLX inference backend + MLX quantization), `metal` (Apple Silicon 8-bit quantization)

## Download the model from Hugging Face — step by step

`Qwen/Qwen3.5-4B` is public (not gated), Apache-2.0, ~**9.3 GB** in BF16 safetensors (2 shards).
You do not need an account; logging in just makes downloads faster and more stable.

```bash
# 1. Get the HF CLI (inside your venv)
pip install -U "huggingface_hub[cli]" hf_transfer

# 2. (optional, recommended) authenticate — token from https://huggingface.co/settings/tokens
hf auth login

# 3. Download into ./models (resumable; re-run the same command if interrupted)
export HF_HUB_ENABLE_HF_TRANSFER=1   # much faster transfers
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B

#    experimental variant, if you want the base model too:
hf download Qwen/Qwen3.5-4B-Base --local-dir models/Qwen3.5-4B-Base

# 4. Verify integrity (loads config + tokenizer, prints the true architecture)
python scripts/download_model.py --model models/Qwen3.5-4B --verify

# 5. Full smoke test against the real backbone (forward, gradients, invariance)
python scripts/smoke_test.py --model models/Qwen3.5-4B
```

Notes:

- The legacy command `huggingface-cli download ...` works identically on older versions.
- No local copy? Just pass the repo id (`Qwen/Qwen3.5-4B`) and `from_pretrained` downloads into
  the HF cache (`~/.cache/huggingface`) automatically.
- Air-gapped box? Set `HF_HUB_OFFLINE=1` after downloading, and point
  `ANTConfig(backbone_model="models/Qwen3.5-4B")` at the local folder.
- On CUDA, install the fused linear-attention kernels for a large speedup:
  `pip install causal-conv1d flash-linear-attention` (transformers falls back to a slow reference
  implementation without them).

### Hardware

| Setup | Feasibility |
|---|---|
| CUDA GPU ≥ 16 GB | recommended; BF16 weights ~9.4 GB + head + activations |
| Apple Silicon 32 GB | works (MPS, FP16, small batches) |
| Apple Silicon 16 GB | works with FP16: 8.4 GB resident, batch 1 |
| CPU only | not recommended for the 4B backbone; use the mock backbone for development |

MPS has no native BF16, so PyTorch emulates it and the forward pass runs several times slower than
FP16 with identical numerics: on the real backbone, FP16 and BF16 plain runs agree to 6 decimal
places on the mean absolute hidden state and both stay NaN-free. `resolve_dtype("auto", mps)`
therefore returns FP16, and the quantized load path passes the dtype through explicitly — without
that the checkpoint's own BF16 declaration wins.

### Inference latency, Apple Silicon (FP16, batch 1)

Warm `backbone.encode` on an M3 / 16 GB, after the improvements below:

| tokens | warm encode |
|---|---|
| 14 | 0.37 s |
| 53 | 0.37 s |
| 210 | 1.4 s |
| 418 | 2.9 s |

versus ~13 s per prediction on the unmodified path. What made the difference: one fused backbone
pass for the context and every candidate (verified `max |Δp| = 0.0` against the two-pass path),
`torch.inference_mode()` on all inference entries, and length bucketing (64/128/256/512/1024) so
MPS reuses its compiled kernels instead of recompiling per distinct shape. A first `predict` after
load still pays ~1.7 s of first-touch compilation for the template shapes. Long inputs remain
bounded by the 24 GatedDeltaNet layers running the reference PyTorch path — CUDA/Triton-only fused
kernels — so quantization-style memory fixes cannot remove that floor.

### 8-bit Metal quantization — NOT usable on this stack yet

Transformers 5.17 ships a native MPS quantizer (`quantization="metal"`, `quantization_bits=8`,
`quantization_group_size=64`, gated by the `metal` extra). The config plumbing, unit tests and the
accuracy gate are all in place and green, but **the quantizer itself is numerically broken on the
real model with this stack** (transformers 5.17.0, kernels 0.16.2, current PyTorch MPS):

- FP16 + 8 bits on the 4B backbone: every hidden state is exactly **zero** (5.08 GB resident).
- BF16 + 8 bits: hidden states are **NaN** (3.80 GB resident).
- FP16 / BF16 without quantization: sane, and equal to each other.
- An identical tiny locally-built Qwen3.5 TextModel quantizes **correctly** (max hidden delta 0.05
  vs FP16), including a BF16-on-disk / FP16-requested dtype cast — so this is not our loading code;
  it is size/shape-specific behavior of the pinned kernel.
- transformers pins `kernels-community/mlx-quantization-metal-kernels` **version 1** (version 2
  exists but is ignored by this release). Re-loading a `save_pretrained()`-saved quantized
  checkpoint additionally **segfaults** in PyTorch's MPS copy kernel, with or without
  `device_map="mps"`.

`scripts/compare_quantization.py` is the gate that caught this — reference FP16 scored accuracy
0.725 / ECE 0.155 on 40 test rows while the quantized arm collapsed to 0.325 with NaN deltas, so
the gate failed at Δaccuracy = 0.4 exactly as designed. `models/Qwen3.5-4B-metal-8bit/` contains
the resulting packed `U32` checkpoint: valid bytes, not loadable on this stack — keep for a future
fixed transformers release or delete it. Defaults keep `quantization=None`; when a fixed release
lands, re-run the gate before trusting any quantized numbers.

### MLX backend and MLX quantization (Apple Silicon)

`backbone_backend="mlx"` runs the frozen Qwen encoder through Apple's `mlx-lm` and keeps the
existing torch ANT head/checkpoint format. This is the recommended local path for the hybrid
GatedDeltaNet layers, which fall back to a slow reference implementation under PyTorch MPS.

```bash
pip install -e ".[mlx]"

# Convert an HF checkpoint to MLX; use --bits 8 (conservative) or --bits 4 (lighter).
python scripts/mlx_convert.py --hf-path models/Qwen3.5-4B \
  --mlx-path models/Qwen3.5-4B-mlx-8bit --bits 8 --group-size 64
```

```python
ant = ANTPredictor.from_config(
    ANTConfig(
        backbone_model="models/Qwen3.5-4B-mlx-8bit",
        backbone_backend="mlx",
        quantization="mlx",
        quantization_bits=8,
        quantization_group_size=64,
    )
)
```

The MLX backend is an **inference** backend: train heads with `--device cuda` on Colab, then
profile the returned checkpoint locally. `quantization="mlx"` accepts only an already-converted
MLX checkpoint whose embedded bits/group size match the request; it never silently requantizes
weights. Always compare accuracy, ECE and argmax flips against an unquantized MLX arm through
`scripts/compare_quantization.py` before shipping a quantized model.

## Quickstart

```python
from ant import ANTConfig
from ant.inference.predictor import ANTPredictor

ant = ANTPredictor.from_config(
    ANTConfig(backbone_model="models/Qwen3.5-4B", device="cuda", torch_dtype="bfloat16")
)

result = ant.predict(
    state="Customer has been charged twice for the same invoice. Refund requested.",
    question="Which department should handle this case?",
    candidates=["Billing", "Technical Support", "Sales", "Legal"],
)
print(result.to_dict())
# {'choice': ..., 'action': ..., 'probabilities': {...}, 'confidence': ...,
#  'margin': ..., 'entropy': ..., 'normalized_entropy': ..., 'scores': {...}}
```

Low-level scoring for ranking/routing:

```python
scores = ant.score_candidates(state, question, candidates)   # raw, input order
```

See `examples/inference.py` for caching, escalation policies and batch prediction.

## Playground and serving

The FastAPI adapter and Vite playground are maintained in the sibling
[`playground`](../playground) repository. It consumes this package and one
or more checkpoint directories; this repository intentionally contains no HTTP or
web application runtime.

## Training (mode 1: frozen Qwen + trainable head)

Training is a cloud workflow: do not run the real-backbone training loop locally. Use the
Colab guide for the 0.8B variant or a CUDA box for 4B, then bring back only the small head
checkpoint and profile it locally with MLX.

```bash
# CPU end-to-end sanity run on the mock backbone (~seconds)
python scripts/train.py --backbone mock --out runs/mock-dot

# Real backbone (CUDA/Colab)
python scripts/train.py --backbone models/Qwen3.5-4B --device cuda --out runs/qwen-dot

# LoRA mode (requires ant-4b[train], CUDA/Colab)
python scripts/train.py --backbone models/Qwen3.5-4B --mode lora --device cuda --out runs/qwen-lora
```

### Training on the multi-use-case dataset

`ant/data/usecases.py` holds hand-written seed situations for 16 use cases (support
routing, incident severity, sentiment, refund approval, email triage, shopper intent,
moderation, expense category, bug triage, lead quality, document type, transaction
risk, IT helpdesk, agent next step, plus Italian routing and sentiment) and 3 held-out
use cases (PR review, shipping method, leave request) for the OOD split.
`scripts/build_dataset.py` expands them into `data/ant-multitask/{train,val,test,ood}.jsonl`
with varied question wording, candidate subsets (N = 2..6), aliased candidate text and
context wrappers. Splits are made per seed situation, so no paraphrase of a val/test
situation is trained on. This is a bootstrap set grown from templates: replace or extend
it with real logged decisions in the same JSONL format
(`{"state","question","candidates","correct","use_case"?}`).

```bash
python scripts/build_dataset.py
python scripts/train.py --backbone models/Qwen3.5-4B --device cuda \
  --data data/ant-multitask --name "ANT multitask" --out runs/ant-4b-2026-10-17
# Serve it from the sibling ../playground repository.
```

`metrics.json` reports test and OOD accuracy per use case; `card.json` and
`examples.json` (real held-out rows) are what the playground shows.

For cloud training of a separate **Qwen3.5-0.8B** variant, see the
[Kaggle guide](docs/training/train-08b-on-kaggle.md) or
[Colab guide](docs/training/train-08b-on-colab.md).

Training uses `L = CrossEntropy(scores, correct_candidate)` on the synthetic multilingual
benchmark, fits a temperature calibrator and a risk-coverage escalation policy on the validation
split, and saves a head-only checkpoint (the backbone stays on the Hub, so checkpoints are ~10 MB;
LoRA adapters are saved alongside). The end-to-end flow is drawn in
[docs/assets/ant-training-pipeline.svg](docs/assets/ant-training-pipeline.svg).

```bash
python scripts/evaluate.py --checkpoint runs/qwen-dot --split test   # + ood
python scripts/benchmark.py --backbone models/Qwen3.5-4B --device cuda
```

`benchmark.py` produces the scorer comparison (dot / cosine / bilinear / MLP), the candidate-count
scaling study (N = 2 … 128) and the batch-latency study into `runs/…`.

## Experiments

Every architecture change is gated by an entry in [EXPERIMENTS.md](EXPERIMENTS.md) with a fixed
shape: hypothesis → configuration → metrics → conclusion. No entry, no change. What follows is
the human-readable summary; the log itself carries full hardware, parameters and seeds.

### Done (all on CPU / mock backbone)

- **E-000 — backbone verification.** Confirmed via `AutoModel` that the official
  `Qwen/Qwen3.5-4B` (config rev `851bf6e`) exposes text-only hidden states `[B, L, 2560]` through
  `Qwen3_5Model` with no LM head loaded, and that every constant ANT needs (hidden size 2560,
  32 hybrid layers, vocab 248,320, 262k positions) can be resolved live from the config —
  `QwenBackbone` hard-codes nothing. Verified by instantiating the real module tree on
  transformers 5.17.0 / torch 2.14.0, inspection run on Apple M3.
- **E-001 — smoke test.** All 9 checks pass on `ANTModel.tiny()`: permutation invariance
  max |Δp| = 0.0 across reordered candidates, gradients reach the 8 head tensors and 0 backbone
  tensors (frozen backbone), candidate counts N = 2/3/5/9 work, duplicated candidates get
  identical probability. 84/84 unit tests pass; the `train → calibrate → escalate → evaluate →
  reload` loop runs end to end.
- **E-002 — mock-backbone sanity training.** `ANTModel.tiny()` (h=64, dot scorer) trained
  8 epochs on the synthetic benchmark (300/100/100/100 split): test accuracy 0.61 vs chance
  ≈ 0.18 at the benchmark's average N ≈ 5.5; OOD accuracy 0.13 with visibly lower confidence —
  the abstention signal behaves as designed. Temperature calibration and the escalation policy
  both fit on real model outputs. Conclusion: the pipeline is sound and ready for the real
  backbone; the absolute numbers are meaningless (toy backbone) and are not claimed anywhere.

Full records with dates and hardware: [EXPERIMENTS.md](EXPERIMENTS.md).

### Planned (templates filled as runs land)

| ID | Question | Command |
|---|---|---|
| E-101 | Which scorer wins: dot / cosine / bilinear / MLP (accuracy, NLL, Brier, ECE, latency, VRAM) | `python scripts/benchmark.py --backbone models/Qwen3.5-4B --device cuda` |
| E-102 | How accuracy and latency scale with candidate count N = 2…128 | `python scripts/benchmark.py … --scaling-sizes 2 4 8 16 32 64 128` |
| E-103 | Per-example latency at fixed candidate set with the embedding cache on | `python scripts/benchmark.py … --batch-sizes 1 8 32 64 128` |
| E-104 | Joint context+candidate encoding (Mode B) vs independent encoding, at N× context-encoding cost | train a `CONTEXT_JOINT_TEMPLATE` variant and compare against Mode A |
| E-105 | Calibration drift across languages en/it/es/fr/de | evaluate the E-101 winner per language on the test split |

Status, stated plainly: **no Qwen3.5-4B real-backbone run has been recorded yet.** The planned
tables in EXPERIMENTS.md are empty by design — they are the log's next entries, not results being
withheld. E-000..E-002 exist to remove pipeline risk before spending GPU hours, and none of their
numbers say anything about real-backbone accuracy. Results land in the log first, then roll up into
[docs/PAPER.md](docs/PAPER.md), the full paper-style write-up.

## Tests

```bash
pytest tests/ -q                 # 117 tests, no download needed (mock backbone)
ANT_TEST_QWEN=models/Qwen3.5-4B pytest tests/test_qwen_integration.py -m integration
```

## Repository layout

```
ant/
├── backbone/        # QwenBackbone (no LM head, vision tower dropped) + MockBackbone
├── encoding/        # pooling strategies, context/candidate encoders, prompt templates
├── scoring/         # dot, cosine, bilinear, MLP-interaction (registry, replaceable)
├── calibration/     # identity, temperature, vector-scaling, MLP calibrators
├── inference/       # predictor, escalation policies, candidate-embedding cache
├── training/        # losses, data, training modes (head/LoRA/partial/full), trainer
├── evaluation/      # metrics, calibration analysis, benchmark harness
├── data/            # synthetic triage benchmark, multi-use-case dataset, JSONL I/O
├── model.py         # ANTModel: score(S, Q, O_i), forward, head checkpoints
├── config.py        # ANTConfig (nothing hard-coded from the checkpoint)
└── types.py         # DecisionInput, DecisionExample, PredictionResult
scripts/             # download_model, smoke_test, train, evaluate, benchmark
runs/                # published ANT head checkpoints (including ant-4b-2026-10-17)
examples/            # inference, training
docs/ARCHITECTURE.md # design decisions, verified backbone facts
docs/PAPER.md        # full paper-style write-up
docs/assets/         # ant-architecture.svg, ant-training-pipeline.svg
docs/research-notes.md # external evidence base, source-linked
EXPERIMENTS.md       # research log (every experiment recorded)
```

## Roadmap

- Joint context/candidate encoding (Mode B) vs independent encoding study
- Learned pooling token; multimodal (image) candidate scoring — the context encoder input is the
  designed extension point, the scoring head is already agnostic
- Real (non-synthetic) benchmark suite

## References

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) — the backbone: 4B params,
  hidden 2560, 32 hybrid linear/full-attention layers, vocab 248,320, 262k context, Apache-2.0.
- [Jev (AI model), Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) — the proprietary
  System One model that defined the category (RLCD, typed questions, zero-shot).
- [Benchmarking Jev, arXiv:2609.37647](https://arxiv.org/abs/2609.37647) — independent 37-dataset
  evaluation: verified calibration, order invariance and selective prediction for choice outputs.
- [Laya, GitHub](https://github.com/NandhaKishorM/laya) — open-source Apache-2.0 System One model
  from ConvAI Innovations; fine-tuning recipe and router.
- [laya model card](https://huggingface.co/convaiinnovations/laya) and
  [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) —
  checkpoints, 0.36 zero-shot vs 0.766 fine-tuned on typed-decisions.
- [Train your own decision model, Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
  — LoRA decision-model recipe for Qwen3.5; Qwen3.5-4B trains on 8 GB VRAM.
- [Guo et al. 2017](https://arxiv.org/abs/1706.04599) — temperature scaling and the ECE metric,
  the calibration layer's baseline.
- [Geifman & El-Yaniv 2017](https://arxiv.org/abs/1705.08500) — selective prediction and
  risk/coverage curves, the escalation policy's framing.
- [FrugalGPT, arXiv:2305.05176](https://arxiv.org/abs/2305.05176) and
  [RouteLLM, arXiv:2406.18665](https://arxiv.org/abs/2406.18665) — LLM cascades and routers,
  the System 2 consumers of ANT's confidence outputs.
- [Qwen3 Embedding, arXiv:2506.05176](https://arxiv.org/abs/2506.05176) and
  [LLM2Vec, arXiv:2404.05961](https://arxiv.org/abs/2404.05961) — decoder-as-encoder precedents
  for using a Qwen backbone without its LM head.

## License

Apache-2.0. The Qwen3.5-4B weights are governed by their own license on the Hub.
