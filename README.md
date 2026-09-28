# ANT-4B — Adaptive Neural Triage

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

The primitive is `score(state, question, candidate) -> scalar`, and the choice primitive is
`choose(state, question, candidates) -> calibrated distribution`. There is no LM head and no
`model.generate(...)` anywhere in the inference path.

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

## Install

```bash
git clone <this-repo> && cd ant-4b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

- Runtime: `torch`, `transformers>=4.57` (native `qwen3_5` support), `numpy`
- Extras: `train` (PEFT/LoRA, accelerate), `hub` (HF CLI + fast transfers), `viz` (matplotlib)

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
| Apple Silicon 32 GB | works (MPS, `torch_dtype="bfloat16"`, small batches) |
| Apple Silicon 16 GB | marginal — BF16 fits but headroom is tight |
| CPU only | not recommended for the 4B backbone; use the mock backbone for development |

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

## Training (mode 1: frozen Qwen + trainable head)

```bash
# CPU end-to-end sanity run on the mock backbone (~seconds)
python scripts/train.py --backbone mock --out runs/mock-dot

# Real backbone
python scripts/train.py --backbone models/Qwen3.5-4B --device cuda --out runs/qwen-dot

# LoRA mode (requires ant-4b[train])
python scripts/train.py --backbone models/Qwen3.5-4B --mode lora --device cuda --out runs/qwen-lora
```

Training uses `L = CrossEntropy(scores, correct_candidate)` on the synthetic multilingual
benchmark, fits a temperature calibrator and a risk-coverage escalation policy on the validation
split, and saves a head-only checkpoint (the backbone stays on the Hub, so checkpoints are ~10 MB;
LoRA adapters are saved alongside).

```bash
python scripts/evaluate.py --checkpoint runs/qwen-dot --split test   # + ood
python scripts/benchmark.py --backbone models/Qwen3.5-4B --device cuda
```

`benchmark.py` produces the scorer comparison (dot / cosine / bilinear / MLP), the candidate-count
scaling study (N = 2 … 128) and the batch-latency study into `runs/…`.

## Tests

```bash
pytest tests/ -q                 # 84 tests, no download needed (mock backbone)
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
├── data/            # synthetic multilingual triage benchmark (en/it/es/fr/de + OOD)
├── model.py         # ANTModel: score(S, Q, O_i), forward, head checkpoints
├── config.py        # ANTConfig (nothing hard-coded from the checkpoint)
└── types.py         # DecisionInput, DecisionExample, PredictionResult
scripts/             # download_model, smoke_test, train, evaluate, benchmark
examples/            # inference, training
docs/ARCHITECTURE.md # design decisions, verified backbone facts
EXPERIMENTS.md       # research log (every experiment recorded)
```

## Roadmap

- Joint context/candidate encoding (Mode B) vs independent encoding study
- Learned pooling token; multimodal (image) candidate scoring — the context encoder input is the
  designed extension point, the scoring head is already agnostic
- Real (non-synthetic) benchmark suite

## License

Apache-2.0. The Qwen3.5-4B weights are governed by their own license on the Hub.
