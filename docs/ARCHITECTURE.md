# ANT-4B Architecture

ANT-4B is a System 1 decision engine: `s_i = f_θ(C(S, Q), E(O_i))` and
`P(O_i | S, Q) = softmax(s / T)`. Nothing in the inference path generates text.

## Verified backbone facts

Inspected against the official checkpoint (revision `851bf6e`, 2026-09-28) and against a
random-init `Qwen3_5Config` model on transformers 5.17 / torch 2.14:

| Property | Value | Source |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | config.json |
| `model_type` | `qwen3_5` (multimodal: `text_config` + `vision_config`) | config.json |
| text hidden size | 2560 | `config.text_config.hidden_size` |
| text layers | 32, hybrid `linear_attention` ×3 + `full_attention` ×1 (interval 4) | `layer_types` |
| vocab size | 248,320 (tied embeddings) | `text_config.vocab_size` |
| native context | 262,144 positions | `max_position_embeddings` |
| parameters | ~4.66B BF16, ~9.3 GB on disk (2 safetensors shards) | Hub API |
| `AutoModel` mapping | `Qwen3_5Model` with children `visual`, `language_model` | verified at runtime |
| text-only forward | `language_model(input_ids, attention_mask)` → `[B, L, 2560]`, no LM head | verified at runtime |

`QwenBackbone` reads every dimension dynamically (`resolve_hidden_size` handles both
`text_config`-wrapped and flat configs), loads via `AutoModel` (never `AutoModelForCausalLM`),
and drops the vision tower by locating the decoder (`layers` ModuleList + `norm`) — verified path:
`("language_model",)`. ANT caps inputs at 8192 tokens; the 262k window is irrelevant for triage.

Without `causal-conv1d` / `flash-linear-attention` installed, the linear-attention layers fall
back to a reference implementation (correct but slow) — install them on CUDA.

## Dataflow

```
state, question ──► ContextEncoder ──► z_context [B, d]        d = scoring_dim (512 default)
candidate_i ──────► CandidateEncoder ► z_candidate_i [N, d]
                            │
z_context × z_candidates ──► Scorer ──► scores [B, N]          (masked slots at -1e9)
                            │
                Calibrator (fitted on validation split)
                            │
                 softmax(scores / T) ──► probabilities [B, N]
                            │
            ConfidenceMetrics (top-1, margin, entropy, normalized entropy)
                            │
                EscalationPolicy ──► "execute" | "escalate"
```

Both encoders share one backbone instance (one copy of Qwen in memory), apply their own pooling +
`LayerNorm` + linear projection, and cast hidden states to FP32 at the head so a BF16 backbone
trains against a stable FP32 head.

### Prompt format

```
<STATE>\n{state}\n</STATE>\n<QUESTION>\n{question}\n</QUESTION>      # context (candidate-free)
<CANDIDATE>\n{candidate}\n</CANDIDATE>                                # one per candidate
```

The context representation is independent of the candidate set by construction (spec §4). The
multimodal extension point is `ContextEncoder.format`: an image-capable variant injects vision
embeddings into the same context stream, and nothing downstream changes.

### Pooling

Registry: `mean` (masked mean, default), `last_token`, `max`, `attention` (learned query).
All are mask-aware and differentiable. A learned `[CLS]`-style token is roadmap: it requires
embedding-level integration with the backbone, while attention pooling reaches similar
expressiveness without touching backbone inputs.

### Scoring

All scorers consume `(z_context [B,d], z_candidates [B,N,d])` and produce `[B,N]`, and depend
only on the pair `(z_context, z_i)` — never on position or on sibling candidates. That single
contract yields permutation invariance and variable N for free.

- `dot`: `z_c · z_i` — v0.1 baseline
- `cosine`: cosine similarity × learnable positive scale (raw cosine ∈ [-1,1] is too narrow for
  logits)
- `bilinear`: `z_cᵀ W z_i`, W initialized near identity (starts at the dot baseline)
- `mlp`: interaction MLP over `[z_c, z_i, z_c ⊙ z_i, |z_c − z_i|]`

### Batch forward

`ANTModel.forward(states, questions, candidates_per_row)` deduplicates candidates across the
batch before encoding, scatters embeddings back per row, pads to N_max with a boolean mask, and
masks padded scores to −1e9 (softmax-neutral, so padded matrices feed losses and calibrators
directly).

## Guarantees and documented behaviors

- **Permutation invariance (Test A)**: candidates are encoded independently and scorers are
  pairwise — mapped probabilities are identical up to float associtivity (unit-tested end-to-end
  for all four scorers; smoke test measures max |Δp|).
- **Duplication (Test B)**: two copies of a candidate receive bit-identical scores, hence equal
  probability; the label's total mass is split across copies. `PredictionResult.probabilities`
  merges duplicates by summing.
- **Variable N (Test §19)**: no component has a fixed class dimension; N = 1 degenerates to
  probability 1.0.
- **Paraphrase (Test C)**: candidates are embedded as text, so "Billing" and "Handle the
  customer's billing issue" are distinct but neighboring points; the scorer comparison and
  `examples/` expose the measured similarity.
- **OOD / abstention (Tests E, F)**: escalation thresholds come from validation risk/coverage
  (AURC, E-AURC reported by `scripts/evaluate.py`), not from a constant.

## Calibration

Post-hoc, fitted on a split disjoint from the final test set:

| Calibrator | Form | Notes |
|---|---|---|
| `identity` | s | raw logits baseline |
| `temperature` | s / T | **default**; the only method well-defined for variable N |
| `vector_scaling` | a·s + b | with variable candidate sets a constant bias cancels in the softmax, so only the scale is identifiable |
| `mlp` | g(s) per candidate | shared across candidates (invariance-safe); can be non-monotone — validate vs temperature |

Metrics: ECE, MCE, Brier (multiclass), NLL, reliability diagrams (`plot_reliability`),
risk-coverage curves with AURC/E-AURC.

## Escalation

`EscalationPolicy.decide(ConfidenceMetrics) -> "execute" | "escalate"`.

- `AlwaysExecute` — default: ANT exposes the metrics; the external System 2 router decides
  (spec §13). ANT never invokes System 2 itself.
- `ThresholdPolicy(metric, threshold)` — metric ∈ {confidence, margin, entropy, normalized_entropy}.
- `RiskCoveragePolicy(target_coverage and/or target_risk).fit(confidences, corrects)` — sweeps
  validation thresholds and picks the smallest one meeting the budget (max automation within the
  stated risk).

## Training modes

| Mode | Trains | Use |
|---|---|---|
| `head_only` (v0.1) | projections + pooling + scorer (~2.6M params at d=512) | fastest validation of candidate scoring |
| `lora` | PEFT adapters on the decoder + head | backbone adaptation at low memory |
| `partial` | last-k decoder layers + head | cheap full-quality experiments |
| `full` | everything | research only |

Objective: masked cross-entropy on the correct candidate (spec §14), with an optional pairwise
margin term (spec §15, off by default). Trainer: AdamW, warmup + cosine schedule, BF16 autocast
on CUDA, grad clipping, best-val-loss checkpointing. Checkpoints persist only the head (plus LoRA
adapter and mock-backbone weights when applicable) — the Qwen backbone is reloaded from the Hub.

The frozen-backbone contract: `QwenBackbone` starts with `requires_grad_(False)`; `encode` only
enables grad when some backbone parameter is trainable, so head-only training builds no graph
through the 4B parameters and LoRA/partial training still backprop into the adapters.

## Encoding modes (spec §7)

- **Mode A (implemented)**: candidates encoded independently; embeddings cacheable.
- **Mode B (roadmap, E-004)**: joint `<STATE>…<QUESTION>…<CANDIDATE>` encoding per candidate,
  extracting a candidate-conditioned state. The scorer interface already accepts any
  `[B,N,d] × [B,d]` pair, so Mode B slots in as an alternative encoder pair without touching
  scoring, calibration, or escalation.

## Candidate cache

`CandidateEmbeddingCache` stores `candidate → z` (optional LRU bound). For a static candidate
set, runtime cost is one context encoding + N dot products; `predict_batch` encodes candidates
once for all states.

## Synthetic benchmark

Six departments with a hard-negative graph (Billing ↔ Sales ↔ Customer Success, Tech Support ↔
Customer Success, Legal ↔ Security), three difficulties (easy/medium/hard — hard mentions a
confusable department but the gold label follows the strongest signal), five languages
(en/it/es/fr/de; candidates always stay English per spec §21), unseen OOD departments
(Facilities, Procurement) plus garbled inputs. Deterministic per (seed, split).

## Design decisions worth recording

- `AutoModel` + decoder extraction instead of `ForConditionalGeneration`: no LM head in memory,
  no generation path to accidentally depend on (spec §34).
- FP32 head over BF16 backbone: pooling/projection stability at negligible memory.
- −1e9 (not −inf) for masked scores: softmax-safe in BF16 and in padded numpy matrices.
- Checkpoints exclude the backbone by construction (name-filtered state dict), keeping them
  small and Hub-friendly.
