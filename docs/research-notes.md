# Research notes — System One decision models (Jev, Laya, and training recipes)

Collected 2026-10-16 via Firecrawl (web search + page scrape). This file is the internal
evidence base for `docs/PAPER.md` and the README experiments section. Every claim used in
the paper must trace back to an entry here.

---

## A. What a "System One decision model" is

TypeSafe AI coined the term for Jev (released 2026-09-15, proprietary, RLCD-trained):
a discriminative AI model that receives a `state` (text / JSON / array) and one or more
typed `questions` (`noul` = true/false, `choice` = unordered options, `score` = ordered
rubric) and returns typed answers with probability estimates — classification, not
generation. Output is meant for software, not for humans. Zero-shot: categories are
provided at inference time, no task-specific training required. Input limit 64k tokens
(32k for `state` plus the longest question). The name references the Jevons paradox and
Kahneman's System 1.

Source: https://en.wikipedia.org/wiki/Jev_(AI_model)

Independent evaluation (Deußer, Sparrenberg, Sifa, arXiv:2609.37647): Jev `jev-1.13.0`
zero-shot on 37 datasets / 346,009 requests under USD 10; 95–99% on IMDB, SST-2,
HellaSwag, ARC; 86.7% on Belebele across 122 languages; beats Qwen3.8-27B on 27/37
datasets; choice probabilities well calibrated, support selective prediction; accuracy
unchanged under option rotation (order invariance); near fixed-0.5 threshold placement
for binary questions (threshold tuning raised UNFAIR-ToS micro-F1 0.50 → 0.75).
Code: github.com/AppliedMachineLearning-Lab/jev-benchmarking.

Source: https://arxiv.org/abs/2609.37647

## B. Laya — the open-source System One model

- Repo: https://github.com/NandhaKishorM/laya (Apache-2.0, ConvAI Innovations, ~0.4.8, Oct 2026)
- Model card: https://huggingface.co/convaiinnovations/laya ; typed-decisions fine-tune:
  https://huggingface.co/convaiinnovations/laya-typed-decisions
- Same request format as Jev: `state` + typed questions (`choice`, `score`, `noul`),
  single forward pass (~33 ms one question, 7.2 ms/question batched on T4), no
  generation.
- Three checkpoints (all share the same head/training approach):
  | checkpoint | encoder backbone | params | context | best at |
  |---|---|---|---|---|
  | laya | ModernBERT-large | 421M | 512 | English text, guardrails, email triage |
  | laya-multilingual | mmBERT-base | 322M | 1024 (up to 8k) | 100+ languages, ~2x faster |
  | laya-typed-decisions | ModernBERT-large | 421M | 1024 | typed-decisions workflows |
- Training: RLCD — reinforcement learning against strictly proper scoring rules
  (log-loss / Brier family); "reporting honest probabilities is the only way to
  maximise reward". GRPO-style policy gradient. Decision head trained from scratch on
  top of a fully fine-tuned encoder (per multimodalart/jev-reproductions-tracker).
- Zero-shot is near chance on typed-decisions (0.36 vs 0.318 random); fine-tuned
  reaches 0.766 vs Jev's published 0.727 and a 0.735 teacher-ceiling. Positioning: "a
  fast base to specialise, not a zero-shot decision engine".
- Fine-tuning recipe (from README ## Fine-Tuning):
  - Kaggle 2x T4 DDP notebook: builds dataset → trains with RLCD (proper-scoring-rule
    rewards, GRPO-style policy gradient) → fits calibration temperatures → evaluates →
    pushes to Hub. ~4–5 h for 4 epochs over ~30k questions.
  - Apple-Silicon MPS script: single-process, micro-batches + grad accumulation
    (`--micro-batch 1 --grad-accum 32` on a 16 GB MacBook), gradient checkpointing on
    encoder AND decision head (`model.head_checkpointing = True` for head-only
    activation checkpointing).
  - Calibration: one temperature per question type (choice/score/noul); after fitting,
    inherited `temperature_by_options` buckets are removed from the exported config so
    the new fit is not silently masked. Calibration samples must be held out — the
    notebook's come from training items, so evaluate on separate data.
  - Worked example (browser-agent): `docs/finetune_browser_agent.md` — element top-1
    among ~45 candidates 0.10 → 0.66, real-task success 0% → 62% at 17–23 ms/step;
    data recipe: reverse-generated goals, executed DONE states, Mind2Web, on-policy
    corrections; published at cklxx/laya-browser.
- Router: detects language/script and dispatches to a checkpoint; routing metadata
  explains the choice.

## C. Jev training background

RLCD = "Reinforcement Learning for Calibrated Decisions" (Typesafe's term). CEO Diogo
Almeida: ~4 years at OpenAI on RLHF / InstructGPT / ChatGPT / GPT-4 before founding
TypeSafe in 2024. Jev released 2026-09-15 with a $40M seed round (DCVC-led).

Sources: https://en.wikipedia.org/wiki/Jev_(AI_model) ;
https://www.mindstudio.ai/blog/jev-system-one-model-launch

## D. Training your own decision model on a decoder backbone

Unsloth docs "Train your own Decision Model with Unsloth"
(https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth):

- Goal: turn Qwen3.8 / Gemma 4 / Qwen3.5 LLMs into decision models "like Jev, Laya and
  Clef" — reads input, scores provided options, returns choice + probability.
- Recipe: fine-tune with a Clef head, LoRA; official settings: epochs=2, LoRA rank=16,
  lr=2e-4 (their earlier sweep table used r=64 one-epoch for 0.8B/2B). Decisions API
  then serves it (model names `laya`, `default`, `jev-latest` reroute to your model).
- Dataset format: rows with `state`, `questions` (instructions + criteria/options),
  `gold` (correct answers). Question types: `choice` (option key), `noul` (True/False),
  `score` (ordered level). Mixed with 12 sources: ag_news, arc, banking77, boolq,
  clinc150, commonsense_qa, mmlu, mnli, prompt_injections, snli, sst5, wanli +
  typed-decisions. Test = 3,000 rows (2,000 typed-decisions + 500 BANKING77 + 500
  CLINC150), decontaminated.
- Results: Qwen3.5-0.8B typed-decisions 36% → 73%, BANKING77 7% → 74%, CLINC150
  19% → 76%, holdout 78%; Qwen3.5-2B 33% → 78% / 1% → 58% / 1% → 62%, holdout 81%.
  VRAM: 0.8B 4 GB (42 min), 2B 8 GB (40 min), Llama 3.2 3B 4.1 GB (30 min), Gemma-4
  E4B 14.4 GB (49 min), "Laya (fine-tuned)" 2.5 GB (10 min). Qwen3.5-4B decision model
  trains on 8 GB VRAM. Free notebooks: colab.research.google.com/github/unslothai/
  notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb.

## E. Related public decision models and benchmarks (context, optional citations)

- OpenDecider (manjunathshiva): open calibrated decision models compared with Jev,
  Laya, Stanford/NVIDIA CLM-8B — https://huggingface.co/blog/manjunathshiva/
  opendecider-beats-laya-and-jev
- S1MB (hotchpotch): System One decision models across 100+ benchmarks, Laya scores on
  Noul/Choice/Score/TP/AP — https://huggingface.co/blog/hotchpotch/system-one-mosaic-benchmark
- "Consider, Then Choose: Order-Invariant Decisions in System One Models"
  (arXiv:2610.18233): studies order invariance; fine-tuned decision models Torchcast
  12B, Kev 9B, Intern-Decision 4B from Gemma 4 / Qwen3.5-9B / …
- JevBench / Jev Decision Index report Laya performing poorly (laya discussion #10):
  https://huggingface.co/convaiinnovations/laya/discussions/10
- JEV/Laya as pentest decision layers (Barbosa, arXiv:2609.28940): four decision
  points (finding adjudication, severity recalibration, agent pruning, confirmation
  loops); discusses RLHF/RLAIF/RLCD/RLHV training approaches.
- Rafe & Das, arXiv:2609.24052 (crash narratives + Jev) and arXiv:2610.13092 (Jev in
  agent workflows).
- Community: Laya browser-controls to Doom (r/LocalLLaMA 1wl1yzq) — anecdote, not a
  source for the paper's claims.

## F. Generic System-1/System-2 literature (from the earlier research agent)

- From System 1 to System 2 (survey) — https://arxiv.org/abs/2502.17419
- Reasoning on a Spectrum — https://arxiv.org/abs/2502.12470
- FrugalGPT cascades — https://arxiv.org/abs/2305.05176 (GPT-4 parity at up to 98%
  cost cut; +4% acc at same cost)
- RouteLLM — https://arxiv.org/abs/2406.18665 (>2x cost cut, router transferable)
- Hybrid LLM — https://arxiv.org/abs/2404.14618
- Speculative decoding — https://arxiv.org/abs/2211.17192 (2–3x latency, draft+verify)
- Qwen3 Embedding (decoder-as-encoder, last-token pooling, reranker) —
  https://arxiv.org/abs/2506.05176
- LLM2Vec (decoder-only → text encoder) — https://arxiv.org/abs/2404.05961
- InstructGPT + reward-model Bradley–Terry template — https://arxiv.org/abs/2203.02155,
  https://rlhfbook.com/c/05-reward-models
- SBERT bi-encoder vs cross-encoder — https://arxiv.org/abs/1908.10084
- MMLU log-prob multiple choice — https://arxiv.org/abs/2009.03300; lm-evaluation-harness
  https://arxiv.org/abs/2306.13398
- Qwen3.5-4B model card — https://huggingface.co/Qwen/Qwen3.5-4B (4B params, hidden
  2560, 32 layers as 8 groups of 3x Gated DeltaNet + 1x Gated Attention, GDN: 32 V
  heads / 16 QK heads; GQA 16Q/4KV head dim 256; FFN 9216; vocab 248,320; 262k context;
  MMLU-Pro 79.1, IFEval 89.8; `causal-conv1d` + `flash-linear-attention` optional
  kernels; linear-attention lineage: Gated DeltaNet arXiv:2412.06464; analysis
  https://huggingface.co/blog/mlabonne/qwen35)
- Guo et al. temperature scaling + ECE — https://arxiv.org/abs/1706.04599
- Geifman & El-Yaniv selective prediction — https://arxiv.org/abs/1705.08500;
  SelectiveNet https://arxiv.org/abs/1901.09192; Chow reject option (1970).
