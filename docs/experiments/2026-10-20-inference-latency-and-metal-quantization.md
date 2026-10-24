# Inference latency & Metal 8-bit quantization on the real backbone (lab report, 2026-10-20)

**Goal**: reduce ANT-4B inference latency from ~13 s per prediction toward milliseconds on
`Qwen/Qwen3.5-4B` (4.66B hybrid GDN + full attention, checkpoint declared bf16) running locally
on MPS; 8-bit quantization was the first approved lever, with long-input support as a
requirement. **Machine**: Apple M3, 16 GB unified memory, macOS; Python 3.13 (`.venv`), PyTorch
on MPS, transformers 5.17.0, kernels 0.16.2, weights cached in `~/.cache/huggingface`.

**Verdicts up front**:

1. **Working config: fp16 plain load (no `device_map`), fused single-pass encode,
   `torch.inference_mode`, length bucketing 64/128/256/512/1024.** Warm medians
   **372–373 ms** on short inputs (14/53 tokens), **1446 ms** at 210 tokens, **2866 ms** at
   418 tokens — **≈8× faster** than the old ~13 s baseline on short inputs.
2. **Metal 8-bit quantization is OFF** (`ANTConfig.quantization` defaults to `None`): it fails
   the accuracy gate by a wide margin (accuracy 0.725 → 0.325, argmax flips 27/40), and
   reloading the saved pre-quantized checkpoint is fatal on this stack (hang, SEGFAULT).
3. **Root cause is size/shape/path-specific to the real 4B model on MPS** — the same quantizer
   and the same kernel are numerically correct on a tiny locally-built model. transformers pins
   version 1 of `kernels-community/mlx-quantization-metal-kernels` while version 2 exists.

All code changes (quantization config plumbing, `resolve_dtype` FP16-on-MPS fix, fused encode,
`CandidateEmbeddingCache.store()`, `scripts/compare_quantization.py`, `scripts/profile_inference.py`,
`scripts/_smoke_metal.py`, length bucketing) were uncommitted at the time of writing; 117 unit
tests were green.

---

## Timeline

### 1. First successful 8-bit load — bf16 (before the dtype fix reached the quantized path)

- **Configuration**: `Qwen/Qwen3.5-4B` loaded through the new quantized path with `dtype`
  still resolving to bf16 (the FP16-on-MPS fix in `resolve_dtype` was not yet forwarded into
  the quantized load).
- **Load**: 139 s; MPS allocated **3.80 GB**; **248 `MetalLinear` modules** quantized; the
  embedding stayed bf16.
- **Warm encode timings** (single-sequence, `scripts/_smoke_metal.py` bench):

| Input (tokens) | Warm encode |
|---:|---:|
| 20 | ≈ 1.34 s |
| 78 | ≈ 1.9 s |
| 309 | ≈ 6.5 s |
| 619 | ≈ 12.5 s |

- **Source**: `/tmp/ant_metal8.log` — *that file was later overwritten with the fp16 rerun of
  experiment 3, so these numbers exist only in this summary.* The similarity of the 619-token
  figure (~12.5 s) to the old ~13 s prediction baseline already showed quantization does not
  touch the linear-attention compute floor (see root-cause analysis).

### 2. Machine hard-reboot at 20:10 local

A full-precision 4B load was in progress when the machine hard-rebooted. Kernel uptime evidence:
`kern.boottime` = **Tue Oct 20 20:10:06**. Loading/quantizing a 4B model on a 16 GB unified-memory
machine is the memory-dangerous moment. Everything survived on disk (weights cache, checkpoint
`runs/ant-4b-2026-10-17`, logs).

### 3. FP16 8-bit load after the dtype fix — the clean baseline for quantized behaviour

- **Load**: **64 s**; MPS allocated **5.07 GB**; a sampled parameter confirmed
  `dtype=torch.float16` on `mps:0`; **248 `MetalLinear` modules**.
- **Encode timings** (cold = first pass at that shape, warm = median of repeated passes):

| Input (tokens) | Cold (ms) | Warm median (ms) |
|---:|---:|---:|
| 17 | 2909 | 374 |
| 67 | 847 | 741 |
| 263 | 3032 | 2999 |
| 526 | 6113 | 5891 |

- **Saved pre-quantized checkpoint**: `backbone.model.save_pretrained` wrote
  `models/Qwen3.5-4B-metal-8bit/` in **3.7 s** — a single `model.safetensors` of **5.07 GB**
  containing **922 tensors**, **248 with `.scales` siblings**, weight tensors in **U32** and the
  embedding in **F16**.
- **Known omission**: the freshly written `config.json` was **missing**
  `quantization_config` (transformers failed to emit it on save); it was patched manually to
  `{"bits": 8, "group_size": 64, "quant_method": "metal"}`.
- **Source**: `/tmp/ant_metal8.log` (the surviving content of that file is this rerun).

### 4. Pre-quantized checkpoint reload — both attempts fatal

| Attempt | Path | Outcome |
|---|---|---|
| Reload with `quantization=None` | loader discovers the embedded quantization config, no `device_map` | **Hung >10 min** at `Loading weights 0%|0/922` burning 4 CPU cores; killed manually. |
| Reload through the normal path with `device_map="mps"` | | **SEGFAULT** (`EXC_BAD_ACCESS` / SIGSEGV) inside the PyTorch MPS copy/cast kernel — stack: `copy_cast_kernel_mps` → `exec_unary_kernel` → kernel-string hash table. Crash report: `python3.13-2026-10-20-225330.ips`. |

A third, collateral crash: the **full-precision plain fp16 load with `device_map="mps"`** on the
reference arm of the gate script **died silently (SIGTRAP in malloc**; crash report
`python3.13-2026-10-20-225718.ips`). The plain load **without** `device_map` (the historic path)
works and is what shipped. Conclusion: **pre-quantized reload is unusable on this stack**; the
checkpoint is kept as an artifact only.

### 5. Accuracy gate — FAIL by design

`scripts/compare_quantization.py`, 40 test rows of `data/ant-multitask`, checkpoint
`runs/ant-4b-2026-10-17`. The reference arm verified the pipeline end-to-end on the real
backbone (fp16 plain: accuracy 0.725, ECE 0.1553 — sane). The quantized arm then collapsed:

| Metric | fp16 plain (reference) | metal 8-bit (quantized) | Δ | Gate tolerance |
|---|---:|---:|---:|---:|
| Accuracy | 0.725 | 0.325 | **−0.400** | \|Δ\| ≤ 0.005 |
| ECE | 0.1553 | 0.0 | **−0.155** | \|Δ\| ≤ 0.02 |
| Argmax flips (of 40) | — | — | **27 (67.5%)** | — |
| Mean/per-row \|Δp\| (max) | — | — | **NaN** (mean; max NaN) | — |

**GATE: FAIL.** Result JSON: `/tmp/ant_gate.json`. The NaN in per-row |Δp| is a diagnostic of
the broken quantized forward, not a gate-script problem.

### 6. Failure-isolation matrix — single real test row, batch 1

| Configuration | Hidden states | \|h\|mean | MPS resident |
|---|---|---:|---:|
| fp16 + 8-bit metal | **all-zero** (no NaN) | 0.000e+00 | 5.08 GB |
| bf16 + 8-bit metal | **NaN** | NaN | 3.80 GB |
| fp16 plain (reference) | sane | 1.977e+00 | 8.42 GB |
| bf16 plain | sane, identical to fp16 plain | 1.977e+00 | 8.42 GB |

The two failure modes (all-zero vs NaN) depend on the compute dtype, not on quantization alone.
Plain fp16 and plain bf16 are numerically identical on this row, so the earlier `resolve_dtype`
FP16-on-MPS fix does not alter the numbers. Note on residency: plain loads show **8.42 GB**
(probe and bench agree); an earlier note of "~9.3 GB nominal fp16 size" was the on-disk size and
is superseded here. Sources: `/tmp/ant_nan2.log` (log deleted), `/tmp/ant_nan3.log` (surviving
and complete).

### 7. Root-cause probes — the failure is not ours

Probed before the final latency bench; narrated in detail in the "Root-cause analysis" section.
On a tiny locally-built `Qwen3_5TextModel` (hidden 256, 2 layers, vocab 512, MPS, in-process):
8-bit Metal quantization is **correct** (max \|hidden − fp16-ref\| = 5.08e-02, no zeros, no
NaN), also when the checkpoint is saved bf16 and loaded with `dtype=float16` (`|h|mean` =
7.996e-01 in both request cases). Same quantizer, same kernel — the garbage on the 4B is
size/shape/path-specific, not our loading code, not dtype casting. Meanwhile transformers pins
kernel version 1 of `kernels-community/mlx-quantization-metal-kernels` while version 2 exists.

### 8. Working-config latency — fp16 plain, fused path, warm medians

The fused single-pass encode (context + all candidates in one backbone pass) with
`torch.inference_mode` and length bucketing (64/128/256/512/1024):

| Input (tokens) | Cold (ms) | Warm median (ms) |
|---:|---:|---:|
| 14 | 745 | **372** |
| 53 | 481 | **373** |
| 210 | 1517 | **1446** |
| 418 | 2942 | **2866** |

First `predict()` call (3 candidates, short input): **1681 ms** — first-touch compilation of the
template shapes. Against the old **~13 s per prediction** baseline this is **≈8× on short
inputs**. The bf16 arm of this run died silently, so **bf16 plain latency is unmeasured**; only
its numerics were verified (experiment 6). Source: `/tmp/ant_plain_bench.log`.

---

## Measured numbers

Consolidated from the experiments above. Latency rows are single-sequence backbone/encode
passes; medians are of repeated warm passes.

**Latency — Metal 8-bit, bf16 load (pre-fix, experiment 1)** *(only in this summary; log overwritten)*

| Input (tokens) | Warm encode |
|---:|---:|
| 20 | ≈ 1.34 s |
| 78 | ≈ 1.9 s |
| 309 | ≈ 6.5 s |
| 619 | ≈ 12.5 s |

**Latency — Metal 8-bit, fp16 load (post-fix, experiment 3)**

| Input (tokens) | Cold (ms) | Warm median (ms) |
|---:|---:|---:|
| 17 | 2909 | 374 |
| 67 | 847 | 741 |
| 263 | 3032 | 2999 |
| 526 | 6113 | 5891 |

**Latency — working config, fp16 plain + fused encode (experiment 8)**

| Input (tokens) | Cold (ms) | Warm median (ms) |
|---:|---:|---:|
| 14 | 745 | 372 |
| 53 | 481 | 373 |
| 210 | 1517 | 1446 |
| 418 | 2942 | 2866 |

First `predict()`: 1681 ms. Old baseline: ~13 s per prediction.

**Load & residency summary**

| Configuration | Load time | MPS resident | Notes |
|---|---:|---:|---|
| 8-bit metal, bf16 (pre-fix) | 139 s | 3.80 GB | embedding left bf16 |
| 8-bit metal, fp16 (post-fix) | 64 s | 5.07 GB | 248 MetalLinear; sample param float16 on mps:0 |
| Plain fp16 / bf16 (no `device_map`) | — | 8.42 GB | shipped path |

**Accuracy gate (experiment 5)** — see table above; all four key rows reproduced in
`/tmp/ant_gate.json` (accuracy 0.725/0.325, ECE 0.15534491017460827 → reported 0.1553, flips
27/40 = 0.675, gate failed on both tolerances).

**Failure-isolation matrix (experiment 6)** — see table above; every cell verified against
`/tmp/ant_nan3.log` (`allzero=True |h|mean=0.000e+00` / `nan=True |h|mean=nan mps_gb=3.80` /
`1.977e+00` twice at `mps_gb=8.42`).

---

## Root-cause analysis

The question after experiments 4–6 was whether the all-zero / NaN garbage came from our loading
code, from dtype casting, or from the quantization stack itself. **Probes on a tiny
locally-built `Qwen3_5TextModel`** (hidden 256, 2 layers, vocab 512, run on MPS in-process,
outside any checkpoint) answered it:

- **8-bit Metal quantization is CORRECT on the tiny model**: max \|hidden − fp16-ref\| =
  **5.08e-02**, with no zeros and no NaN — same quantizer, same kernel version.
- It is **also correct** when the tiny checkpoint is saved bf16 and reloaded with
  `dtype=float16` (`|h|mean` = **7.996e-01** both when bf16 is requested and when fp16 is
  requested) — so casting the weights from bf16 storage to fp16 at load is not the failure.

The same quantizer and kernel therefore produce garbage **only on the real 4B model at its real
shapes on MPS**. The failure is **size/shape/path-specific**; our loading code and dtype
handling are exonerated.

**Kernel version pinning.** transformers pins **version 1** of
`kernels-community/mlx-quantization-metal-kernels`: the pin is taken from
`transformers/integrations/metal_quantization.py` (`_get_metal_kernel`, calling
`get_kernel(..., version=1)`), while **version 2 exists** — every quantized load printed
*"You are using version 1 of 'kernels-community/mlx-quantization-metal-kernels', but version 2
is available."* A fixed kernel release (or matching transformers/PyTorch stack) is the term to
hold until re-testing; no point in re-running the gate against a stack that still pins v1.

**Independent constraint — the linear-attention floor.** `flash-linear-attention` and
`causal_conv1d` are CUDA/Triton-only, so the **24 GatedDeltaNet layers run the reference PyTorch
path on MPS** and dominate long inputs: latency grows roughly linearly with length (visible in
both quantized and plain tables: 17 → 526 tokens scales from ~0.37 s to ~5.9 s warm). 8-bit
quantization addresses **memory residency** (3.80–5.07 GB vs 8.42 GB resident), not that compute
floor. The `linear_attention_chunk_size` knob was implemented but is **not yet
runtime-benchmarked**.

---

## Conclusions & follow-ups

**Conclusions**

- **Ship the working config**: plain fp16 load **without** `device_map` (the device_map path
  crashes even for plain loads on this stack — SIGTRAP in malloc), fused single-pass encode,
  `torch.inference_mode`, length bucketing 64/128/256/512/1024. ≈8× on short inputs; ~2.9 s at
  418 tokens. Milliseconds are not reachable on a 4.66B hybrid on this machine today.
- **Quantization stays OFF** (`ANTConfig.quantization` default `None`) and is currently
  mis-broken twice over: it fails the accuracy gate, and its saved checkpoint cannot be
  reloaded. Re-enable only after a fixed transformers/PyTorch stack.
- **Gate-first rule**: any future quantized number must come after a passing
  `scripts/compare_quantization.py` run (tolerances: \|Δaccuracy\| ≤ 0.005, \|ΔECE\| ≤ 0.02,
  plus argmax-flip inspection) against a fresh fp16 reference arm.
- **Artifacts**: `models/Qwen3.5-4B-metal-8bit/` (4.7 GB) is kept as an artifact of the
  experiments and is deletable; `config.json` in it still carries the manual
  `quantization_config` patch. Nothing is git-committed yet.

**Follow-ups**

1. **Upstream bug report** to transformers/kernels-community with the tiny-model evidence in
   hand (correct at small size, garbage at 4B on MPS; v1 pinned while v2 exists; save omits
   `quantization_config`; pre-quantized reload SEGFAULTs in the MPS copy/cast kernel).
2. **Re-run the accuracy gate after any transformers/PyTorch (or kernels) bump** — before
   trusting any quantized number again.
3. **Runtime-benchmark `linear_attention_chunk_size`** — implemented, unmeasured; it tunes the
   reference GDN path that dominates long inputs.
4. **MLX port remains the path to a real millisecond range** for this hybrid on a 16 GB M3 —
   deliberately deferred as a later step.

---

## Appendix: provenance

Session logs live under `/tmp` and may disappear; the crash reports live under
`~/Library/Logs/DiagnosticReports/` and the numbers are also recorded in this file.

| Artifact | Status at writing | Contents |
|---|---|---|
| `/tmp/ant_metal8.log` | present, **overwritten** | Now holds the fp16 rerun (experiment 3: load 64 s, 5.07 GB, 248 MetalLinear, fp16 param, encode table, checkpoint save in 3.7 s). The bf16 first-load numbers of experiment 1 (139 s, 3.80 GB, warm 1.34/1.9/6.5/12.5 s) exist only in this summary. |
| `/tmp/ant_gate.log` | present | Gate run console output (experiment 5). |
| `/tmp/ant_gate.json` | present | Full gate result (accuracy/ECE/deltas/flips/tolerances). |
| `/tmp/ant_nan2.log` | **deleted** | Failure-isolation rows (session log); numbers in this summary. |
| `/tmp/ant_nan3.log` | present | Complete failure-isolation matrix (experiment 6), including `mps_gb` per arm. |
| `/tmp/ant_reload.log` | present | Reload attempt 1 — embedded-config discovery warning `Set device_map='mps'…`, progress stall at `0/922`, multiprocessing shutdown warnings after the manual kill. |
| `/tmp/ant_reload2.log` | present | Reload attempt 2 — progress stall at `0/922`, shutdown warnings after the SIGSEGV. |
| `/tmp/ant_plain_bench.log` | present | Working-config bench (experiment 8): fp16 plain arm prints `mps 8.42 GB`, the four latency rows and `predict_ms=1681`; the bf16 arm starts loading and produces no output (silent death). |
| `~/Library/Logs/DiagnosticReports/python3.13-2026-10-20-225330.ips` | present | Pre-quantized reload SIGSEGV: `copy_cast_kernel_mps` → `exec_unary_kernel` → kernel-string hash table. |
| `~/Library/Logs/DiagnosticReports/python3.13-2026-10-20-225718.ips` | present | Plain fp16 + `device_map="mps"` silent death (SIGTRAP in malloc), reference arm of the gate script. |
| `kern.boottime` | verified live | `Tue Oct 20 20:10:06 2026` — the reboot of experiment 2. (No log file; uptime evidence only.) |

Not separately logged: the tiny-model root-cause probes of experiment 7 ran in-process on a
locally-built model (no log file); their numbers (5.08e-02, 7.996e-01) are given in the
root-cause analysis above and were command-verified in session.
