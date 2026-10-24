# Train the ANT 0.8B variant on Kaggle

Use this workflow if Colab GPU limits prevent the run. It trains a **new ANT scoring head**
on `Qwen/Qwen3.5-0.8B`; the 4B head cannot be reused because it was trained on different
backbone representations. The model is public on Hugging Face and downloads into the Kaggle
runtime. Only the small ANT checkpoint needs to be brought back.

## 1. Make a Kaggle dataset from the current working tree

The implementation currently has uncommitted changes. Do not use a GitHub clone or `git
archive`, which would omit them. From the repo root, create a source archive. This includes
the source and `data/ant-multitask`, but excludes model weights, virtualenvs and old runs:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
output = Path.home() / "Downloads" / "ant-4b-kaggle.zip"
excluded = {".git", ".venv", ".pytest_cache", "__pycache__", "models", "runs"}

with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts) or path.name == ".DS_Store":
            continue
        if path.is_file():
            archive.write(path, relative.as_posix())

print(f"Created {output} ({output.stat().st_size / 1e6:.1f} MB)")
PY
```

On Kaggle, create a **private Dataset** (for example, `ant-4b-source`) and upload
`ant-4b-kaggle.zip`. Keep that dataset attached to the notebook as an input.

## 2. Configure the Kaggle notebook

1. Create a Kaggle Notebook and attach the private source dataset with **Add Input**.
2. In **Notebook options**, select an available **GPU accelerator** (T4 is sufficient for
   head-only training; choose another available GPU if Kaggle offers it).
3. Enable **Internet** so the runtime can download `Qwen/Qwen3.5-0.8B` from Hugging Face.
4. Start a fresh notebook session. Kaggle's `/kaggle/input` is read-only; extract the repo and
   write checkpoints under `/kaggle/working`.

First, inspect the attached dataset path and extract the archive:

```python
from pathlib import Path
from zipfile import ZipFile

inputs = list(Path("/kaggle/input").glob("*"))
print("Attached inputs:", inputs)
archives = list(Path("/kaggle/input").rglob("ant-4b-kaggle.zip"))
assert len(archives) == 1, f"Expected one source archive, found: {archives}"

repo = Path("/kaggle/working/ant-4b")
repo.mkdir(parents=True, exist_ok=True)
with ZipFile(archives[0]) as archive:
    archive.extractall(repo)
print("Extracted to", repo)
```

Install the repo dependencies while preserving Kaggle's preinstalled CUDA-enabled PyTorch.
Pin transformers to the version used to develop and verify this code:

```python
%cd /kaggle/working/ant-4b
%pip install -e '.[train]' 'transformers==5.17.0'
```

Do **not** reinstall `torch` from pip; that can replace Kaggle's CUDA build with a CPU-only
wheel. Verify the runtime before loading the model:

```python
import torch, transformers
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("transformers:", transformers.__version__)
print("dataset files:", sorted(p.name for p in Path("data/ant-multitask").glob("*.jsonl")))
assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator before training"
assert transformers.__version__ == "5.17.0"
```

If Internet is disabled or unavailable, the Hugging Face download will fail; enable it in the
notebook options and restart the session. Do not upload the 0.8B backbone as part of the source
dataset; it is downloaded directly by the training script.

## 3. Train, calibrate and evaluate

Run the complete train command in one cell. Defaults: 3 epochs, batch size 8, learning rate
`2e-4`, seed 7, and calibration enabled. The script trains on `train`, selects the best head
epoch on `val`, fits temperature calibration and the risk/coverage escalation policy on `val`,
then evaluates the same `test` and `ood` splits and writes `metrics.json`.

```python
from pathlib import Path

run_name = "ant-08b-2026-10-21"
out = Path("/kaggle/working/runs") / run_name
out.parent.mkdir(parents=True, exist_ok=True)
```

```bash
!python -u scripts/train.py \
  --backbone Qwen/Qwen3.5-0.8B \
  --mode head_only \
  --device cuda \
  --dtype auto \
  --data data/ant-multitask \
  --name 'ANT 0.8B' \
  --out /kaggle/working/runs/ant-08b-2026-10-21
```

`--dtype auto` selects a dtype supported by the GPU. Do not force BF16 on a T4. The backbone
stays frozen; this is not full-model fine-tuning. Let the cell finish before interrupting the
session. The best head is checkpointed when validation improves, but optimizer state is not
saved, so the current training command is not a resumable training job.

Confirm the completed output and inspect the metrics:

```python
import json

run_dir = Path("/kaggle/working/runs/ant-08b-2026-10-21")
print(sorted(p.name for p in run_dir.iterdir()))
metrics = json.loads((run_dir / "metrics.json").read_text())
print(json.dumps({k: metrics[k] for k in ("training", "test", "ood") if k in metrics}, indent=2))
```

The full multitask test and OOD metrics come from `scripts/train.py`. Do not use
`scripts/evaluate.py` for this comparison: it evaluates the separate synthetic benchmark, not
`data/ant-multitask`.

## 4. Save and download the checkpoint

Create a zip in `/kaggle/working`, then use **Save Version** on the notebook and choose to save
the notebook output. Kaggle retains files under `/kaggle/working` as version output; the attached
input dataset remains read-only.

```bash
!cd /kaggle/working/runs && zip -rq /kaggle/working/ant-08b-2026-10-21.zip ant-08b-2026-10-21
!ls -lh /kaggle/working/ant-08b-2026-10-21.zip
```

After saving the notebook version, download `ant-08b-2026-10-21.zip` from the notebook's
**Output** panel. If using the Kaggle API/CLI instead, download the notebook version's output
from its generated kernel-output dataset. Do not download or attach the Hugging Face backbone.

Unzip the result on the local machine into this repo's `runs/` directory:

```bash
cd /path/to/ant-4b
unzip ~/Downloads/ant-08b-2026-10-21.zip -d runs/
```

The local backbone is already available at `models/Qwen3.5-0.8B`. If it is not present, download
it once:

```bash
.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## 5. Profile the serving path locally

Profile on the same M3 and with the same input lengths used for the 4B reference:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/profile_inference.py \
  --checkpoint runs/ant-08b-2026-10-21 \
  --backbone models/Qwen3.5-0.8B \
  --device mps --dtype auto \
  --lengths 32 128 512 1024 --iters 3 --warmup 1
```

Compare test accuracy and ECE on the identical multitask test JSONL. The recorded 4B reference
for the 40-row test sample is accuracy `0.725`, ECE `0.1553`; a comparison against a different
split, synthetic data, or a different number of rows is not apples-to-apples. Report latency,
resident memory, dtype, and first-predict latency alongside accuracy; use the 0.8B variant only
after reviewing both predictive quality and local serving behavior.

## Troubleshooting

- **CUDA is false / no GPU listed:** enable a GPU accelerator and restart the session.
- **Out of CUDA memory:** retry with `--batch-size 4 --encode-batch-size 4`; record that change
  with the run because it changes training throughput and reproducibility.
- **Session time limit/disconnect:** a partial head file is not a completed model artifact; check
  that `metrics.json`, calibration, escalation, and card files exist before downloading. If the
  run did not finish, rerun to a new output directory rather than treating the partial head as
  calibrated.
- **Non-finite loss:** preserve the full traceback and GPU/runtime details; do not report the
  run as successful.
- **Quantization:** keep Metal quantization disabled. The current MPS path produces zeros/NaNs
  on the real 4B model; this Kaggle recipe uses plain CUDA weights.

## Related guide

The equivalent Google Colab workflow is in
[`train-08b-on-colab.md`](train-08b-on-colab.md).
