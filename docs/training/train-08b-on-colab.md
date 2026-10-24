# Train the ANT 0.8B variant on Google Colab

This guide trains a **new ANT scoring head** on `Qwen/Qwen3.5-0.8B`. Do not reuse the
4B checkpoint's head: it was trained on a different backbone representation. Training the
0.8B model locally on an M3 / 16 GB machine was stopped because the reference PyTorch
GatedDeltaNet path made the run impractically slow. Use a Colab GPU instead.

The workflow transfers the current working tree as a zip, trains on the same multitask
dataset, saves the small ANT checkpoint to Google Drive, then downloads only that checkpoint
back to the local machine. The backbone weights are fetched from Hugging Face by Colab and
do not need to be copied back.

## 1. Prepare a source archive locally

The current implementation has uncommitted changes. A GitHub clone or `git archive` will not
contain them, so create an archive from the working tree. This excludes the virtualenv, model
weights, previous runs, and caches; it **includes** the source and `data/ant-multitask`.

From the repository root:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
output = Path.home() / "Downloads" / "ant-4b-colab.zip"
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

The archive contains the current code and dataset, but no model weights or old checkpoints.

## 2. Create the Colab runtime

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Upload `ant-4b-colab.zip` with the first cell below. Mount Drive so the checkpoint survives
   a Colab disconnect.

```python
from google.colab import drive, files
drive.mount("/content/drive")
uploaded = files.upload()  # choose ~/Downloads/ant-4b-colab.zip
archive_name = next(iter(uploaded))
```

Extract it and install the project. Pin transformers to the version used to develop and verify
this code; do not install the Apple-only `metal` extra or CUDA GatedDeltaNet packages for the
first training run.

```python
from pathlib import Path
from zipfile import ZipFile

repo = Path("/content/ant-4b")
repo.mkdir(parents=True, exist_ok=True)
with ZipFile(archive_name) as archive:
    archive.extractall(repo)
%cd /content/ant-4b
```

```bash
%pip install -e '.[train]' 'transformers==5.17.0'
```

Check that Colab sees the GPU and the expected source/data before starting the long step:

```bash
!nvidia-smi
!ls data/ant-multitask/{train,val,test,ood}.jsonl
!python -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__)'
```

If `cuda` prints `False`, stop and change the runtime type before training.

## 3. Train, calibrate, and evaluate

Run the training script once. It trains the head on `train`, selects the best epoch on `val`,
fits a temperature calibrator and risk/coverage escalation policy on `val`, then prints test and
OOD metrics and writes the complete checkpoint. Defaults are 3 epochs, batch size 8, learning
rate `2e-4`, seed 7, and calibration enabled.

```bash
!mkdir -p /content/drive/MyDrive/ant-4b/runs
!python -u scripts/train.py \
  --backbone Qwen/Qwen3.5-0.8B \
  --mode head_only \
  --device cuda \
  --dtype auto \
  --data data/ant-multitask \
  --name 'ANT 0.8B' \
  --out /content/drive/MyDrive/ant-4b/runs/ant-08b-2026-10-21
```

`--dtype auto` selects a dtype supported by the selected GPU. Do not force BF16 on a T4. The
backbone is frozen (`head_only`); only the compact ANT encoders/scorer trainable parameters are
updated. The first run downloads `Qwen/Qwen3.5-0.8B` from the public Hugging Face repository.

Confirm that training completed and the output has the expected files:

```bash
!ls -lh /content/drive/MyDrive/ant-4b/runs/ant-08b-2026-10-21
!python - <<'PY'
import json
from pathlib import Path

path = Path('/content/drive/MyDrive/ant-4b/runs/ant-08b-2026-10-21')
metrics = json.loads((path / 'metrics.json').read_text())
print(json.dumps({k: metrics[k] for k in ('training', 'test', 'ood') if k in metrics}, indent=2))
PY
```

The full multitask test and OOD metrics are produced by `scripts/train.py` itself. **Do not use
`scripts/evaluate.py` for this comparison**: that script currently evaluates the separate
synthetic benchmark, not `data/ant-multitask`.

## 4. Download the trained ANT checkpoint

Zip only the run directory. The 0.8B backbone is not part of the run checkpoint.

```python
from google.colab import files

!cd /content/drive/MyDrive/ant-4b/runs && zip -rq /content/ant-08b-2026-10-21.zip ant-08b-2026-10-21
files.download('/content/ant-08b-2026-10-21.zip')
```

On the local machine, extract it into the repository's `runs/` directory. For example:

```bash
cd /path/to/ant-4b
unzip ~/Downloads/ant-08b-2026-10-21.zip -d runs/
```

The local 0.8B backbone weights are already at `models/Qwen3.5-0.8B`. If they are absent,
download them before profiling:

```bash
.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## 5. Profile locally and compare

The model-specific test/OOD results are in `runs/ant-08b-2026-10-21/metrics.json`. Profile
the local serving path on the same machine as the 4B reference:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/profile_inference.py \
  --checkpoint runs/ant-08b-2026-10-21 \
  --backbone models/Qwen3.5-0.8B \
  --device mps --dtype auto \
  --lengths 32 128 512 1024 --iters 3 --warmup 1
```

For a fair accuracy comparison, use the **same test JSONL, split, seed, and row count** used
for the reference checkpoint. The existing 4B gate result on the 40-row test sample was
accuracy `0.725` / ECE `0.1553`; do not compare it to a different split or to synthetic metrics.

## Notes and troubleshooting

- **Colab disconnects:** the best head is checkpointed to the Drive output directory when
  validation improves. Restarting from that head is not currently a supported resume workflow;
  if interrupted, rerun training to a new output directory rather than assuming optimizer state
  was saved.
- **CUDA out of memory:** first lower `--batch-size` and `--encode-batch-size` to `4`; keep the
  backbone frozen. Record the change so the run remains reproducible.
- **Non-finite training loss:** stop and preserve the complete traceback and runtime/GPU details;
  do not silently report the checkpoint as a successful variant.
- **Model ID:** use `Qwen/Qwen3.5-0.8B`, not `Qwen3.5-0.8B-Base`, for the first comparison.
- **Do not enable Metal quantization:** the 8-bit MPS path is numerically broken on the current
  stack (zeros/NaNs on the real 4B model); this Colab procedure uses plain CUDA weights.

## Expected deliverables

The Colab run directory should contain the ANT head/config, calibration and escalation files,
`metrics.json`, card metadata, and held-out examples. The returned comparison should report at
least accuracy and ECE on test/OOD, along with local warm latency at the requested lengths,
first-predict latency, dtype, and resident memory. Treat the 0.8B variant as a candidate until
both its test metrics and local serving behavior have been reviewed.
