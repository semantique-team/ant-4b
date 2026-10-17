from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ant.model import ANTModel
from ant.training.data import DecisionDataset, decision_loader
from ant.training.losses import combined_loss
from ant.training.modes import set_training_mode
from ant.utils import count_parameters, supports_bf16


@dataclass
class TrainingConfig:
    mode: str = "head_only"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    amp: bool | None = None  # None: bf16 autocast on Ampere+ CUDA only
    seed: int = 42
    eval_every: int = 1
    log_every: int = 25
    num_workers: int = 0
    checkpoint_dir: str | None = None
    rank_weight: float = 0.0
    margin: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    unfreeze_last_k_layers: int = 4


@dataclass
class TrainingReport:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    trainable_parameters: int = 0
    total_parameters: int = 0
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class Trainer:
    def __init__(self, model: ANTModel, config: TrainingConfig | None = None, **overrides) -> None:
        self.model = model
        self.config = config or TrainingConfig(**overrides)
        self.device = next(model.backbone.parameters()).device

    def fit(
        self,
        train_dataset: DecisionDataset,
        val_dataset: DecisionDataset | None = None,
    ) -> TrainingReport:
        cfg = self.config
        torch.manual_seed(cfg.seed)

        trainable = set_training_mode(
            self.model,
            cfg.mode,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            unfreeze_last_k_layers=cfg.unfreeze_last_k_layers,
        )
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        train_loader = decision_loader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed,
            num_workers=cfg.num_workers,
        )
        steps_per_epoch = max(1, len(train_loader))
        total_steps = cfg.epochs * steps_per_epoch
        warmup = max(1, int(cfg.warmup_ratio * total_steps))

        def lr_multiplier(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        use_amp = cfg.amp if cfg.amp is not None else supports_bf16(self.device)

        trainable_count, total_count = count_parameters(self.model)
        report = TrainingReport(trainable_parameters=trainable_count, total_parameters=total_count)

        step = 0
        best_head: dict[str, torch.Tensor] | None = None
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            # the backbone stays in eval mode: frozen BatchNorm-free layers behave
            # identically, but dropout inside a partially unfrozen Qwen must not
            # perturb frozen-encoder reproducibility
            self.model.backbone.eval()
            epoch_start, epoch_loss, epoch_examples = time.time(), 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    scores, mask = self.model(batch.states, batch.questions, batch.candidates)
                    loss_output = combined_loss(
                        scores,
                        batch.labels.to(scores.device),
                        mask.to(scores.device),
                        ce_weight=1.0,
                        rank_weight=cfg.rank_weight,
                        margin=cfg.margin,
                    )
                loss = loss_output.total
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {step + 1}; with a float16 backbone this "
                        "means activation overflow: retry with --dtype bfloat16"
                    )
                loss.backward()
                if cfg.max_grad_norm:
                    nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                step += 1
                epoch_loss += float(loss) * len(batch.states)
                epoch_examples += len(batch.states)
                if cfg.log_every and step % cfg.log_every == 0:
                    components = " ".join(f"{k}={v:.4f}" for k, v in loss_output.components.items())
                    print(
                        f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.4f} "
                        f"{components} lr {scheduler.get_last_lr()[0]:.2e}"
                    )

            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, epoch_examples),
                "seconds": time.time() - epoch_start,
            }
            if val_dataset is not None and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
                val_loss, val_accuracy = self.evaluate(val_dataset)
                entry.update({"val_loss": val_loss, "val_accuracy": val_accuracy})
                if val_loss < report.best_val_loss:
                    report.best_val_loss = val_loss
                    report.best_val_accuracy = val_accuracy
                    report.best_epoch = epoch
                    if cfg.mode == "head_only":
                        best_head = {
                            name: tensor.detach().clone()
                            for name, tensor in self.model.head_state_dict().items()
                        }
                    if cfg.checkpoint_dir:
                        report.checkpoint_path = str(self.model.save_head(cfg.checkpoint_dir))
            report.history.append(entry)
            summary = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items())
            print(f"[epoch {epoch}] {summary}")

        if best_head is not None:
            # calibration and the saved checkpoint must describe the best epoch,
            # not whatever the last (possibly overfit) epoch left behind
            self.model.load_head_state_dict(best_head)

        if cfg.checkpoint_dir and report.checkpoint_path is None:
            report.checkpoint_path = str(self.model.save_head(cfg.checkpoint_dir))
        if cfg.checkpoint_dir:
            report.save(Path(cfg.checkpoint_dir) / "training_report.json")
        return report

    def evaluate(self, dataset: DecisionDataset) -> tuple[float, float]:
        self.model.eval()
        loader = decision_loader(dataset, batch_size=self.config.batch_size, shuffle=False)
        total_loss, total, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                scores, mask = self.model(batch.states, batch.questions, batch.candidates)
                labels = batch.labels.to(scores.device)
                loss = nn.functional.cross_entropy(scores.float(), labels)
                total_loss += float(loss) * len(batch.states)
                correct += int((scores.argmax(dim=1) == labels).sum())
                total += len(batch.states)
        return total_loss / max(1, total), correct / max(1, total)
