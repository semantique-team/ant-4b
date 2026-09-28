from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ant.model import ANTModel

TRAINING_MODES = ("head_only", "lora", "partial", "full")

DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _head_parameters(model: ANTModel) -> list[nn.Parameter]:
    return model.head_parameters()


def _unfreeze_head(model: ANTModel) -> None:
    for parameter in _head_parameters(model):
        parameter.requires_grad_(True)


def _decoder_layers(backbone: nn.Module) -> nn.ModuleList | None:
    for module in (getattr(backbone, "model", None), backbone):
        if module is None:
            continue
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    return None


def set_training_mode(
    model: ANTModel,
    mode: str,
    *,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    unfreeze_last_k_layers: int = 4,
) -> list[nn.Parameter]:
    """Configure what trains. Returns the trainable parameters.

    head_only: frozen backbone + trainable projections/scorer (v0.1 default).
    lora: PEFT adapters on the text decoder + head.
    partial: last-k decoder layers unfrozen + head.
    full: everything unfrozen.
    """
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}, got {mode!r}")

    model.backbone.requires_grad_(False)
    _unfreeze_head(model)

    if mode == "head_only":
        return _head_parameters(model)

    if mode == "full":
        model.backbone.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    if mode == "partial":
        model.backbone.requires_grad_(True)
        layers = _decoder_layers(model.backbone)
        if layers is None:
            raise RuntimeError("could not locate decoder layers for partial fine-tuning")
        keep = len(layers) - max(1, unfreeze_last_k_layers)
        for layer in list(layers)[:keep]:
            layer.requires_grad_(False)
        for name, module in model.backbone.named_children():
            if name != "layers" and "embed" in name:
                module.requires_grad_(False)
        return [p for p in model.parameters() if p.requires_grad]

    # mode == "lora"
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA mode requires peft; install ant-4b[train]") from error
    if type(getattr(model.backbone, "model", None)).__name__ == "PeftModel":
        return [p for p in model.parameters() if p.requires_grad]
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
        target_modules=list(lora_targets),
    )
    model.backbone.model = get_peft_model(model.backbone.model, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attached but no trainable parameters found")
    return trainable
