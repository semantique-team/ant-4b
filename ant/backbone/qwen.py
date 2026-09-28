from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput, concat_outputs
from ant.utils import chunked, resolve_device, resolve_dtype

logger = logging.getLogger(__name__)


def _resolve_attr(config: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def resolve_text_config(config: Any) -> Any:
    """Multimodal Qwen checkpoints (e.g. qwen3_5) keep text settings in
    ``config.text_config``; text-only checkpoints expose them directly."""
    return getattr(config, "text_config", None) or config


def resolve_hidden_size(config: Any) -> int:
    hidden = _resolve_attr(resolve_text_config(config), ("hidden_size", "d_model"))
    if hidden is None:
        raise ValueError(f"could not resolve hidden_size from config of type {type(config).__name__}")
    return int(hidden)


def resolve_num_layers(config: Any) -> int:
    layers = _resolve_attr(resolve_text_config(config), ("num_hidden_layers", "num_layers"))
    return int(layers) if layers is not None else -1


def _looks_like_text_decoder(module: nn.Module) -> bool:
    return isinstance(getattr(module, "layers", None), nn.ModuleList) and hasattr(module, "norm")


def extract_text_decoder(model: nn.Module) -> nn.Module | None:
    """Return the text decoder submodule of a (possibly multimodal) model."""
    paths = (
        ("language_model",),
        ("text_model",),
        ("model", "language_model"),
        ("model", "text_model"),
        ("model",),
        ("transformer",),
        ("decoder",),
    )
    for path in paths:
        module: Any = model
        try:
            for attr in path:
                module = getattr(module, attr)
        except AttributeError:
            continue
        if _looks_like_text_decoder(module):
            return module
    return None


def _load_auto_model(model_id: str, dtype: torch.dtype, attn_implementation: str | None) -> nn.Module:
    from transformers import AutoModel

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return AutoModel.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    except TypeError:
        # newer transformers renamed torch_dtype to dtype
        return AutoModel.from_pretrained(model_id, dtype=dtype, **kwargs)


class QwenBackbone(nn.Module):
    """Text-encoder view of a Qwen checkpoint: last hidden states, no LM head.

    Loads the base model via ``AutoModel`` (never ``AutoModelForCausalLM``), so
    the LM head and the generation path are never part of ANT. For multimodal
    checkpoints the vision tower is dropped: only the text decoder is kept.
    Starts frozen; training modes (LoRA/partial/full) unfreeze it explicitly.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        *,
        torch_dtype: str | torch.dtype | None = "auto",
        device: str | torch.device | None = "auto",
        max_length: int | None = None,
        attn_implementation: str | None = None,
        text_only: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(torch_dtype, resolved_device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model = _load_auto_model(model_id, resolved_dtype, attn_implementation)
        self.config = model.config
        self._hidden_size = resolve_hidden_size(model.config)
        self._num_layers = resolve_num_layers(model.config)

        if text_only:
            decoder = extract_text_decoder(model)
            if decoder is not None and decoder is not model:
                dropped = sum(p.numel() for p in model.parameters()) - sum(
                    p.numel() for p in decoder.parameters()
                )
                logger.info(
                    "keeping text decoder only: dropped %.1fM parameters (vision tower / wrappers)",
                    dropped / 1e6,
                )
                model = decoder
            elif decoder is None:
                logger.warning("could not isolate a text decoder; using the full model")

        self.model = model
        self.model_id = model_id
        self.requires_grad_(False)

        model_max = getattr(self.tokenizer, "model_max_length", None)
        cap = 8192  # ANT contexts are short; the 262k window is never needed
        if isinstance(model_max, int) and 0 < model_max < cap:
            cap = model_max
        self.max_length = max_length or cap

        self.to(resolved_device)
        self.eval()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def summary(self) -> dict[str, Any]:
        text_config = resolve_text_config(self.config)
        params = next(self.model.parameters())
        layer_types = _resolve_attr(
            text_config, ("layer_types", "layer_type", "layers_block_type")
        )
        return {
            "model_id": self.model_id,
            "model_type": getattr(self.config, "model_type", None),
            "architectures": getattr(self.config, "architectures", None),
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "vocab_size": _resolve_attr(text_config, ("vocab_size",)),
            "max_position_embeddings": _resolve_attr(text_config, ("max_position_embeddings",)),
            "layer_types": layer_types,
            "dtype": str(params.dtype),
            "device": str(params.device),
            "param_count_M": sum(p.numel() for p in self.model.parameters()) / 1e6,
        }

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        return hidden

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> BackboneOutput:
        length = max_length or self.max_length
        outputs: list[BackboneOutput] = []
        grad_enabled = any(p.requires_grad for p in self.model.parameters())
        for chunk in chunked(list(texts), batch_size):
            tokens = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=length, return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(self.model.device)
            attention_mask = tokens["attention_mask"].to(self.model.device)
            with torch.set_grad_enabled(grad_enabled):
                hidden = self._forward(input_ids, attention_mask)
            outputs.append(BackboneOutput(hidden_states=hidden, attention_mask=attention_mask))
        return concat_outputs(outputs)

    def trainable_summary(self) -> str:
        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
        return f"{trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total"
