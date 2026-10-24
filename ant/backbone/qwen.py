from __future__ import annotations

import functools
import logging
from typing import Any, Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput, concat_outputs
from ant.utils import bucket_max_length, chunked, resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

# The pristine transformers GatedDeltaNet torch kernel, captured the first time a
# chunk-size patch is applied. Keeping our own reference is what makes the patch
# reversible and idempotent: reading the module attribute again would return our
# wrapper, and a second wrap would multiply the Python-level call overhead.
_original_chunk_gated_delta_rule: Any = None
_active_chunk_size: int | None = None


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


def build_quantization_config(
    quantization: str | None, bits: int, group_size: int
) -> "MetalConfig | None":
    """Build the transformers quantization config for the frozen backbone.

    ``None`` means "keep the checkpoint dtype". ``"metal"`` selects the native
    MPS quantizer, which packs weights into uint32 and fuses dequantization into
    the matmul: the 9.3 GB FP16 backbone drops to roughly 5.4 GB at 8 bits (the
    tied embedding stays FP16 at 1.27 GB) and stops paging to swap.

    Every optional dependency is imported lazily so that importing ``ant`` works
    on Linux/CUDA machines and on hosts without the ``kernels`` package; only
    actually asking for Metal quantization requires them.
    """
    if quantization is None:
        return None
    if quantization != "metal":
        if quantization == "mlx":
            raise ValueError(
                "quantization='mlx' belongs to ant.backbone.mlx, not the torch Qwen backbone"
            )
        raise ValueError(
            f"unknown quantization {quantization!r}; allowed values are None and 'metal'"
        )

    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        raise RuntimeError(
            "quantization='metal' requires an Apple Silicon GPU with MPS available; "
            "use quantization=None on this machine"
        )

    from transformers.utils import is_kernels_available

    if not is_kernels_available():
        raise ImportError(
            "quantization='metal' needs the Hugging Face kernels package that provides "
            "the fused dequant+matmul; install it with: pip install kernels"
        )

    from transformers.utils.quantization_config import MetalConfig

    return MetalConfig(bits=bits, group_size=group_size)


def _chunk_gated_delta_rule_with_default(
    func: Any, default_size: int, *args: Any, **kwargs: Any
) -> Any:
    """Call ``func`` with ``chunk_size=default_size`` unless the caller set one."""
    kwargs.setdefault("chunk_size", default_size)
    return func(*args, **kwargs)


def apply_linear_attention_chunk_size(size: int | None) -> None:
    """Override the default ``chunk_size`` of Qwen3.5's GatedDeltaNet torch kernel.

    On an Apple M3 the pure-PyTorch fallback at L=1024 is ~1.3x faster with
    chunk_size=128 than with the transformers default of 64: doubling the chunk
    halves the Python-side loop iterations and the per-chunk state bookkeeping,
    which dominates when no fused FLA kernel is installed.

    The override is process-wide because it rebinds a module-level function. It
    is idempotent (the original is recorded once, so repeated calls never stack
    wrappers) and reversible: passing ``None`` restores the exact original
    callable. An explicit ``chunk_size`` passed by a caller still wins.
    Model families without a ``qwen3_5`` module are left untouched.
    """
    global _original_chunk_gated_delta_rule, _active_chunk_size

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except ImportError:
        logger.debug("transformers.models.qwen3_5 is absent; chunk-size override skipped")
        return

    if size is None:
        if _original_chunk_gated_delta_rule is not None:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = _original_chunk_gated_delta_rule
            _original_chunk_gated_delta_rule = None
            _active_chunk_size = None
            logger.info("linear-attention chunk size restored to the transformers default")
        return

    if size <= 0:
        raise ValueError("linear_attention_chunk_size must be positive")

    if _active_chunk_size == size:
        return

    if _original_chunk_gated_delta_rule is None:
        _original_chunk_gated_delta_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule

    modeling_qwen3_5.torch_chunk_gated_delta_rule = functools.partial(
        _chunk_gated_delta_rule_with_default,
        _original_chunk_gated_delta_rule,
        size,
    )
    _active_chunk_size = size
    logger.info("linear-attention chunk size set to %d", size)


def _auto_from_pretrained(model_id: str, dtype: torch.dtype, **kwargs: Any) -> nn.Module:
    """``from_pretrained`` with an explicit dtype across transformers versions."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        # transformers < 4.56 calls the argument torch_dtype
        return AutoModel.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def _load_auto_model(
    model_id: str,
    dtype: torch.dtype,
    attn_implementation: str | None,
    quantization_config: Any | None = None,
) -> nn.Module:
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    # Quantized weights must be placed while loading: device_map puts the packed
    # uint32 tensors on MPS directly, and a later .to() would treat them as
    # ordinary floats and corrupt them.
    #
    # The dtype has to be explicit here too. Without it the checkpoint's own
    # config decides, and Qwen3.5-4B declares bfloat16; MPS has no native bf16
    # support, so PyTorch emulates it and the forward pass measured roughly 4x
    # slower than fp16 with identical weights.
    from transformers import AutoConfig

    declared = getattr(AutoConfig.from_pretrained(model_id), "quantization_config", None)
    if declared is not None:
        # A checkpoint saved in quantized form already carries its own scheme;
        # passing ours again would try to quantize packed weights. Passing
        # device_map is still required: without it the loader stages every tensor
        # on CPU and ships it to MPS one Metal copy at a time (measured: the load
        # never finishes), while a plain load of an fp16 checkpoint needs no
        # device_map at all. No full-precision shard is ever read on this path,
        # which is what makes repeated use safe on 16 GB machines.
        logger.info(
            "checkpoint %s is already quantized; using its embedded quantization config",
            model_id,
        )
        kwargs["device_map"] = "mps"
        return _auto_from_pretrained(model_id, dtype, **kwargs)

    if quantization_config is None:
        # Historical note: plain loads must NOT go through device_map on this
        # stack. Loads staged via accelerate + MPS copies die silently
        # (EXC_BREAKPOINT in malloc) at the first shard; the plain CPU-then-.to()
        # path is slower in memory but has never crashed.
        return _auto_from_pretrained(model_id, dtype, **kwargs)
    kwargs["device_map"] = "mps"
    return _auto_from_pretrained(
        model_id, dtype, quantization_config=quantization_config, **kwargs
    )


class QwenBackbone(nn.Module):
    """Text-encoder view of a Qwen checkpoint: last hidden states, no LM head.

    Loads the base model via ``AutoModel`` (never ``AutoModelForCausalLM``), so
    the LM head and the generation path are never part of ANT. For multimodal
    checkpoints the vision tower is dropped: only the text decoder is kept.
    Optional Metal quantization keeps the frozen weights resident on MPS; the
    quantized model is placed by ``device_map`` at load time and never moved
    afterwards. Starts frozen; training modes (LoRA/partial/full) unfreeze it
    explicitly.
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
        quantization: str | None = None,
        quantization_bits: int = 8,
        quantization_group_size: int = 64,
        linear_attention_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(torch_dtype, resolved_device)
        quantization_config = build_quantization_config(
            quantization, quantization_bits, quantization_group_size
        )
        if quantization_config is not None and resolved_device.type != "mps":
            raise ValueError(
                f"quantization={quantization!r} always places the model on MPS, but the "
                f"device resolved to {resolved_device.type!r}; pass device='mps' or "
                "disable quantization"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model = _load_auto_model(
            model_id, resolved_dtype, attn_implementation, quantization_config
        )
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

        if getattr(model, "hf_quantizer", None) is not None:
            # device_map (ours, or the one the quantizer installs for a
            # pre-quantized checkpoint) already placed the weights while loading.
            # A post-hoc .to() on a quantized module would reinterpret the packed
            # uint32 payload as floats and silently corrupt every weight.
            logger.debug("quantized backbone already placed by device_map; skipping .to()")
        else:
            self.to(resolved_device)

        if linear_attention_chunk_size is not None:
            apply_linear_attention_chunk_size(linear_attention_chunk_size)

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
            input_ids, attention_mask = self._pad_to_bucket(
                input_ids, attention_mask, cap=length
            )
            with torch.set_grad_enabled(grad_enabled):
                hidden = self._forward(input_ids, attention_mask)
            outputs.append(BackboneOutput(hidden_states=hidden, attention_mask=attention_mask))
        return concat_outputs(outputs)

    def _pad_to_bucket(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *, cap: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a freshly tokenized batch up to the next width bucket.

        Right padding with a zeroed mask leaves real-token activations untouched
        under causal attention, so this is numerically a no-op; it only keeps the
        tensor shape stable so MPS reuses its compiled kernels. Padding is skipped
        when it would exceed the truncation cap, because that would change the
        padded length the caller asked for without any benefit.
        """
        padded = int(input_ids.shape[1])
        target = bucket_max_length([padded])
        if target <= padded or target > cap:
            return input_ids, attention_mask
        pad_width = target - padded
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return input_ids, attention_mask
        return (
            torch.nn.functional.pad(input_ids, (0, pad_width), value=pad_id),
            torch.nn.functional.pad(attention_mask, (0, pad_width), value=0),
        )

    def trainable_summary(self) -> str:
        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
        return f"{trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total"
