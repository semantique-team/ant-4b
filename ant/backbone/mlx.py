from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ant.backbone.base import BackboneOutput, concat_outputs
from ant.utils import bucket_max_length, chunked, resolve_device

logger = logging.getLogger(__name__)


class MlxBackbone(nn.Module):
    """Apple Silicon Qwen encoder backed by ``mlx-lm``.

    This is an inference backend only: the frozen model runs in MLX and its
    hidden states are copied into torch tensors so the existing ANT head,
    calibration and checkpoint format remain unchanged. Use ``backbone_backend
    = "mlx"`` with ``quantization=None`` for an unquantized MLX model, or
    ``quantization="mlx"`` for an MLX checkpoint already packed at the requested
    bits/group size (create one with ``scripts/mlx_convert.py``).
    """

    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: str | torch.dtype | None = "auto",
        device: str | torch.device | None = "auto",
        max_length: int | None = None,
        quantization: str | None = None,
        quantization_bits: int = 8,
        quantization_group_size: int = 64,
    ) -> None:
        super().__init__()
        if quantization not in (None, "mlx"):
            raise ValueError(
                f"quantization must be None or 'mlx' for the MLX backend, got {quantization!r}"
            )
        if quantization == "mlx" and quantization_bits not in (2, 4, 8):
            raise ValueError("quantization_bits must be one of 2, 4 or 8")

        try:
            import mlx.core as mx  # noqa: F401
            import mlx_lm
        except ImportError as error:
            raise ImportError(
                "backbone_backend='mlx' requires Apple Silicon MLX packages; "
                "install them with: pip install mlx mlx-lm"
            ) from error

        resolved_device = resolve_device(device)
        if resolved_device.type not in ("cpu", "mps"):
            raise ValueError("the MLX backend runs on Apple Silicon; use device='auto' or 'mps'")

        self.model_id = model_id
        self.quantization = quantization
        self.quantization_bits = quantization_bits
        self.quantization_group_size = quantization_group_size
        self.torch_dtype = torch_dtype
        self._mx = mx
        self._mlx_lm = mlx_lm

        model, tokenizer, model_config = mlx_lm.load(model_id, return_config=True)
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self._model_config = model_config
        if hasattr(self.model, "eval"):
            self.model.eval()
        text_config = model_config.get("text_config", model_config)
        hidden_size = text_config.get("hidden_size", text_config.get("d_model"))
        num_layers = text_config.get("num_hidden_layers", text_config.get("num_layers"))
        if hidden_size is None or num_layers is None:
            raise ValueError(f"could not read hidden_size/num_layers from {model_id}")
        self._hidden_size = int(hidden_size)
        self._num_layers = int(num_layers)

        if quantization == "mlx":
            self._require_requested_quantization(model_config)

        # The MLX weights are not torch parameters, but ANTModel/Trainer expect
        # one device-bearing parameter to place the small torch head on.
        self._placeholder = nn.Parameter(torch.empty(0, device=resolved_device), requires_grad=False)
        self.max_length = max_length or 8192
        self.eval()

    def _require_requested_quantization(self, model_config: dict[str, Any]) -> None:
        quantization = model_config.get("quantization") or model_config.get("quantization_config")
        if not isinstance(quantization, dict):
            raise RuntimeError(
                f"{self.model_id} is not an MLX-quantized checkpoint; run "
                "scripts/mlx_convert.py or pass an MLX quantized model id"
            )
        actual_bits = int(quantization.get("bits", -1))
        actual_group = int(quantization.get("group_size", -1))
        if actual_bits != self.quantization_bits or actual_group != self.quantization_group_size:
            raise RuntimeError(
                f"{self.model_id} is {actual_bits}-bit/group {actual_group}, but the request is "
                f"{self.quantization_bits}-bit/group {self.quantization_group_size}; "
                "convert a matching checkpoint with scripts/mlx_convert.py"
            )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def summary(self) -> dict[str, Any]:
        quantization = self._model_config.get("quantization") or self._model_config.get(
            "quantization_config"
        )
        return {
            "backend": "mlx",
            "model_id": self.model_id,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "quantization": quantization,
            "requested_quantization": self.quantization,
            "torch_device": str(self._placeholder.device),
        }

    def _text_decoder(self) -> Any:
        """Return the module that emits hidden states, never vocabulary logits."""
        candidate = getattr(self.model, "model", None)
        if callable(candidate):
            return candidate
        language_model = getattr(self.model, "language_model", self.model)
        return getattr(language_model, "model", language_model)

    def _token_batch(self, texts: Sequence[str], cap: int) -> tuple[np.ndarray, np.ndarray]:
        # TokenizerWrapper exposes encode() but is not callable like a raw HF tokenizer.
        rows = [
            [int(token) for token in self.tokenizer.encode(text, truncation=True, max_length=cap)]
            for text in texts
        ]
        width = max((len(row) for row in rows), default=1)
        target = bucket_max_length([width])
        if target > cap:
            target = width
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        ids = np.full((len(rows), target), pad_id, dtype=np.int32)
        mask = np.zeros((len(rows), target), dtype=np.int32)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = 1
        return ids, mask

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> BackboneOutput:
        length = max_length or self.max_length
        outputs: list[BackboneOutput] = []
        decoder = self._text_decoder()
        for chunk in chunked(list(texts), batch_size):
            ids_np, mask_np = self._token_batch(chunk, cap=length)
            hidden = decoder(self._mx.array(ids_np))
            # MLX is lazy: force evaluation before leaving its graph.
            self._mx.eval(hidden)
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            hidden_np = np.asarray(hidden.astype(self._mx.float32))
            outputs.append(
                BackboneOutput(
                    hidden_states=torch.from_numpy(hidden_np),
                    attention_mask=torch.from_numpy(mask_np.astype(np.int64)),
                )
            )
        return concat_outputs(outputs)


__all__ = ["MlxBackbone"]
