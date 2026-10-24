from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from ant.backbone.base import BackboneOutput
from ant.config import ANTConfig
from ant.scoring import build_scorer
from ant.types import DecisionExample

MASK_SCORE = -1.0e9


def _build_default_backbone(config: ANTConfig) -> nn.Module:
    """Construct the frozen encoder described by ``config``.

    Shared by ``ANTModel.__init__`` and ``ANTModel.load`` so a checkpoint that
    recorded quantization settings is re-created with the same backbone rather
    than silently loading 9.3 GB of fp16 weights.
    """
    if config.backbone_model == "mock":
        from ant.backbone.mock import MockBackbone

        return MockBackbone(hidden_size=config.scoring_dim * 2)
    if config.backbone_backend == "mlx":
        from ant.backbone.mlx import MlxBackbone

        return MlxBackbone(
            config.backbone_model,
            torch_dtype=config.torch_dtype,
            device=config.device,
            max_length=config.max_context_tokens,
            quantization=config.quantization,
            quantization_bits=config.quantization_bits,
            quantization_group_size=config.quantization_group_size,
        )
    from ant.backbone.qwen import QwenBackbone

    return QwenBackbone(
        config.backbone_model,
        torch_dtype=config.torch_dtype,
        device=config.device,
        max_length=config.max_context_tokens,
        attn_implementation=config.attn_implementation,
        quantization=config.quantization,
        quantization_bits=config.quantization_bits,
        quantization_group_size=config.quantization_group_size,
        linear_attention_chunk_size=config.linear_attention_chunk_size,
    )


class ANTModel(nn.Module):
    """Candidate-scoring decision model.

    score(state, question, candidate) -> scalar, computed as
    f_theta(C(S, Q), E(O_i)); no LM head, no generation, ever.
    """

    def __init__(self, config: ANTConfig, *, backbone: nn.Module | None = None) -> None:
        super().__init__()
        if backbone is None:
            backbone = _build_default_backbone(config)
        self.config = config
        self.backbone = backbone

        from ant.encoding.candidate_encoder import CandidateEncoder
        from ant.encoding.context_encoder import ContextEncoder

        self.context_encoder = ContextEncoder(
            backbone,
            pooling=config.context_pooling,
            proj_dim=config.scoring_dim,
            dropout=config.dropout,
            template=config.context_template,
        )
        self.candidate_encoder = CandidateEncoder(
            backbone,
            pooling=config.candidate_pooling,
            proj_dim=config.scoring_dim,
            dropout=config.dropout,
            template=config.candidate_template,
        )
        self.scorer = build_scorer(config.scorer, config.scoring_dim)

        device = next(backbone.parameters()).device
        self.context_encoder.to(device)
        self.candidate_encoder.to(device)
        self.scorer.to(device)

    @classmethod
    def tiny(cls, **overrides) -> "ANTModel":
        """CPU-friendly model on a mock backbone, for tests and demos."""
        from ant.backbone.mock import MockBackbone

        config = ANTConfig(
            backbone_model="mock",
            scoring_dim=overrides.pop("scoring_dim", 32),
            context_pooling=overrides.pop("context_pooling", "mean"),
            candidate_pooling=overrides.pop("candidate_pooling", "mean"),
            scorer=overrides.pop("scorer", "dot"),
            **overrides,
        )
        backbone = MockBackbone(hidden_size=config.scoring_dim * 2)
        return cls(config, backbone=backbone)

    def encode_contexts(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        return self.context_encoder(
            states,
            questions,
            batch_size=batch_size or self.config.encode_batch_size,
            max_length=self.config.max_context_tokens,
        )

    def encode_candidates(
        self,
        candidates: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        return self.candidate_encoder(
            candidates,
            batch_size=batch_size or self.config.encode_batch_size,
            max_length=self.config.max_candidate_tokens,
        )

    def score_embeddings(self, z_context: torch.Tensor, z_candidates: torch.Tensor) -> torch.Tensor:
        return self.scorer(z_context, z_candidates)

    def encode_state_and_candidates(
        self, state: str, question: str, candidates: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One backbone pass for the context and every candidate.

        Returns (z_context [1, d], z_candidates [N, d]). Both encoders share the
        backbone, so batching them together removes a whole pass over the frozen
        weights — the dominant cost on memory-bound devices.

        One cap governs the whole batch, so candidates are truncated at
        ``max_context_tokens`` rather than ``max_candidate_tokens``. That only
        matters for candidates longer than 128 tokens, which the head never saw
        in training; contexts are unaffected because they are the longest row.
        """
        if not candidates:
            raise ValueError("at least one candidate is required")
        context_text = self.context_encoder.format(state, question)
        candidate_texts = [self.candidate_encoder.format(candidate) for candidate in candidates]
        output = self.backbone.encode(
            [context_text, *candidate_texts],
            batch_size=len(candidate_texts) + 1,
            max_length=self.config.max_context_tokens,
        )
        # Right padding plus causal attention means the fused batch does not
        # change any real token: pad slots are masked out of attention, and the
        # decoder numbers positions 0..n-1 from the left, so a shorter row keeps
        # its own positions regardless of how wide the batch is padded. Widening
        # to the longest text only reorders floating-point accumulation.
        context_output = BackboneOutput(
            hidden_states=output.hidden_states[:1],
            attention_mask=output.attention_mask[:1],
        )
        candidate_output = BackboneOutput(
            hidden_states=output.hidden_states[1:],
            attention_mask=output.attention_mask[1:],
        )

        # mirror ContextEncoder.forward / CandidateEncoder.forward exactly
        z_context = self.context_encoder.dropout(
            self.context_encoder.projection(
                self.context_encoder.input_norm(
                    self.context_encoder.pool_hidden(context_output).float()
                )
            )
        )
        z_candidates = self.candidate_encoder.dropout(
            self.candidate_encoder.projection(
                self.candidate_encoder.input_norm(
                    self.candidate_encoder.pool_hidden(candidate_output).float()
                )
            )
        )
        return z_context.float(), z_candidates.float()

    def forward(
        self,
        states: Sequence[str],
        questions: Sequence[str],
        candidates: Sequence[Sequence[str]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of decisions with ragged candidate sets.

        Returns (scores [B, N_max] with padded slots at MASK_SCORE, mask [B, N_max]).
        Candidates are deduplicated across the batch before encoding.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        unique = sorted({candidate for row in candidates for candidate in row})
        lookup = {candidate: i for i, candidate in enumerate(unique)}
        z_unique = self.encode_candidates(unique)

        z_context = self.encode_contexts(states, questions)
        width = max(len(row) for row in candidates)
        z_batch = z_unique.new_zeros(len(states), width, z_unique.size(-1))
        mask = torch.zeros(len(states), width, dtype=torch.bool, device=z_unique.device)
        for i, row in enumerate(candidates):
            index = torch.tensor([lookup[c] for c in row], device=z_unique.device)
            z_batch[i, : len(row)] = z_unique[index]
            mask[i, : len(row)] = True

        scores = self.scorer(z_context, z_batch).masked_fill(~mask, MASK_SCORE)
        return scores, mask

    def score_candidates(
        self, state: str, question: str, candidates: Sequence[str]
    ) -> torch.Tensor:
        """Low-level primitive: raw score per candidate, input order preserved."""
        scores, _ = self.forward([state], [question], [list(candidates)])
        return scores[0]

    def loss(self, examples: Sequence[DecisionExample]) -> torch.Tensor:
        states = [e.state for e in examples]
        questions = [e.question for e in examples]
        candidate_rows = [list(e.candidates) for e in examples]
        scores, mask = self.forward(states, questions, candidate_rows)
        labels = torch.tensor(
            [row.index(e.correct) for row, e in zip(candidate_rows, examples)],
            device=scores.device,
        )
        return nn.functional.cross_entropy(scores.float(), labels)

    # -- checkpointing: only the head is persisted, the backbone stays on the Hub --

    def head_parameters(self) -> list[nn.Parameter]:
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("backbone.") and ".backbone." not in name
        ]

    def head_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, tensor)
            for name, tensor in self.state_dict().items()
            if not name.startswith("backbone.") and ".backbone." not in name
        )

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        backbone_missing = all(".backbone." in k or k.startswith("backbone.") for k in missing)
        if not backbone_missing or unexpected:
            raise RuntimeError(
                f"head checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
            )

    def save_head(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        blob = {
            "state_dict": self.head_state_dict(),
            "config": self.config.to_dict(),
            "backbone_hidden_size": self.backbone.hidden_size,
        }
        if self.config.backbone_model == "mock":
            # a mock backbone has no Hub checkpoint to reload from, so persist it
            blob["mock_backbone_state"] = self.backbone.state_dict()
        self._save_blob(blob, path / "ant_head.pt")
        self.config.save(path / "config.json")
        self._maybe_save_lora(path)
        return path

    def _save_blob(self, blob: dict, destination: Path) -> None:
        # Atomic write: a Colab disconnect must not leave a truncated head file.
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(blob, temporary)
        temporary.replace(destination)

    def _maybe_save_lora(self, path: Path) -> None:
        inner = getattr(self.backbone, "model", None)
        if inner is None or type(inner).__name__ != "PeftModel":
            return
        wrapped = getattr(inner, "base_model", None)
        target = getattr(wrapped, "model", inner)
        target.save_pretrained(path / "lora_adapter")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backbone: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
        device: str = "auto",
        quantization: str | None = None,
    ) -> "ANTModel":
        """Load a head checkpoint. ``device`` is where it runs now: the device saved in
        the checkpoint is where it was *trained* (e.g. cuda on Colab) and is ignored.

        ``quantization`` overrides the value recorded in the checkpoint, so an
        existing fp16 checkpoint can be benchmarked at 8 bits without editing
        ``config.json``.
        """
        path = Path(path)
        blob = torch.load(path / "ant_head.pt", map_location=map_location, weights_only=False)
        saved = ANTConfig.from_dict(blob["config"])
        config = replace(
            saved,
            device=device,
            quantization=quantization or saved.quantization,
        )
        if backbone is None:
            if config.backbone_model == "mock":
                from ant.backbone.mock import MockBackbone

                backbone = MockBackbone(hidden_size=blob.get("backbone_hidden_size", 64))
                if "mock_backbone_state" in blob:
                    backbone.load_state_dict(blob["mock_backbone_state"])
            else:
                backbone = _build_default_backbone(config)
        model = cls(config, backbone=backbone)
        model.load_head_state_dict(blob["state_dict"])
        model._maybe_load_lora(path)
        return model

    def _maybe_load_lora(self, path: Path) -> None:
        adapter_dir = path / "lora_adapter"
        if not adapter_dir.exists() or not hasattr(self.backbone, "model"):
            return
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("checkpoint contains a LoRA adapter; install ant-4b[train]") from error
        self.backbone.model = PeftModel.from_pretrained(self.backbone.model, adapter_dir)

    def summary(self) -> dict:
        trainable, total = 0, 0
        for name, p in self.named_parameters():
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
        return {
            "scorer": self.config.scorer,
            "scoring_dim": self.config.scoring_dim,
            "context_pooling": self.config.context_pooling,
            "candidate_pooling": self.config.candidate_pooling,
            "head_parameters_M": trainable / 1e6,
            "total_parameters_M": total / 1e6,
        }
