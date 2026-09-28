from __future__ import annotations

import torch

from ant.training.modes import set_training_mode


def test_head_only_mode_freezes_backbone(model, examples):
    trainable = set_training_mode(model, "head_only")
    loss = model.loss(examples)
    loss.backward()

    assert trainable
    assert all(not p.requires_grad for p in model.backbone.parameters())
    head_with_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert head_with_grad
    backbone_grads = [n for n, p in model.backbone.named_parameters() if p.grad is not None]
    assert backbone_grads == []


def test_partial_mode_unfreezes_last_layers(model, examples):
    set_training_mode(model, "partial", unfreeze_last_k_layers=1)
    assert model.loss(examples).requires_grad
    layers = model.backbone.layers
    frozen = [p.requires_grad for layer in layers[:-1] for p in layer.parameters()]
    trainable = [p.requires_grad for p in layers[-1].parameters()]
    assert not any(frozen)
    assert any(trainable)


def test_full_mode_unfreezes_everything(model):
    set_training_mode(model, "full")
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_unknown_mode_raises(model):
    import pytest

    with pytest.raises(ValueError):
        set_training_mode(model, "bogus")
