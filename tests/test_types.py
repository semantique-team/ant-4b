from __future__ import annotations

import pytest

from ant.types import DecisionExample, DecisionInput


def test_decision_input_coerces_candidates_to_tuple():
    item = DecisionInput(state="s", question="q", candidates=["a", "b"])
    assert item.candidates == ("a", "b")
    assert item.n_candidates == 2


def test_decision_input_rejects_empty():
    with pytest.raises(ValueError):
        DecisionInput(state="s", question="q", candidates=[])
    with pytest.raises(ValueError):
        DecisionInput(state="", question="q", candidates=["a"])


def test_decision_example_requires_correct_in_candidates():
    with pytest.raises(ValueError):
        DecisionExample(state="s", question="q", candidates=["a", "b"], correct="c")
    example = DecisionExample(state="s", question="q", candidates=["a", "b"], correct="a")
    assert example.correct == "a"
