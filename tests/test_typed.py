from __future__ import annotations

import pytest

from ant.inference.typed import (
    MAX_QUESTIONS,
    TypedQuestionError,
    answer_question,
    answer_questions,
)


@pytest.fixture
def ant(model):
    from ant.inference.predictor import ANTPredictor

    return ANTPredictor(model)


STATE = "The customer was charged twice for the same invoice and asks for a refund."


def test_choice_question(ant):
    answer = answer_question(
        ant,
        STATE,
        {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "invoices and refunds", "technical": "bugs"},
        },
    )
    assert answer["type"] == "choice"
    assert answer["choice"] in {"billing", "technical"}
    assert set(answer["probabilities"]) == {"billing", "technical"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-5
    assert 0.0 <= answer["confidence"] <= 1.0
    assert answer["action"] in {"execute", "escalate"}


def test_choice_accepts_plain_option_list(ant):
    answer = answer_question(
        ant, STATE, {"type": "choice", "instructions": "Pick one", "criteria": ["a", "b"]}
    )
    assert answer["choice"] in {"a", "b"}
    assert set(answer["probabilities"]) == {"a", "b"}


def test_noul_question(ant):
    answer = answer_question(
        ant,
        STATE,
        {
            "type": "noul",
            "instructions": "Is the customer asking for money back?",
            "criteria": {"true": "asks for a refund", "false": "no refund mentioned"},
        },
    )
    assert answer["type"] == "noul"
    assert 0.0 <= answer["noul"] <= 1.0
    assert set(answer["probabilities"]) == {"yes", "no"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-5
    assert answer["noul"] == pytest.approx(answer["probabilities"]["yes"])


def test_score_question(ant):
    answer = answer_question(
        ant,
        STATE,
        {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
    )
    assert answer["type"] == "score"
    assert 0.0 <= answer["score"] <= 2.0
    assert set(answer["legend"].values()) == {"not urgent", "soon", "blocking"}
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    expected = sum(int(level) * p for level, p in answer["probabilities"].items())
    assert answer["score"] == pytest.approx(expected)


def test_answer_questions_keys_by_id(ant):
    answers = answer_questions(
        ant,
        STATE,
        {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": ["a", "b"]},
            "refund": {"type": "noul", "instructions": "Asks for money back?"},
        },
    )
    assert set(answers) == {"team", "refund"}
    assert answers["team"]["type"] == "choice"
    assert answers["refund"]["type"] == "noul"


def test_structured_instructions_are_serialized(ant):
    answer = answer_question(
        ant,
        STATE,
        {
            "type": "noul",
            "instructions": {"question": "Same person?", "record": {"name": "Ada"}},
        },
    )
    assert 0.0 <= answer["noul"] <= 1.0


def test_rejects_unknown_type(ant):
    with pytest.raises(TypedQuestionError, match="question type"):
        answer_question(ant, STATE, {"type": "box", "instructions": "Find the logo"})


def test_rejects_bad_criteria(ant):
    with pytest.raises(TypedQuestionError, match="choice needs at least one option"):
        answer_question(ant, STATE, {"type": "choice", "instructions": "Pick", "criteria": {}})
    with pytest.raises(TypedQuestionError, match="score needs 2..10 levels"):
        answer_question(ant, STATE, {"type": "score", "instructions": "How bad?", "criteria": ["one"]})
    with pytest.raises(TypedQuestionError, match="noul criteria"):
        answer_question(
            ant, STATE, {"type": "noul", "instructions": "Yes?", "criteria": {"maybe": "x"}}
        )


def test_rejects_duplicate_candidates(ant):
    with pytest.raises(TypedQuestionError, match="duplicate choice option"):
        answer_question(
            ant, STATE, {"type": "choice", "instructions": "Pick", "criteria": ["a", "a"]}
        )
    with pytest.raises(TypedQuestionError, match="duplicate candidates"):
        answer_question(
            ant,
            STATE,
            {"type": "choice", "instructions": "Pick", "criteria": {"a": "x", "a: x": None}},
        )


def test_error_names_the_question(ant):
    with pytest.raises(TypedQuestionError, match="bad:"):
        answer_questions(
            ant, STATE, {"bad": {"type": "score", "instructions": "How bad?", "criteria": ["one"]}}
        )


def test_rejects_empty_or_oversized_requests(ant):
    with pytest.raises(TypedQuestionError, match="non-empty object"):
        answer_questions(ant, STATE, {})
    huge = {
        f"q{i}": {"type": "noul", "instructions": "Yes?"}
        for i in range(MAX_QUESTIONS + 1)
    }
    with pytest.raises(TypedQuestionError, match="at most"):
        answer_questions(ant, STATE, huge)
