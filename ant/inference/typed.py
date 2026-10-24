"""Typed decision questions over the ANT candidate scorer: choice, noul, score.

``choice``, ``noul`` and ``score`` are the question types of the decision-model
APIs (TypeSafe ``/v1/systemone``, Laya, vLLM Structured Decisions). Each is the
candidate scoring ANT already does, rendered and named for its type -- nothing
here is trained or fitted:

- ``choice``: one candidate per option; the answer is the picked option's name
  and the probability of every option.
- ``noul``: two candidates, yes and no, optionally described by ``criteria``;
  the answer is the probability that the proposition is true.
- ``score``: one candidate per level of an ordered rubric; the answer is the
  expected level (the probability-weighted average of level indices, so it can
  land between two levels) plus the probability of each level.

Every answer keeps the calibrated ``confidence`` and the ``action``
(execute/escalate) of the underlying prediction, so callers can act on it
directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ant.inference.predictor import ANTPredictor

CHOICE = "choice"
NOUL = "noul"
SCORE = "score"
QUESTION_TYPES = (CHOICE, NOUL, SCORE)

# caps from the published decision-model contracts (vLLM Structured Decisions)
MAX_QUESTIONS = 64
MIN_LEVELS = 2
MAX_LEVELS = 10


class TypedQuestionError(ValueError):
    """A malformed typed question; the message names what is wrong and where."""


def _render(instructions: Any) -> str:
    """Instructions as the question text: strings pass through, structure is JSON."""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, (Mapping, Sequence)) and not isinstance(instructions, (str, bytes)):
        return json.dumps(instructions, ensure_ascii=False)
    raise TypedQuestionError("question instructions must be a string, an object or an array")


def _pairs_choice(criteria: Any) -> list[tuple[str, str]]:
    """[(option name, candidate text)] from a choice's criteria."""
    if isinstance(criteria, Mapping):
        items = list(criteria.items())
    elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
        items = [(name, None) for name in criteria]
    else:
        raise TypedQuestionError(
            "choice criteria must be a map of option -> description, or a list of options"
        )
    if not items:
        raise TypedQuestionError("choice needs at least one option")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, description in items:
        if not isinstance(name, str) or not name.strip():
            raise TypedQuestionError("choice option names must be non-empty strings")
        if name in seen:
            raise TypedQuestionError(f"duplicate choice option {name!r}")
        seen.add(name)
        if description is not None and not isinstance(description, str):
            raise TypedQuestionError(f"choice description for {name!r} must be a string or null")
        pairs.append((name, f"{name}: {description}" if description else name))
    return pairs


def _pairs_noul(criteria: Any) -> list[tuple[str, str]]:
    """[(answer key, candidate text)] for the yes/no pair."""
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, Mapping):
        raise TypedQuestionError("noul criteria must be a map with optional 'true'/'false' descriptions")
    unknown = set(criteria) - {"true", "false"}
    if unknown:
        raise TypedQuestionError(f"noul criteria only accepts 'true' and 'false', got {sorted(unknown)}")
    yes, no = criteria.get("true"), criteria.get("false")
    for label, text in (("true", yes), ("false", no)):
        if text is not None and not isinstance(text, str):
            raise TypedQuestionError(f"noul criteria {label!r} must be a string")
    return [
        ("yes", f"Yes -- {yes}" if yes else "Yes"),
        ("no", f"No -- {no}" if no else "No"),
    ]


def _pairs_score(criteria: Any) -> list[tuple[str, str]]:
    """[(level index, candidate text)] from the ordered rubric."""
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
        raise TypedQuestionError("score criteria must be a list of level descriptions")
    if not (MIN_LEVELS <= len(criteria) <= MAX_LEVELS):
        raise TypedQuestionError(f"score needs {MIN_LEVELS}..{MAX_LEVELS} levels, got {len(criteria)}")
    pairs: list[tuple[str, str]] = []
    for index, level in enumerate(criteria):
        if not isinstance(level, str) or not level.strip():
            raise TypedQuestionError(f"score level {index} must be a non-empty string")
        pairs.append((str(index), level))
    if len({text for _, text in pairs}) != len(pairs):
        raise TypedQuestionError("score levels must be distinct descriptions")
    return pairs


def answer_question(
    predictor: ANTPredictor, state: str, question: Mapping[str, Any]
) -> dict[str, Any]:
    """Answer one typed question about ``state``; raises TypedQuestionError if malformed."""
    if not isinstance(question, Mapping):
        raise TypedQuestionError("each question must be an object with type and instructions")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise TypedQuestionError(f"question type must be one of {', '.join(QUESTION_TYPES)}, got {kind!r}")
    instructions = _render(question.get("instructions"))
    if not instructions.strip():
        raise TypedQuestionError("question instructions must not be empty")
    criteria = question.get("criteria")
    if kind == CHOICE:
        pairs = _pairs_choice(criteria)
    elif kind == NOUL:
        pairs = _pairs_noul(criteria)
    else:
        pairs = _pairs_score(criteria)
    texts = [text for _, text in pairs]
    if len(set(texts)) != len(texts):
        raise TypedQuestionError("the question's options/levels render to duplicate candidates")

    result = predictor.predict(state, instructions, texts)
    probabilities = [float(result.probabilities[text]) for text in texts]

    if kind == NOUL:
        p_yes, p_no = probabilities
        return {
            "type": NOUL,
            "noul": p_yes,
            "probabilities": {"yes": p_yes, "no": p_no},
            "confidence": float(result.confidence),
            "action": result.action,
        }
    if kind == SCORE:
        by_level = {key: p for (key, _), p in zip(pairs, probabilities)}
        return {
            "type": SCORE,
            "score": sum(int(key) * p for key, p in by_level.items()),
            "legend": dict(pairs),
            "probabilities": by_level,
            "confidence": float(result.confidence),
            "action": result.action,
        }
    by_option = {key: p for (key, _), p in zip(pairs, probabilities)}
    key_by_text = {text: key for key, text in pairs}
    return {
        "type": CHOICE,
        "choice": key_by_text[result.choice],
        "probabilities": by_option,
        "confidence": float(result.confidence),
        "action": result.action,
    }


def answer_questions(
    predictor: ANTPredictor, state: str, questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Answer several typed questions about one state, keyed by question id."""
    if not isinstance(questions, Mapping) or not questions:
        raise TypedQuestionError("questions must be a non-empty object of id -> question")
    if len(questions) > MAX_QUESTIONS:
        raise TypedQuestionError(f"at most {MAX_QUESTIONS} questions per request, got {len(questions)}")
    answers: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise TypedQuestionError("question ids must be non-empty strings")
        try:
            answers[question_id] = answer_question(predictor, state, question)
        except TypedQuestionError as error:
            raise TypedQuestionError(f"{question_id}: {error}") from error
    return answers
