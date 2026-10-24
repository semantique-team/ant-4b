"""The multi-use-case dataset: valid rows, leak-free splits, JSONL round trip."""

from __future__ import annotations

import pytest

from ant.data.jsonl import load_jsonl, load_splits, write_jsonl
from ant.data.multitask import build_multitask
from ant.data.usecases import USE_CASES


@pytest.fixture(scope="module")
def rows():
    return build_multitask(variants={"train": 3, "val": 2, "test": 2, "ood": 2})


def test_use_cases_cover_english_and_italian():
    languages = {use_case.language for use_case in USE_CASES}
    assert {"en", "it"} <= languages
    assert len(USE_CASES) >= 20


def test_variant_counts_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        build_multitask(variants={"train": 0})


def test_no_state_leaks_across_splits(rows):
    """A seed situation lives in exactly one split (wrappers are stripped for the check)."""
    seeds = {
        state: label_split
        for use_case in USE_CASES
        for label in use_case.labels.values()
        for state, label_split in zip(label.states, [None] * len(label.states))
    }
    home: dict[str, set[str]] = {state: set() for state in seeds}
    for row in rows:
        for state in seeds:
            if state in row.example.state:
                home[state].add(row.split)
    assert all(len(splits) <= 1 for splits in home.values())


def test_held_out_use_cases_only_in_ood(rows):
    held_out = {uc.name for uc in USE_CASES if uc.held_out}
    for row in rows:
        assert (row.use_case in held_out) == (row.split == "ood")


def test_every_split_covers_several_use_cases_and_candidate_counts(rows):
    for split in ("train", "val", "test"):
        tagged = [r for r in rows if r.split == split]
        assert len({r.use_case for r in tagged}) >= 10
        assert len({r.example.n_candidates for r in tagged}) >= 3


def test_jsonl_round_trip_keeps_use_case(tmp_path, rows):
    examples = [r.example for r in rows[:20]]
    write_jsonl(tmp_path / "train.jsonl", examples)
    write_jsonl(tmp_path / "val.jsonl", examples)
    assert load_jsonl(tmp_path / "train.jsonl") == examples
    assert set(load_splits(tmp_path)) == {"train", "val"}


def test_bad_line_names_file_and_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"state": "s", "question": "q", "candidates": ["a"], "correct": "z"}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        load_jsonl(path)


def test_correct_must_be_a_candidate(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"state": "s", "question": "q", "candidates": ["a", "b"], "correct": "z"}\n'
    )
    with pytest.raises(ValueError, match="correct must be one of candidates"):
        load_jsonl(path)
