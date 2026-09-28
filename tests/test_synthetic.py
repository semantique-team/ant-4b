from __future__ import annotations

from ant.data.synthetic import (
    DEPARTMENTS,
    DIFFICULTIES,
    HARD_NEGATIVES,
    LANGUAGES,
    OOD_DEPARTMENTS,
    SyntheticConfig,
    generate_dataset,
    generate_examples,
)


def test_deterministic_generation():
    config = SyntheticConfig(seed=3, n_train=20)
    first = generate_examples(20, split="train", config=config)
    second = generate_examples(20, split="train", config=config)
    assert first == second
    other_seed = generate_examples(20, split="train", config=SyntheticConfig(seed=4, n_train=20))
    assert first != other_seed


def test_every_example_is_valid():
    for split in ("train", "val", "test"):
        for example in generate_examples(50, split=split, config=SyntheticConfig(seed=1)):
            assert example.correct in example.candidates
            assert example.metadata["difficulty"] in DIFFICULTIES
            assert example.metadata["language"] in LANGUAGES
            assert len(set(example.candidates)) == len(example.candidates)
            assert 4 <= len(example.candidates) <= 6


def test_hard_examples_include_hard_negatives():
    config = SyntheticConfig(seed=5, n_train=300)
    hard = [
        e
        for e in generate_examples(300, split="train", config=config)
        if e.metadata["difficulty"] == "hard"
    ]
    assert hard
    with_negative = [
        e
        for e in hard
        if any(n in e.candidates for n in HARD_NEGATIVES[e.metadata["department"]])
    ]
    assert len(with_negative) >= 0.9 * len(hard)


def test_multilingual_coverage_in_test():
    examples = generate_examples(200, split="test", config=SyntheticConfig(seed=2, n_test=200))
    languages = {e.metadata["language"] for e in examples}
    assert languages == set(LANGUAGES)


def test_ood_split_uses_unseen_departments():
    splits = generate_dataset(n_train=30, n_val=10, n_test=10, n_ood=30)
    train_departments = {e.metadata["department"] for e in splits.train}
    ood_departments = {e.metadata["department"] for e in splits.ood}
    assert ood_departments <= set(OOD_DEPARTMENTS)
    assert train_departments <= set(DEPARTMENTS)
    assert not (train_departments & ood_departments)


def test_splits_are_disjoint_objects():
    splits = generate_dataset(n_train=25, n_val=10, n_test=10, n_ood=5)
    assert len(splits.train) == 25 and len(splits.val) == 10 and len(splits.test) == 10
    assert len(splits.ood) == 5
