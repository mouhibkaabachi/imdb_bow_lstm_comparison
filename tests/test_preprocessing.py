"""Unit tests for src/preprocessing.py."""

from pathlib import Path

import numpy as np
import pytest

from src.preprocessing import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    build_vocabulary,
    contains_negation,
    encode_text,
    load_vocabulary,
    pad_or_truncate,
    save_vocabulary,
    text_to_bow,
    tokenize,
)


def test_tokenize_lowercases_and_separates_punctuation() -> None:
    assert tokenize("Great movie!") == ["great", "movie", "!"]
    assert tokenize("It was OK, 10/10.") == [
        "it", "was", "ok", ",", "10", "/", "10", ".",
    ]


def test_build_vocabulary_is_deterministic() -> None:
    texts = ["good good movie", "bad movie", "good acting"]
    first = build_vocabulary(texts, min_frequency=1, max_size=100)
    second = build_vocabulary(texts, min_frequency=1, max_size=100)
    assert first == second
    # "good" (3) and "movie" (2) come before ties sorted alphabetically.
    assert first["good"] == 2
    assert first["movie"] == 3
    assert first["acting"] == 4  # alphabetical among frequency-1 tokens
    assert first["bad"] == 5


def test_special_token_ids() -> None:
    vocabulary = build_vocabulary(["some text"], min_frequency=1)
    assert vocabulary[PAD_TOKEN] == PAD_ID == 0
    assert vocabulary[UNK_TOKEN] == UNK_ID == 1


def test_min_frequency_and_max_size_are_respected() -> None:
    texts = ["aa aa bb bb cc"]
    vocabulary = build_vocabulary(texts, min_frequency=2, max_size=3)
    assert "cc" not in vocabulary  # below min_frequency
    assert len(vocabulary) == 3  # <PAD>, <UNK> and one token


def test_encode_text_maps_unknown_tokens_to_unk() -> None:
    vocabulary = build_vocabulary(["good movie good movie"], min_frequency=1)
    ids = encode_text("good unseen movie", vocabulary)
    assert ids == [vocabulary["good"], UNK_ID, vocabulary["movie"]]


def test_pad_or_truncate_right_pads_short_sequences() -> None:
    assert pad_or_truncate([5, 6], max_length=5) == [5, 6, 0, 0, 0]


def test_pad_or_truncate_truncates_long_sequences() -> None:
    assert pad_or_truncate([1, 2, 3, 4, 5], max_length=3) == [1, 2, 3]


def test_text_to_bow_counts_occurrences() -> None:
    vocabulary = build_vocabulary(["good good bad"], min_frequency=1)
    vector = text_to_bow("good good bad unseen", vocabulary)
    assert vector.dtype == np.float32
    assert vector.shape == (len(vocabulary),)
    assert vector[vocabulary["good"]] == 2.0
    assert vector[vocabulary["bad"]] == 1.0
    assert vector[UNK_ID] == 1.0


def test_text_to_bow_binary_stores_presence_only() -> None:
    vocabulary = build_vocabulary(["good good bad"], min_frequency=1)
    vector = text_to_bow("good good good", vocabulary, binary=True)
    assert vector[vocabulary["good"]] == 1.0
    assert vector[vocabulary["bad"]] == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "This is not a good film.",
        "I never enjoyed it.",
        "A film without any charm.",
        "It isn't worth watching.",
        "The plot wasn't believable.",
    ],
)
def test_contains_negation_detects_markers(text: str) -> None:
    assert contains_negation(text) is True


def test_contains_negation_false_for_plain_sentence() -> None:
    assert contains_negation("A wonderful and moving film.") is False


def test_save_and_load_vocabulary_roundtrip(tmp_path: Path) -> None:
    vocabulary = build_vocabulary(["good movie good"], min_frequency=1)
    path = tmp_path / "vocab.json"
    save_vocabulary(vocabulary, path)
    assert load_vocabulary(path) == vocabulary
