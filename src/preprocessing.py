"""Text preprocessing utilities: tokenization, vocabulary, encoding, BoW.

All functions are deterministic so that experiments are reproducible.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from config import (
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    MIN_TOKEN_FREQUENCY,
    NEGATION_MARKERS,
)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

# Words, numbers, or single punctuation characters.
_TOKEN_PATTERN = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into words, numbers and punctuation.

    Example: "Don't stop!" -> ["don", "'", "t", "stop", "!"]
    """
    return _TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(
        texts: Iterable[str],
        min_frequency: int = MIN_TOKEN_FREQUENCY,
        max_size: int = MAX_VOCAB_SIZE,
) -> dict[str, int]:
    """Build a token-to-ID vocabulary from a collection of texts.

    IMPORTANT: to avoid data leakage, call this function with the
    TRAINING texts only. Validation and test texts must never
    influence the vocabulary.

    "<PAD>" is reserved as ID 0 and "<UNK>" as ID 1. The remaining
    tokens keep only those with frequency >= min_frequency, sorted by
    descending frequency and alphabetically for ties, capped at
    max_size total entries (special tokens included).
    """
    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(tokenize(text))

    eligible = [
        (token, count)
        for token, count in frequencies.items()
        if count >= min_frequency
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))

    vocabulary: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _ in eligible[: max_size - len(vocabulary)]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    """Convert a text into a list of token IDs, using <UNK> for unseen tokens."""
    unk_id = vocabulary[UNK_TOKEN]
    return [vocabulary.get(token, unk_id) for token in tokenize(text)]


def pad_or_truncate(
        sequence: list[int],
        max_length: int = MAX_SEQUENCE_LENGTH,
        pad_id: int = PAD_ID,
) -> list[int]:
    """Truncate a sequence to max_length or right-pad it with pad_id."""
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + [pad_id] * (max_length - len(sequence))


def text_to_bow(
        text: str, vocabulary: dict[str, int], binary: bool = False
) -> np.ndarray:
    """Convert a text into a float32 bag-of-words vector.

    The vector has one dimension per vocabulary entry. If binary is
    False, each dimension holds the token count; if True, it holds 1.0
    for presence and 0.0 for absence. Tokens outside the vocabulary are
    counted under <UNK>.
    """
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    for token_id in encode_text(text, vocabulary):
        if binary:
            vector[token_id] = 1.0
        else:
            vector[token_id] += 1.0
    return vector


def contains_negation(text: str, markers: Iterable[str] | None = None) -> bool:
    """Return True if the text contains an English negation marker.

    Contractions such as "isn't" or "wasn't" are recognized via the
    "n't" marker, even though the tokenizer splits them into separate
    word and punctuation tokens (e.g. ["isn", "'", "t"]).
    """
    marker_set = set(NEGATION_MARKERS if markers is None else markers)
    tokens = tokenize(text)

    for index, token in enumerate(tokens):
        if token in marker_set:
            return True
        # Detect "n't" contractions split as [...n, ', t].
        if (
                "n't" in marker_set
                and token == "'"
                and index > 0
                and index + 1 < len(tokens)
                and tokens[index - 1].endswith("n")
                and tokens[index + 1] == "t"
        ):
            return True
    return False


def save_vocabulary(vocabulary: dict[str, int], path: str | Path) -> None:
    """Save a vocabulary to a JSON file with a stable key order."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=2, sort_keys=True)


def load_vocabulary(path: str | Path) -> dict[str, int]:
    """Load a vocabulary from a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return {token: int(token_id) for token, token_id in json.load(file).items()}
