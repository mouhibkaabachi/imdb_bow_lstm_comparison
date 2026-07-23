"""PyTorch Dataset classes for the Bag-of-Words and LSTM models.

Both datasets convert raw texts on the fly in __getitem__ instead of
precomputing every vector, which keeps memory usage low for large
vocabularies and long review collections.
"""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from src.preprocessing import PAD_TOKEN, encode_text, pad_or_truncate, text_to_bow


def _validate_texts_and_labels(texts: list[str], labels: list[int]) -> None:
    """Raise ValueError if texts and labels are empty or of unequal length."""
    if len(texts) == 0:
        raise ValueError("texts must not be empty.")
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have the same length, "
            f"got {len(texts)} texts and {len(labels)} labels."
        )


class BoWDataset(Dataset):
    """Dataset yielding bag-of-words feature vectors and sentiment labels."""

    def __init__(
            self,
            texts: Sequence[str],
            labels: Sequence[int],
            vocabulary: dict[str, int],
            binary: bool = False,
    ) -> None:
        """Store texts and labels; BoW vectors are computed lazily.

        Args:
            texts: raw review texts.
            labels: integer sentiment labels (0 = negative, 1 = positive).
            vocabulary: token-to-ID mapping built on the training texts.
            binary: if True, use presence/absence instead of counts.
        """
        self.texts: list[str] = list(texts)
        self.labels: list[int] = list(labels)
        _validate_texts_and_labels(self.texts, self.labels)
        self.vocabulary = vocabulary
        self.binary = binary

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, label) with shapes (vocabulary_size,) and ()."""
        features = torch.from_numpy(
            text_to_bow(self.texts[index], self.vocabulary, binary=self.binary)
        ).to(torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return features, label


class SequenceDataset(Dataset):
    """Dataset yielding padded token ID sequences, lengths and labels.

    The effective length of each sequence is returned so that the LSTM
    model can use packed padded sequences and ignore right-padding.
    """

    def __init__(
            self,
            texts: Sequence[str],
            labels: Sequence[int],
            vocabulary: dict[str, int],
            max_length: int,
    ) -> None:
        """Store texts and labels; encoding and padding happen lazily.

        Args:
            texts: raw review texts.
            labels: integer sentiment labels (0 = negative, 1 = positive).
            vocabulary: token-to-ID mapping built on the training texts.
            max_length: fixed sequence length after padding/truncation.
        """
        self.texts: list[str] = list(texts)
        self.labels: list[int] = list(labels)
        _validate_texts_and_labels(self.texts, self.labels)
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.pad_id = vocabulary[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(
            self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sequence, length, label).

        The sequence has shape (max_length,) and dtype torch.long. The
        length is min(number of encoded tokens, max_length), i.e. the
        number of real (non-padding) positions in the sequence.
        """
        encoded = encode_text(self.texts[index], self.vocabulary)
        padded = pad_or_truncate(encoded, self.max_length, pad_id=self.pad_id)
        length = min(len(encoded), self.max_length)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )
