"""Model definitions: bag-of-words feedforward baseline and LSTM classifier.

Both models return raw class logits (no softmax) because training uses
torch.nn.CrossEntropyLoss, which applies log-softmax internally.
"""

import torch
from torch import nn


def _validate_positive(**dimensions: int) -> None:
    """Raise ValueError if any named dimension is not positive."""
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")


def _validate_dropout(dropout: float) -> None:
    """Raise ValueError if dropout is outside [0, 1)."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


class BoWFeedForwardClassifier(nn.Module):
    """Feedforward classifier over bag-of-words vectors (the baseline)."""

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int = 64,
            output_dim: int = 2,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()
        _validate_positive(
            vocab_size=vocab_size, hidden_dim=hidden_dim, output_dim=output_dim
        )
        _validate_dropout(dropout)

        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (batch_size, vocab_size) features to (batch_size, output_dim) logits."""
        hidden = self.dropout(self.relu(self.fc1(features)))
        return self.fc2(hidden)


class LSTMSentimentClassifier(nn.Module):
    """One-layer unidirectional LSTM classifier over token ID sequences."""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            output_dim: int = 2,
            padding_idx: int = 0,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()
        _validate_positive(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        _validate_dropout(dropout)

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
            self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Classify padded token sequences using their effective lengths.

        Args:
            sequences: padded token IDs of shape (batch_size, sequence_length).
            lengths: effective lengths of shape (batch_size,), i.e. the
                number of real tokens before right-padding.

        Returns:
            Raw class logits of shape (batch_size, output_dim).
        """
        embedded = self.embedding(sequences)
        # pack_padded_sequence requires lengths on the CPU; converting here
        # keeps the model compatible with GPU training.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n[-1] is the final hidden state of the (single) LSTM layer.
        return self.fc(self.dropout(h_n[-1]))
