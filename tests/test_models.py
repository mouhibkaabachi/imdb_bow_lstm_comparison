"""Unit tests for src/models.py using tiny tensors."""

import pytest
import torch
from torch import nn

from src.models import BoWFeedForwardClassifier, LSTMSentimentClassifier

VOCAB_SIZE = 10
BATCH_SIZE = 4
SEQ_LENGTH = 6


def make_bow_model() -> BoWFeedForwardClassifier:
    torch.manual_seed(42)
    return BoWFeedForwardClassifier(vocab_size=VOCAB_SIZE, hidden_dim=8)


def make_lstm_model() -> LSTMSentimentClassifier:
    torch.manual_seed(42)
    return LSTMSentimentClassifier(
        vocab_size=VOCAB_SIZE, embedding_dim=8, hidden_dim=8
    )


# --- BoWFeedForwardClassifier ----------------------------------------------

def test_bow_model_architecture() -> None:
    model = make_bow_model()
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.relu, nn.ReLU)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.fc2, nn.Linear)
    assert model.fc1.in_features == VOCAB_SIZE
    assert model.fc1.out_features == 8
    assert model.fc2.in_features == 8
    assert model.fc2.out_features == 2


def test_bow_model_forward_output_shape() -> None:
    model = make_bow_model()
    features = torch.rand(BATCH_SIZE, VOCAB_SIZE)
    logits = model(features)
    assert logits.shape == (BATCH_SIZE, 2)


def test_bow_model_logits_require_grad() -> None:
    model = make_bow_model()
    logits = model(torch.rand(BATCH_SIZE, VOCAB_SIZE))
    assert logits.requires_grad


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"vocab_size": -5},
        {"vocab_size": VOCAB_SIZE, "hidden_dim": 0},
        {"vocab_size": VOCAB_SIZE, "output_dim": -1},
    ],
)
def test_bow_model_invalid_dimensions(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="positive"):
        BoWFeedForwardClassifier(**kwargs)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_bow_model_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        BoWFeedForwardClassifier(vocab_size=VOCAB_SIZE, dropout=dropout)


# --- LSTMSentimentClassifier -----------------------------------------------

def test_lstm_model_architecture() -> None:
    model = make_lstm_model()
    assert isinstance(model.embedding, nn.Embedding)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.dropout, nn.Dropout)
    assert isinstance(model.fc, nn.Linear)
    assert model.embedding.num_embeddings == VOCAB_SIZE
    assert model.embedding.padding_idx == 0


def test_lstm_is_one_layer_unidirectional_batch_first() -> None:
    model = make_lstm_model()
    assert model.lstm.num_layers == 1
    assert model.lstm.bidirectional is False
    assert model.lstm.batch_first is True


def test_lstm_forward_output_shape() -> None:
    model = make_lstm_model()
    sequences = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    lengths = torch.full((BATCH_SIZE,), SEQ_LENGTH, dtype=torch.long)
    logits = model(sequences, lengths)
    assert logits.shape == (BATCH_SIZE, 2)


def test_lstm_forward_variable_lengths_with_padding() -> None:
    model = make_lstm_model()
    sequences = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 0, 0, 0],
            [5, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([6, 3, 1], dtype=torch.long)
    logits = model(sequences, lengths)
    assert logits.shape == (3, 2)


def test_lstm_logits_require_grad() -> None:
    model = make_lstm_model()
    sequences = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    lengths = torch.full((BATCH_SIZE,), SEQ_LENGTH, dtype=torch.long)
    logits = model(sequences, lengths)
    assert logits.requires_grad


def test_lstm_accepts_unsorted_lengths() -> None:
    model = make_lstm_model()
    sequences = torch.tensor(
        [
            [2, 0, 0, 0, 0, 0],
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([1, 6, 3], dtype=torch.long)  # not sorted
    logits = model(sequences, lengths)
    assert logits.shape == (3, 2)


def test_lstm_ignores_right_padding_in_eval_mode() -> None:
    model = make_lstm_model()
    model.eval()
    lengths = torch.tensor([3], dtype=torch.long)
    padded_with_zeros = torch.tensor([[2, 3, 4, 0, 0, 0]], dtype=torch.long)
    padded_with_garbage = torch.tensor([[2, 3, 4, 9, 8, 7]], dtype=torch.long)
    with torch.no_grad():
        logits_zeros = model(padded_with_zeros, lengths)
        logits_garbage = model(padded_with_garbage, lengths)
    assert torch.allclose(logits_zeros, logits_garbage)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 0},
        {"vocab_size": VOCAB_SIZE, "embedding_dim": 0},
        {"vocab_size": VOCAB_SIZE, "hidden_dim": -3},
        {"vocab_size": VOCAB_SIZE, "output_dim": 0},
    ],
)
def test_lstm_model_invalid_dimensions(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="positive"):
        LSTMSentimentClassifier(**kwargs)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 2.0])
def test_lstm_model_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        LSTMSentimentClassifier(vocab_size=VOCAB_SIZE, dropout=dropout)
