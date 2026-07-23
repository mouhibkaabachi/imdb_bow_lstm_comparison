"""Unit tests for src/training.py using tiny synthetic data and models."""

import inspect
import math
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    EarlyStopping,
    compute_batch_logits,
    evaluate_one_epoch,
    fit_model,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    set_random_seeds,
    train_one_epoch,
)

NUM_FEATURES = 4
NUM_EXAMPLES = 8
SEQ_LENGTH = 5
VOCAB_SIZE = 6
DEVICE = "cpu"


class TinyBoWModel(nn.Module):
    """Minimal model with the BoW interface: forward(features)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class TinySequenceModel(nn.Module):
    """Minimal model with the sequence interface: forward(sequences, lengths)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 3, padding_idx=0)
        self.linear = nn.Linear(3, 2)

    def forward(
            self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        summed = self.embedding(sequences).sum(dim=1)
        return self.linear(summed / lengths.unsqueeze(1).float())


def make_bow_loader(batch_size: int = 4) -> DataLoader:
    torch.manual_seed(42)
    features = torch.randn(NUM_EXAMPLES, NUM_FEATURES)
    labels = (features[:, 0] > 0).long()
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size)


def make_sequence_loader(batch_size: int = 4) -> DataLoader:
    torch.manual_seed(42)
    sequences = torch.randint(1, VOCAB_SIZE, (NUM_EXAMPLES, SEQ_LENGTH))
    lengths = torch.full((NUM_EXAMPLES,), SEQ_LENGTH, dtype=torch.long)
    labels = (sequences[:, 0] > VOCAB_SIZE // 2).long()
    return DataLoader(
        TensorDataset(sequences, lengths, labels), batch_size=batch_size
    )


def make_bow_setup() -> tuple[TinyBoWModel, DataLoader, nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(42)
    model = TinyBoWModel()
    loader = make_bow_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, loader, criterion, optimizer


# --- set_random_seeds --------------------------------------------------------

def test_set_random_seeds_is_reproducible() -> None:
    set_random_seeds(123)
    first = (random.random(), np.random.rand(2).tolist(), torch.rand(3))
    set_random_seeds(123)
    second = (random.random(), np.random.rand(2).tolist(), torch.rand(3))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


# --- move_batch_to_device ----------------------------------------------------

def test_move_batch_to_device_two_items() -> None:
    features = torch.rand(2, NUM_FEATURES)
    labels = torch.tensor([0, 1])
    moved = move_batch_to_device((features, labels), DEVICE)
    assert isinstance(moved, tuple)
    assert len(moved) == 2
    assert torch.equal(moved[0], features)
    assert torch.equal(moved[1], labels)
    assert all(t.device.type == "cpu" for t in moved)


def test_move_batch_to_device_three_items() -> None:
    sequences = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
    lengths = torch.tensor([5, 3])
    labels = torch.tensor([1, 0])
    moved = move_batch_to_device((sequences, lengths, labels), DEVICE)
    assert len(moved) == 3
    assert torch.equal(moved[0], sequences)
    assert torch.equal(moved[1], lengths)
    assert torch.equal(moved[2], labels)


@pytest.mark.parametrize("length", [1, 4])
def test_move_batch_to_device_rejects_unsupported_length(length: int) -> None:
    batch = tuple(torch.zeros(2) for _ in range(length))
    with pytest.raises(ValueError):
        move_batch_to_device(batch, DEVICE)


# --- compute_batch_logits ----------------------------------------------------

def test_compute_batch_logits_bow_dispatch() -> None:
    torch.manual_seed(42)
    model = TinyBoWModel()
    features = torch.rand(3, NUM_FEATURES)
    labels = torch.tensor([0, 1, 0])
    logits, returned_labels = compute_batch_logits(model, (features, labels))
    assert logits.shape == (3, 2)
    assert torch.equal(returned_labels, labels)


def test_compute_batch_logits_sequence_dispatch() -> None:
    torch.manual_seed(42)
    model = TinySequenceModel()
    sequences = torch.randint(1, VOCAB_SIZE, (3, SEQ_LENGTH))
    lengths = torch.full((3,), SEQ_LENGTH, dtype=torch.long)
    labels = torch.tensor([1, 0, 1])
    logits, returned_labels = compute_batch_logits(
        model, (sequences, lengths, labels)
    )
    assert logits.shape == (3, 2)
    assert torch.equal(returned_labels, labels)


def test_compute_batch_logits_rejects_unsupported_length() -> None:
    model = TinyBoWModel()
    with pytest.raises(ValueError):
        compute_batch_logits(model, (torch.zeros(2),))


# --- train_one_epoch ---------------------------------------------------------

def test_train_one_epoch_returns_expected_metrics() -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    metrics = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    assert set(metrics) == {"loss", "accuracy", "examples"}
    assert metrics["examples"] == NUM_EXAMPLES
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_train_one_epoch_updates_parameters() -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    before = [param.clone() for param in model.parameters()]
    train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    after = list(model.parameters())
    assert any(
        not torch.equal(b, a.detach()) for b, a in zip(before, after)
    )


def test_train_one_epoch_supports_sequence_batches() -> None:
    torch.manual_seed(42)
    model = TinySequenceModel()
    loader = make_sequence_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, DEVICE
    )
    assert metrics["examples"] == NUM_EXAMPLES


def test_train_one_epoch_rejects_empty_loader() -> None:
    model, _, criterion, optimizer = make_bow_setup()
    with pytest.raises(ValueError):
        train_one_epoch(model, [], criterion, optimizer, DEVICE)


# --- evaluate_one_epoch ------------------------------------------------------

def test_evaluate_one_epoch_does_not_change_parameters() -> None:
    model, loader, criterion, _ = make_bow_setup()
    before = [param.clone() for param in model.parameters()]
    evaluate_one_epoch(model, loader, criterion, DEVICE)
    assert all(
        torch.equal(b, a.detach())
        for b, a in zip(before, model.parameters())
    )


def test_evaluate_one_epoch_returns_expected_metrics() -> None:
    model, loader, criterion, _ = make_bow_setup()
    metrics = evaluate_one_epoch(model, loader, criterion, DEVICE)
    assert set(metrics) == {"loss", "accuracy", "examples"}
    assert metrics["examples"] == NUM_EXAMPLES
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_one_epoch_rejects_empty_loader() -> None:
    model, _, criterion, _ = make_bow_setup()
    with pytest.raises(ValueError):
        evaluate_one_epoch(model, [], criterion, DEVICE)


# --- EarlyStopping -----------------------------------------------------------

def test_early_stopping_detects_improvement() -> None:
    stopper = EarlyStopping(patience=2)
    assert stopper.step(1.0) is True
    assert stopper.best_loss == 1.0
    assert stopper.step(0.5) is True
    assert stopper.best_loss == 0.5
    assert stopper.should_stop is False


def test_early_stopping_respects_min_delta() -> None:
    stopper = EarlyStopping(patience=3, min_delta=0.1)
    assert stopper.step(1.0) is True
    # 0.95 is lower but not by more than min_delta, so not an improvement.
    assert stopper.step(0.95) is False
    assert stopper.best_loss == 1.0
    assert stopper.step(0.85) is True
    assert stopper.best_loss == 0.85


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2)
    stopper.step(1.0)
    assert stopper.step(1.2) is False
    assert stopper.should_stop is False
    assert stopper.step(1.1) is False
    assert stopper.should_stop is True
    assert stopper.epochs_without_improvement == 2


def test_early_stopping_reset() -> None:
    stopper = EarlyStopping(patience=1)
    stopper.step(1.0)
    stopper.step(2.0)
    assert stopper.should_stop is True
    stopper.reset()
    assert stopper.best_loss == math.inf
    assert stopper.epochs_without_improvement == 0
    assert stopper.should_stop is False


@pytest.mark.parametrize("patience", [0, -1, 1.5])
def test_early_stopping_rejects_invalid_patience(patience) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience)


def test_early_stopping_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=-0.01)


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf])
def test_early_stopping_rejects_non_finite_loss(loss: float) -> None:
    stopper = EarlyStopping()
    with pytest.raises(ValueError):
        stopper.step(loss)


# --- save_checkpoint / load_checkpoint ---------------------------------------

def test_checkpoint_roundtrip_restores_model_and_metadata(tmp_path) -> None:
    model, _, _, optimizer = make_bow_setup()
    path = tmp_path / "checkpoints" / "best.pt"
    metrics = {"loss": 0.4, "accuracy": 0.9}
    save_checkpoint(path, model, optimizer, epoch=3, metrics=metrics,
                    extra={"note": "test"})

    torch.manual_seed(7)
    fresh_model = TinyBoWModel()
    fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
    metadata = load_checkpoint(path, fresh_model, fresh_optimizer, DEVICE)

    assert metadata["epoch"] == 3
    assert metadata["metrics"] == metrics
    assert metadata["extra"] == {"note": "test"}
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh_model.state_dict()[key])


def test_load_checkpoint_without_optimizer(tmp_path) -> None:
    model, _, _, optimizer = make_bow_setup()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, optimizer, epoch=1, metrics={"loss": 0.5})

    torch.manual_seed(7)
    fresh_model = TinyBoWModel()
    metadata = load_checkpoint(path, fresh_model)
    assert metadata["extra"] == {}
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh_model.state_dict()[key])


def test_load_checkpoint_missing_file_raises(tmp_path) -> None:
    model, _, _, _ = make_bow_setup()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)


# --- fit_model ----------------------------------------------------------------

def test_fit_model_history_has_expected_keys(tmp_path) -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    history = fit_model(
        model, loader, loader, criterion, optimizer, DEVICE,
        max_epochs=2, checkpoint_path=tmp_path / "best.pt",
    )
    assert 1 <= len(history) <= 2
    expected_keys = {
        "epoch", "train_loss", "train_accuracy",
        "validation_loss", "validation_accuracy",
    }
    for entry_index, entry in enumerate(history, start=1):
        assert set(entry) == expected_keys
        assert entry["epoch"] == entry_index


def test_fit_model_saves_best_checkpoint(tmp_path) -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    path = tmp_path / "best.pt"
    fit_model(
        model, loader, loader, criterion, optimizer, DEVICE,
        max_epochs=2, checkpoint_path=path,
    )
    assert path.exists()


def test_fit_model_restores_best_checkpoint_weights(tmp_path) -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    path = tmp_path / "best.pt"
    fit_model(
        model, loader, loader, criterion, optimizer, DEVICE,
        max_epochs=3, checkpoint_path=path,
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(value, model.state_dict()[key])


def test_fit_model_has_no_test_data_parameter() -> None:
    parameters = inspect.signature(fit_model).parameters
    assert not any("test" in name for name in parameters)


@pytest.mark.parametrize("max_epochs", [0, -2])
def test_fit_model_rejects_invalid_max_epochs(tmp_path, max_epochs: int) -> None:
    model, loader, criterion, optimizer = make_bow_setup()
    with pytest.raises(ValueError):
        fit_model(
            model, loader, loader, criterion, optimizer, DEVICE,
            max_epochs=max_epochs, checkpoint_path=tmp_path / "best.pt",
        )
