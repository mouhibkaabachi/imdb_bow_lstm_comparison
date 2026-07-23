"""Reusable training utilities for the BoW and LSTM sentiment models.

Provides seeding, device handling, explicit train/evaluation epoch loops,
early stopping on validation loss, and checkpoint save/load helpers. The
functions support both project batch formats:

- BoW batch: (features, labels)
- Sequence batch: (sequences, lengths, labels)
"""

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments.

    Also seeds all CUDA devices and configures deterministic CuDNN
    behaviour when CUDA is available; on CPU-only machines this is a no-op.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch_to_device(
        batch: tuple[torch.Tensor, ...], device: torch.device | str
) -> tuple[torch.Tensor, ...]:
    """Move every tensor of a 2-item (BoW) or 3-item (sequence) batch.

    Returns a tuple with the same length and order as the input batch.
    """
    if len(batch) not in (2, 3):
        raise ValueError(
            f"Unsupported batch length {len(batch)}; expected 2 "
            f"(features, labels) or 3 (sequences, lengths, labels)."
        )
    return tuple(tensor.to(device) for tensor in batch)


def compute_batch_logits(
        model: nn.Module, batch: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model forward pass appropriate for the batch format.

    Returns:
        (logits, labels), where logits are raw scores (no sigmoid/softmax).
    """
    if len(batch) == 2:
        features, labels = batch
        return model(features), labels
    if len(batch) == 3:
        sequences, lengths, labels = batch
        return model(sequences, lengths), labels
    raise ValueError(
        f"Unsupported batch length {len(batch)}; expected 2 "
        f"(features, labels) or 3 (sequences, lengths, labels)."
    )


def train_one_epoch(
        model: nn.Module,
        data_loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
) -> dict[str, float | int]:
    """Train the model for one epoch and return aggregate metrics.

    Returns:
        {"loss": average_loss, "accuracy": accuracy, "examples": total}.
        Loss is example-weighted across batches.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits, labels = compute_batch_logits(model, batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("data_loader produced zero examples.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }


def evaluate_one_epoch(
        model: nn.Module,
        data_loader: Any,
        criterion: nn.Module,
        device: torch.device | str,
) -> dict[str, float | int]:
    """Evaluate the model for one epoch without updating parameters.

    Returns the same {"loss", "accuracy", "examples"} dictionary as
    train_one_epoch, with example-weighted average loss.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits, labels = compute_batch_logits(model, batch)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("data_loader produced zero examples.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }


class EarlyStopping:
    """Early stopping on validation loss (lower is better).

    An epoch counts as an improvement only when
    validation_loss < best_loss - min_delta. After `patience` consecutive
    non-improving epochs, `should_stop` becomes True.
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.0) -> None:
        if not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience!r}.")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}.")
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self) -> None:
        """Restore the initial state so the instance can be reused."""
        self.best_loss = math.inf
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """Record one epoch's validation loss.

        Returns:
            True only if the supplied loss is a genuine improvement.
        """
        if not math.isfinite(validation_loss):
            raise ValueError(
                f"validation_loss must be finite, got {validation_loss}."
            )
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer state dicts plus metadata to `path`.

    Creates the parent directory if needed. Only state dicts are stored,
    never the whole model object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "extra": {} if extra is None else extra,
        },
        path,
    )


def load_checkpoint(
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into an existing model (and optionally optimizer).

    Returns:
        {"epoch": ..., "metrics": ..., "extra": ...} from the checkpoint.

    Raises:
        FileNotFoundError: if `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "extra": checkpoint["extra"],
    }


def fit_model(
        model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        max_epochs: int,
        checkpoint_path: Path | str,
        patience: int = 2,
        min_delta: float = 0.0,
) -> list[dict[str, float]]:
    """Train with early stopping on validation loss and best-model restore.

    Each epoch trains on `train_loader`, evaluates on `validation_loader`,
    and appends one history entry. A checkpoint is written only when the
    validation loss genuinely improves (epoch numbering starts at 1).
    After training, the best checkpoint's weights are loaded back into
    the model.

    Returns:
        History as a list of dictionaries with keys: epoch, train_loss,
        train_accuracy, validation_loss, validation_accuracy.
    """
    if max_epochs <= 0:
        raise ValueError(f"max_epochs must be positive, got {max_epochs}.")
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_metrics = evaluate_one_epoch(
            model, validation_loader, criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        if early_stopping.step(validation_metrics["loss"]):
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, history[-1]
            )
        if early_stopping.should_stop:
            break
    load_checkpoint(checkpoint_path, model, device=device)
    return history
