"""Evaluation and error-analysis utilities for the BoW and LSTM models.

Provides prediction collection, classification metrics, inference timing,
parameter counting, per-example prediction records, negation-subset
analysis, deterministic error-example selection, and JSON/CSV persistence.
Supports both project batch formats:

- BoW batch: (features, labels)
- Sequence batch: (sequences, lengths, labels)
"""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

from src.preprocessing import contains_negation
from src.training import compute_batch_logits, move_batch_to_device

POSITIVE_LABEL = 1

RECORD_COLUMNS = [
    "text",
    "true_label",
    "predicted_label",
    "negative_probability",
    "positive_probability",
    "contains_negation",
    "is_correct",
]

_EMPTY_SUBSET_METRICS: dict[str, Any] = {
    "examples": 0,
    "accuracy": None,
    "precision": None,
    "recall": None,
    "f1": None,
    "confusion_matrix": None,
}


def collect_predictions(
        model: nn.Module,
        data_loader: Any,
        device: torch.device | str,
) -> dict[str, np.ndarray]:
    """Run inference over a DataLoader and gather labels, predictions, scores.

    Supports both BoW batches (features, labels) and sequence batches
    (sequences, lengths, labels). Probabilities are computed with softmax
    for evaluation output only; the models themselves return raw logits.

    Returns:
        {"labels": int64 (n,), "predictions": int64 (n,),
         "probabilities": float32 (n, num_classes),
         "logits": float32 (n, num_classes)}, in DataLoader order.

    Raises:
        ValueError: if the DataLoader produces no examples.
    """
    model.eval()
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits, labels = compute_batch_logits(model, batch)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    if not all_labels:
        raise ValueError("data_loader produced zero examples.")
    return {
        "labels": np.concatenate(all_labels).astype(np.int64),
        "predictions": np.concatenate(all_predictions).astype(np.int64),
        "probabilities": np.concatenate(all_probabilities).astype(np.float32),
        "logits": np.concatenate(all_logits).astype(np.float32),
    }


def calculate_classification_metrics(
        labels: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1 and the confusion matrix.

    Precision, recall and F1 treat class 1 ("positive") as the positive
    class and use zero_division=0. Values are not rounded.

    Returns:
        JSON-serializable dictionary with keys "accuracy", "precision",
        "recall", "f1" and "confusion_matrix" (nested Python list).

    Raises:
        ValueError: if inputs are not one-dimensional, are empty, or
            differ in length.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError(
            f"labels and predictions must be one-dimensional, got shapes "
            f"{labels.shape} and {predictions.shape}."
        )
    if labels.size == 0:
        raise ValueError("labels and predictions must not be empty.")
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"labels and predictions must have the same length, got "
            f"{labels.shape[0]} and {predictions.shape[0]}."
        )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(
                labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }


def measure_inference_time(
        model: nn.Module,
        data_loader: Any,
        device: torch.device | str,
        warmup_batches: int = 1,
) -> dict[str, float | int]:
    """Time a full inference pass over the DataLoader, excluding warmup.

    Up to `warmup_batches` initial batches run outside the measured
    section. When the device is CUDA, the GPU is synchronized right
    before and after the timed section so wall-clock times are accurate.

    Returns:
        {"total_seconds": float, "examples": int,
         "seconds_per_example": float}, counting timed examples only.

    Raises:
        ValueError: if warmup_batches is invalid or no timed batches remain.
    """
    if (
            not isinstance(warmup_batches, int)
            or isinstance(warmup_batches, bool)
            or warmup_batches < 0
    ):
        raise ValueError(
            f"warmup_batches must be a non-negative integer, got "
            f"{warmup_batches!r}."
        )
    device = torch.device(device)
    model.eval()
    timed_examples = 0
    with torch.no_grad():
        iterator = iter(data_loader)
        for _ in range(warmup_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch = move_batch_to_device(batch, device)
            compute_batch_logits(model, batch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for batch in iterator:
            batch = move_batch_to_device(batch, device)
            logits, _ = compute_batch_logits(model, batch)
            timed_examples += logits.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_seconds = time.perf_counter() - start
    if timed_examples == 0:
        raise ValueError(
            "No timed examples remain after warmup; use fewer warmup "
            "batches or a larger DataLoader."
        )
    return {
        "total_seconds": total_seconds,
        "examples": timed_examples,
        "seconds_per_example": total_seconds / timed_examples,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the total number of trainable (requires_grad) parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def build_prediction_records(
        texts: list[str],
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        negation_flags: list[bool] | None = None,
) -> pd.DataFrame:
    """Combine texts, labels and model outputs into a per-example DataFrame.

    When negation_flags is None, negation is detected with
    src.preprocessing.contains_negation. Input order is preserved.

    Returns:
        DataFrame with columns: text, true_label, predicted_label,
        negative_probability, positive_probability, contains_negation,
        is_correct.

    Raises:
        ValueError: if lengths do not match, inputs are empty, or
            probabilities is not a 2D array with exactly two columns.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    if len(texts) == 0:
        raise ValueError("texts must not be empty.")
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must be two-dimensional, got shape "
            f"{probabilities.shape}."
        )
    if probabilities.shape[1] != 2:
        raise ValueError(
            f"probabilities must have exactly two class columns, got "
            f"{probabilities.shape[1]}."
        )
    lengths = {
        "texts": len(texts),
        "labels": len(labels),
        "predictions": len(predictions),
        "probabilities": probabilities.shape[0],
    }
    if negation_flags is not None:
        lengths["negation_flags"] = len(negation_flags)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All inputs must have the same length, got {lengths}.")

    if negation_flags is None:
        negation_flags = [contains_negation(text) for text in texts]

    return pd.DataFrame(
        {
            "text": list(texts),
            "true_label": labels.astype(np.int64),
            "predicted_label": predictions.astype(np.int64),
            "negative_probability": probabilities[:, 0].astype(np.float32),
            "positive_probability": probabilities[:, 1].astype(np.float32),
            "contains_negation": [bool(flag) for flag in negation_flags],
            "is_correct": (labels == predictions).astype(bool),
        }
    )


def _subset_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    """Metrics plus example count for one subset; safe values when empty."""
    if len(subset) == 0:
        return dict(_EMPTY_SUBSET_METRICS)
    metrics = calculate_classification_metrics(
        subset["true_label"].to_numpy(), subset["predicted_label"].to_numpy()
    )
    return {"examples": int(len(subset)), **metrics}


def evaluate_negation_subsets(records: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute metrics for all, with-negation, and without-negation examples.

    Accepts the DataFrame produced by build_prediction_records. Empty
    subsets report an example count of 0 and None for every metric.

    Returns:
        {"all": {...}, "with_negation": {...}, "without_negation": {...}},
        each containing "examples" plus the classification metrics.

    Raises:
        ValueError: if a required column is missing.
    """
    required = {"true_label", "predicted_label", "contains_negation"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing required columns: {sorted(missing)}.")
    with_negation = records[records["contains_negation"]]
    without_negation = records[~records["contains_negation"]]
    return {
        "all": _subset_metrics(records),
        "with_negation": _subset_metrics(with_negation),
        "without_negation": _subset_metrics(without_negation),
    }


def select_error_examples(
        records: pd.DataFrame, max_examples: int = 10
) -> pd.DataFrame:
    """Select a deterministic, balanced sample of misclassified examples.

    False positives (true 0, predicted 1) and false negatives (true 1,
    predicted 0) are interleaved in original order so both error types
    appear when both exist. An "error_type" column is added. At most
    max_examples rows are returned; with no errors, an empty DataFrame
    with the expected columns is returned.

    Raises:
        ValueError: if max_examples is not a positive integer or a
            required column is missing.
    """
    if (
            not isinstance(max_examples, int)
            or isinstance(max_examples, bool)
            or max_examples <= 0
    ):
        raise ValueError(
            f"max_examples must be a positive integer, got {max_examples!r}."
        )
    required = {"true_label", "predicted_label", "is_correct"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records is missing required columns: {sorted(missing)}.")

    errors = records[~records["is_correct"]].copy()
    expected_columns = list(records.columns) + ["error_type"]
    if len(errors) == 0:
        return pd.DataFrame(columns=expected_columns)

    errors["error_type"] = np.where(
        errors["predicted_label"] == POSITIVE_LABEL,
        "false_positive",
        "false_negative",
    )
    false_positives = errors[errors["error_type"] == "false_positive"]
    false_negatives = errors[errors["error_type"] == "false_negative"]

    # Alternate between the two error types (original order within each)
    # so the selection stays balanced and deterministic.
    selected_indices: list[Any] = []
    fp_indices = list(false_positives.index)
    fn_indices = list(false_negatives.index)
    for position in range(max(len(fp_indices), len(fn_indices))):
        if len(selected_indices) < max_examples and position < len(fp_indices):
            selected_indices.append(fp_indices[position])
        if len(selected_indices) < max_examples and position < len(fn_indices):
            selected_indices.append(fn_indices[position])
        if len(selected_indices) >= max_examples:
            break
    return errors.loc[selected_indices]


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into plain Python values.

    Raises:
        ValueError: if a non-finite float (NaN or infinity) is encountered.
    """
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite value {value} to JSON.")
    return value


def save_metrics(metrics: dict[str, Any], path: Path | str) -> None:
    """Save a metrics dictionary as sorted, indented UTF-8 JSON.

    NumPy scalars and arrays are converted to plain Python values first.
    Creates the parent directory if needed.

    Raises:
        ValueError: if the metrics contain non-finite float values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_json_compatible(metrics)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            serializable,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )


def save_prediction_records(records: pd.DataFrame, path: Path | str) -> None:
    """Save prediction records as a UTF-8 CSV without the index.

    Creates the parent directory if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path, index=False, encoding="utf-8")
