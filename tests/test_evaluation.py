"""Unit tests for src/evaluation.py using tiny synthetic data and models."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    RECORD_COLUMNS,
    build_prediction_records,
    calculate_classification_metrics,
    collect_predictions,
    count_trainable_parameters,
    evaluate_negation_subsets,
    measure_inference_time,
    save_metrics,
    save_prediction_records,
    select_error_examples,
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


class SignModel(nn.Module):
    """Parameter-free model predicting class 1 iff the first feature > 0.

    The logits equal (-x0, x0), which makes predictions and ordering
    fully deterministic for order-preservation tests.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first = features[:, 0]
        return torch.stack([-first, first], dim=1)


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


def make_records() -> pd.DataFrame:
    """Six-row prediction records with 2 false positives and 2 false negatives."""
    return build_prediction_records(
        texts=["t0", "t1", "t2", "t3", "t4", "t5"],
        labels=np.array([0, 0, 1, 1, 0, 1]),
        predictions=np.array([1, 1, 0, 0, 0, 1]),
        probabilities=np.array(
            [
                [0.4, 0.6],
                [0.3, 0.7],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        ),
        negation_flags=[True, False, True, False, True, False],
    )


# ---------------------------------------------------------------------------
# collect_predictions
# ---------------------------------------------------------------------------


def test_collect_predictions_bow_loader() -> None:
    torch.manual_seed(0)
    result = collect_predictions(TinyBoWModel(), make_bow_loader(), DEVICE)
    assert len(result["labels"]) == NUM_EXAMPLES
    assert set(result) == {"labels", "predictions", "probabilities", "logits"}


def test_collect_predictions_sequence_loader() -> None:
    torch.manual_seed(0)
    result = collect_predictions(
        TinySequenceModel(), make_sequence_loader(), DEVICE
    )
    assert len(result["labels"]) == NUM_EXAMPLES
    assert result["probabilities"].shape == (NUM_EXAMPLES, 2)


def test_collect_predictions_shapes_and_dtypes() -> None:
    torch.manual_seed(0)
    result = collect_predictions(TinyBoWModel(), make_bow_loader(), DEVICE)
    assert result["labels"].shape == (NUM_EXAMPLES,)
    assert result["predictions"].shape == (NUM_EXAMPLES,)
    assert result["probabilities"].shape == (NUM_EXAMPLES, 2)
    assert result["logits"].shape == (NUM_EXAMPLES, 2)
    assert result["labels"].dtype == np.int64
    assert result["predictions"].dtype == np.int64
    assert result["probabilities"].dtype == np.float32
    assert result["logits"].dtype == np.float32


def test_collect_predictions_probabilities_sum_to_one() -> None:
    torch.manual_seed(0)
    result = collect_predictions(TinyBoWModel(), make_bow_loader(), DEVICE)
    np.testing.assert_allclose(
        result["probabilities"].sum(axis=1), np.ones(NUM_EXAMPLES), rtol=1e-5
    )


def test_collect_predictions_preserves_order() -> None:
    # Strictly increasing first feature makes every example identifiable.
    features = torch.zeros(NUM_EXAMPLES, NUM_FEATURES)
    features[:, 0] = torch.arange(NUM_EXAMPLES, dtype=torch.float32) - 3.5
    labels = torch.arange(NUM_EXAMPLES) % 2
    loader = DataLoader(TensorDataset(features, labels), batch_size=3)

    result = collect_predictions(SignModel(), loader, DEVICE)
    np.testing.assert_array_equal(result["labels"], labels.numpy())
    np.testing.assert_allclose(
        result["logits"][:, 1], features[:, 0].numpy(), rtol=1e-6
    )
    np.testing.assert_array_equal(
        result["predictions"], (features[:, 0] > 0).long().numpy()
    )


def test_collect_predictions_empty_loader_raises() -> None:
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, NUM_FEATURES), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    with pytest.raises(ValueError):
        collect_predictions(TinyBoWModel(), empty_loader, DEVICE)


# ---------------------------------------------------------------------------
# calculate_classification_metrics
# ---------------------------------------------------------------------------


def test_classification_metrics_known_example() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metrics = calculate_classification_metrics(labels, predictions)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert isinstance(metrics["confusion_matrix"], list)


def test_classification_metrics_zero_division_is_safe() -> None:
    # No positive predictions at all: precision must be 0, not an error.
    metrics = calculate_classification_metrics(
        np.array([1, 1, 0]), np.array([0, 0, 0])
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        (np.array([[0, 1]]), np.array([0, 1])),  # 2D labels
        (np.array([0, 1]), np.array([[0, 1]])),  # 2D predictions
        (np.array([]), np.array([])),  # empty
        (np.array([0, 1]), np.array([0, 1, 1])),  # length mismatch
    ],
)
def test_classification_metrics_input_validation(
        labels: np.ndarray, predictions: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        calculate_classification_metrics(labels, predictions)


# ---------------------------------------------------------------------------
# measure_inference_time
# ---------------------------------------------------------------------------


def test_measure_inference_time_returns_valid_values() -> None:
    torch.manual_seed(0)
    result = measure_inference_time(
        TinyBoWModel(), make_bow_loader(batch_size=2), DEVICE, warmup_batches=0
    )
    assert result["total_seconds"] >= 0.0
    assert result["examples"] == NUM_EXAMPLES
    assert result["seconds_per_example"] == pytest.approx(
        result["total_seconds"] / result["examples"]
    )


def test_measure_inference_time_sequence_loader() -> None:
    torch.manual_seed(0)
    result = measure_inference_time(
        TinySequenceModel(),
        make_sequence_loader(batch_size=4),
        DEVICE,
        warmup_batches=0,
    )
    assert result["examples"] == NUM_EXAMPLES


def test_measure_inference_time_excludes_warmup_examples() -> None:
    torch.manual_seed(0)
    # 4 batches of 2 examples; 1 warmup batch leaves 6 timed examples.
    result = measure_inference_time(
        TinyBoWModel(), make_bow_loader(batch_size=2), DEVICE, warmup_batches=1
    )
    assert result["examples"] == NUM_EXAMPLES - 2


@pytest.mark.parametrize("warmup_batches", [-1, 1.5, "1", True])
def test_measure_inference_time_invalid_warmup(warmup_batches: object) -> None:
    with pytest.raises(ValueError):
        measure_inference_time(
            TinyBoWModel(), make_bow_loader(), DEVICE, warmup_batches=warmup_batches
        )


def test_measure_inference_time_no_timed_batches_raises() -> None:
    # 2 batches of 4 examples; warming up 2 batches leaves nothing to time.
    with pytest.raises(ValueError):
        measure_inference_time(
            TinyBoWModel(), make_bow_loader(batch_size=4), DEVICE, warmup_batches=2
        )


# ---------------------------------------------------------------------------
# count_trainable_parameters
# ---------------------------------------------------------------------------


def test_count_trainable_parameters() -> None:
    model = TinyBoWModel()  # Linear(4, 2): 4*2 weights + 2 biases = 10.
    assert count_trainable_parameters(model) == 10


def test_count_trainable_parameters_ignores_frozen() -> None:
    model = TinyBoWModel()
    model.linear.bias.requires_grad = False
    assert count_trainable_parameters(model) == 8


# ---------------------------------------------------------------------------
# build_prediction_records
# ---------------------------------------------------------------------------


def test_build_prediction_records_with_provided_flags() -> None:
    records = make_records()
    assert list(records.columns) == RECORD_COLUMNS
    assert records["text"].tolist() == ["t0", "t1", "t2", "t3", "t4", "t5"]
    assert records["contains_negation"].tolist() == [
        True, False, True, False, True, False,
    ]
    assert records["is_correct"].tolist() == [
        False, False, False, False, True, True,
    ]
    assert records["positive_probability"].iloc[0] == pytest.approx(0.6)
    assert records["negative_probability"].iloc[0] == pytest.approx(0.4)


def test_build_prediction_records_automatic_negation_detection() -> None:
    records = build_prediction_records(
        texts=["This is not good.", "A great movie!"],
        labels=np.array([0, 1]),
        predictions=np.array([0, 1]),
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )
    assert records["contains_negation"].tolist() == [True, False]


def test_build_prediction_records_validation() -> None:
    probabilities = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):  # empty inputs
        build_prediction_records([], np.array([]), np.array([]), np.empty((0, 2)))
    with pytest.raises(ValueError):  # length mismatch
        build_prediction_records(
            ["a"], np.array([0, 1]), np.array([0, 1]), probabilities
        )
    with pytest.raises(ValueError):  # 1D probabilities
        build_prediction_records(
            ["a", "b"], np.array([0, 1]), np.array([0, 1]), np.array([0.5, 0.5])
        )
    with pytest.raises(ValueError):  # wrong number of class columns
        build_prediction_records(
            ["a", "b"],
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]),
        )


# ---------------------------------------------------------------------------
# evaluate_negation_subsets
# ---------------------------------------------------------------------------


def test_evaluate_negation_subsets_separate_metrics() -> None:
    records = make_records()
    result = evaluate_negation_subsets(records)
    assert set(result) == {"all", "with_negation", "without_negation"}
    assert result["all"]["examples"] == 6
    assert result["all"]["accuracy"] == pytest.approx(2 / 6)
    # Negation rows t0, t2, t4: only t4 is correct.
    assert result["with_negation"]["examples"] == 3
    assert result["with_negation"]["accuracy"] == pytest.approx(1 / 3)
    # Non-negation rows t1, t3, t5: only t5 is correct.
    assert result["without_negation"]["examples"] == 3
    assert result["without_negation"]["accuracy"] == pytest.approx(1 / 3)


def test_evaluate_negation_subsets_empty_subset_is_safe() -> None:
    records = build_prediction_records(
        texts=["not bad", "never dull"],
        labels=np.array([1, 1]),
        predictions=np.array([1, 0]),
        probabilities=np.array([[0.3, 0.7], [0.6, 0.4]]),
        negation_flags=[True, True],
    )
    result = evaluate_negation_subsets(records)
    empty = result["without_negation"]
    assert empty == {
        "examples": 0,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None,
    }
    assert result["with_negation"]["examples"] == 2


def test_evaluate_negation_subsets_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_negation_subsets(pd.DataFrame({"true_label": [0]}))


# ---------------------------------------------------------------------------
# select_error_examples
# ---------------------------------------------------------------------------


def test_select_error_examples_balanced_and_deterministic() -> None:
    records = make_records()  # FPs at rows 0, 1; FNs at rows 2, 3.
    first = select_error_examples(records, max_examples=2)
    second = select_error_examples(records, max_examples=2)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2
    assert sorted(first["error_type"]) == ["false_negative", "false_positive"]
    assert list(first.columns) == RECORD_COLUMNS + ["error_type"]


def test_select_error_examples_returns_all_errors_when_few() -> None:
    records = make_records()
    selected = select_error_examples(records, max_examples=10)
    assert len(selected) == 4
    assert (~selected["is_correct"]).all()
    assert selected["error_type"].tolist().count("false_positive") == 2
    assert selected["error_type"].tolist().count("false_negative") == 2


def test_select_error_examples_no_errors() -> None:
    records = build_prediction_records(
        texts=["good", "bad"],
        labels=np.array([1, 0]),
        predictions=np.array([1, 0]),
        probabilities=np.array([[0.1, 0.9], [0.8, 0.2]]),
        negation_flags=[False, False],
    )
    selected = select_error_examples(records)
    assert len(selected) == 0
    assert list(selected.columns) == RECORD_COLUMNS + ["error_type"]


@pytest.mark.parametrize("max_examples", [0, -1, 1.5, "3", True])
def test_select_error_examples_invalid_max_examples(max_examples: object) -> None:
    with pytest.raises(ValueError):
        select_error_examples(make_records(), max_examples=max_examples)


# ---------------------------------------------------------------------------
# save_metrics / save_prediction_records
# ---------------------------------------------------------------------------


def test_save_metrics_round_trip(tmp_path) -> None:
    metrics = {"accuracy": 0.75, "confusion_matrix": [[1, 1], [0, 2]], "examples": 4}
    path = tmp_path / "nested" / "metrics.json"
    save_metrics(metrics, path)
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded == metrics


def test_save_metrics_numpy_values(tmp_path) -> None:
    metrics = {
        "accuracy": np.float32(0.5),
        "examples": np.int64(8),
        "matrix": np.array([[1, 2], [3, 4]]),
        "flag": np.bool_(True),
    }
    path = tmp_path / "metrics.json"
    save_metrics(metrics, path)
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded["accuracy"] == pytest.approx(0.5)
    assert loaded["examples"] == 8
    assert loaded["matrix"] == [[1, 2], [3, 4]]
    assert loaded["flag"] is True


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), np.float64("-inf")])
def test_save_metrics_rejects_non_finite(tmp_path, bad_value: float) -> None:
    with pytest.raises(ValueError):
        save_metrics({"accuracy": bad_value}, tmp_path / "metrics.json")


def test_save_prediction_records_round_trip(tmp_path) -> None:
    records = make_records()
    path = tmp_path / "outputs" / "predictions.csv"
    save_prediction_records(records, path)
    loaded = pd.read_csv(path, encoding="utf-8")
    assert list(loaded.columns) == RECORD_COLUMNS
    assert len(loaded) == len(records)
    assert loaded["text"].tolist() == records["text"].tolist()
    assert loaded["true_label"].tolist() == records["true_label"].tolist()
    np.testing.assert_allclose(
        loaded["positive_probability"].to_numpy(),
        records["positive_probability"].to_numpy(),
        rtol=1e-5,
    )
