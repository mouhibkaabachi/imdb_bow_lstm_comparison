"""Unit tests for run_experiment.py using stubs instead of real training.

No test touches data/IMDB_Dataset.csv, trains a model on real data, or
writes outside tmp_path. The run_experiment workflow is exercised with
tiny DataFrames and deterministic stand-ins for the heavy functions.
"""

import json
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import run_experiment as experiment
from src.datasets import BoWDataset, SequenceDataset
from src.models import BoWFeedForwardClassifier, LSTMSentimentClassifier

TINY_VOCABULARY = {"<PAD>": 0, "<UNK>": 1, "bad": 2, "good": 3}


def make_frame(num_rows: int) -> pd.DataFrame:
    """Build a tiny alternating-label DataFrame of synthetic reviews."""
    texts = [
        "good movie" if index % 2 else "bad movie not good"
        for index in range(num_rows)
    ]
    labels = [index % 2 for index in range(num_rows)]
    return pd.DataFrame({"text": texts, "label": labels})


# ---------------------------------------------------------------------------
# parse_arguments
# ---------------------------------------------------------------------------


def test_parse_arguments_defaults() -> None:
    args = experiment.parse_arguments(["--model", "bow"])
    assert args.model == "bow"
    assert args.epochs == 8
    assert args.batch_size == 64
    assert args.learning_rate == pytest.approx(0.001)
    assert args.hidden_dim == 64
    assert args.dropout == pytest.approx(0.3)
    assert args.embedding_dim == 64
    assert args.max_sequence_length == experiment.MAX_SEQUENCE_LENGTH
    assert args.binary_bow is False
    assert args.patience == 2
    assert args.min_delta == pytest.approx(0.0)
    assert args.num_workers == 0
    assert args.device == "auto"
    assert args.data_path == experiment.DATA_PATH
    assert args.output_prefix is None


def test_parse_arguments_bow_options() -> None:
    args = experiment.parse_arguments(
        ["--model", "bow", "--binary-bow", "--hidden-dim", "32", "--epochs", "3"]
    )
    assert args.model == "bow"
    assert args.binary_bow is True
    assert args.hidden_dim == 32
    assert args.epochs == 3


def test_parse_arguments_lstm_options() -> None:
    args = experiment.parse_arguments(
        [
            "--model", "lstm",
            "--embedding-dim", "128",
            "--max-sequence-length", "50",
            "--batch-size", "16",
            "--learning-rate", "0.01",
            "--device", "cpu",
            "--output-prefix", "my_run",
        ]
    )
    assert args.model == "lstm"
    assert args.embedding_dim == 128
    assert args.max_sequence_length == 50
    assert args.batch_size == 16
    assert args.learning_rate == pytest.approx(0.01)
    assert args.device == "cpu"
    assert args.output_prefix == "my_run"


@pytest.mark.parametrize(
    "option",
    [
        "--epochs",
        "--batch-size",
        "--hidden-dim",
        "--embedding-dim",
        "--max-sequence-length",
        "--patience",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_arguments_rejects_non_positive_integers(
        option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "bow", option, value])


@pytest.mark.parametrize("value", ["0", "-0.5"])
def test_parse_arguments_rejects_non_positive_learning_rate(value: str) -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "bow", "--learning-rate", value])


@pytest.mark.parametrize("value", ["-0.1", "1.0", "1.5"])
def test_parse_arguments_rejects_invalid_dropout(value: str) -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "bow", "--dropout", value])


def test_parse_arguments_rejects_negative_min_delta() -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "bow", "--min-delta", "-0.01"])


def test_parse_arguments_rejects_negative_num_workers() -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "bow", "--num-workers", "-1"])


def test_parse_arguments_rejects_binary_bow_with_lstm() -> None:
    with pytest.raises(SystemExit):
        experiment.parse_arguments(["--model", "lstm", "--binary-bow"])


# ---------------------------------------------------------------------------
# select_device
# ---------------------------------------------------------------------------


def test_select_device_auto_falls_back_to_cpu(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert experiment.select_device("auto") == torch.device("cpu")


def test_select_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert experiment.select_device("auto") == torch.device("cuda")


def test_select_device_explicit_cpu() -> None:
    assert experiment.select_device("cpu") == torch.device("cpu")


def test_select_device_unavailable_cuda_raises(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        experiment.select_device("cuda")


# ---------------------------------------------------------------------------
# create_data_loaders
# ---------------------------------------------------------------------------


def make_loaders(model_name: str, binary_bow: bool = False) -> dict[str, DataLoader]:
    return experiment.create_data_loaders(
        model_name=model_name,
        train_dataframe=make_frame(6),
        validation_dataframe=make_frame(4),
        test_dataframe=make_frame(4),
        vocabulary=TINY_VOCABULARY,
        batch_size=2,
        max_sequence_length=8,
        binary_bow=binary_bow,
    )


def test_create_data_loaders_bow_dataset_classes() -> None:
    loaders = make_loaders("bow", binary_bow=True)
    assert set(loaders) == {"train", "validation", "test"}
    for loader in loaders.values():
        assert isinstance(loader.dataset, BoWDataset)
        assert loader.dataset.binary is True


def test_create_data_loaders_lstm_dataset_classes() -> None:
    loaders = make_loaders("lstm")
    assert set(loaders) == {"train", "validation", "test"}
    for loader in loaders.values():
        assert isinstance(loader.dataset, SequenceDataset)
        assert loader.dataset.max_length == 8


def test_create_data_loaders_only_train_is_shuffled() -> None:
    loaders = make_loaders("bow")
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["validation"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)


def test_create_data_loaders_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model name"):
        make_loaders("transformer")


# ---------------------------------------------------------------------------
# create_model_and_optimizer
# ---------------------------------------------------------------------------


def test_create_model_and_optimizer_bow() -> None:
    model, optimizer = experiment.create_model_and_optimizer(
        "bow", vocabulary_size=4, hidden_dim=8, dropout=0.1, learning_rate=0.01
    )
    assert isinstance(model, BoWFeedForwardClassifier)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_create_model_and_optimizer_lstm() -> None:
    model, optimizer = experiment.create_model_and_optimizer(
        "lstm",
        vocabulary_size=4,
        hidden_dim=8,
        dropout=0.1,
        learning_rate=0.001,
        embedding_dim=6,
    )
    assert isinstance(model, LSTMSentimentClassifier)
    assert isinstance(optimizer, torch.optim.Adam)
    assert model.embedding.embedding_dim == 6


def test_create_model_and_optimizer_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model name"):
        experiment.create_model_and_optimizer(
            "transformer", vocabulary_size=4, hidden_dim=8,
            dropout=0.1, learning_rate=0.01,
        )


# ---------------------------------------------------------------------------
# Run prefix sanitization
# ---------------------------------------------------------------------------


def test_sanitize_run_prefix_blocks_path_traversal() -> None:
    sanitized = experiment.sanitize_run_prefix("../../etc/passwd")
    assert "/" not in sanitized
    assert "\\" not in sanitized
    assert ".." not in sanitized
    assert sanitized == "etc_passwd"


def test_sanitize_run_prefix_rejects_empty_result() -> None:
    with pytest.raises(ValueError):
        experiment.sanitize_run_prefix("../..")


def test_determine_run_prefix_defaults() -> None:
    bow = experiment.parse_arguments(["--model", "bow"])
    binary = experiment.parse_arguments(["--model", "bow", "--binary-bow"])
    lstm = experiment.parse_arguments(["--model", "lstm"])
    assert experiment.determine_run_prefix(bow) == "bow_count"
    assert experiment.determine_run_prefix(binary) == "bow_binary"
    assert experiment.determine_run_prefix(lstm) == "lstm"


def test_determine_run_prefix_uses_sanitized_custom_prefix() -> None:
    args = experiment.parse_arguments(
        ["--model", "bow", "--output-prefix", "my run/1"]
    )
    assert experiment.determine_run_prefix(args) == "my_run_1"


# ---------------------------------------------------------------------------
# build_run_metadata
# ---------------------------------------------------------------------------


def make_metadata() -> dict[str, Any]:
    args = experiment.parse_arguments(["--model", "bow"])
    return experiment.build_run_metadata(
        args,
        device=torch.device("cpu"),
        vocabulary_size=4,
        split_sizes={"train": 6, "validation": 4, "test": 4},
        class_distributions={"train": {"counts": {0: 3, 1: 3}}},
    )


def test_build_run_metadata_is_json_serializable() -> None:
    json.dumps(make_metadata())


def test_build_run_metadata_contains_required_fields() -> None:
    metadata = make_metadata()
    assert metadata["random_seed"] == experiment.RANDOM_SEED
    assert metadata["split_sizes"] == {"train": 6, "validation": 4, "test": 4}
    assert metadata["hyperparameters"]["epochs"] == 8
    assert metadata["hyperparameters"]["learning_rate"] == pytest.approx(0.001)
    assert "test split is evaluated exactly once" in metadata["note"]
    assert metadata["model"] == "bow"
    assert metadata["device"] == "cpu"
    assert metadata["binary_bow"] is False


# ---------------------------------------------------------------------------
# run_experiment (fully stubbed workflow)
# ---------------------------------------------------------------------------

STUB_HISTORY = [
    {
        "epoch": 1,
        "train_loss": 0.7,
        "train_accuracy": 0.5,
        "validation_loss": 0.6,
        "validation_accuracy": 0.5,
    },
    {
        "epoch": 2,
        "train_loss": 0.5,
        "train_accuracy": 0.75,
        "validation_loss": 0.4,
        "validation_accuracy": 0.75,
    },
    {
        "epoch": 3,
        "train_loss": 0.4,
        "train_accuracy": 0.8,
        "validation_loss": 0.45,
        "validation_accuracy": 0.7,
    },
]

STUB_CLASSIFICATION = {
    "accuracy": 0.75,
    "precision": 0.75,
    "recall": 0.75,
    "f1": 0.75,
    "confusion_matrix": [[1, 1], [0, 2]],
}


@pytest.fixture
def stubbed_run(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> dict[str, Any]:
    """Patch run_experiment's collaborators with recording stubs.

    Returns a context dictionary with the call-order list, captured
    arguments, the synthetic split DataFrames, and the tmp directories.
    """
    calls: list[str] = []
    captured: dict[str, Any] = {}

    # Deterministic fake wall clock: perf_counter reads the clock, and each
    # stubbed workflow stage advances it by a known amount so tests can
    # verify exactly which stages are included in the measured duration.
    clock = {"now": 100.0}

    def fake_perf_counter() -> float:
        calls.append("perf_counter")
        return clock["now"]

    monkeypatch.setattr(
        experiment, "time", types.SimpleNamespace(perf_counter=fake_perf_counter)
    )

    def fake_cuda_synchronize(*args: Any, **kwargs: Any) -> None:
        calls.append("cuda_synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", fake_cuda_synchronize)

    train_frame = make_frame(6)
    validation_frame = make_frame(4)
    test_frame = make_frame(4)
    full_frame = make_frame(14)
    num_test = len(test_frame)

    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(experiment, "CHECKPOINTS_DIR", tmp_path / "checkpoints")

    def fake_load_imdb_data(path: Any) -> pd.DataFrame:
        calls.append("load_imdb_data")
        captured["data_path"] = path
        clock["now"] += 30.0
        return full_frame

    def fake_create_data_splits(
            dataframe: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        calls.append("create_data_splits")
        captured["split_seed"] = seed
        return train_frame, validation_frame, test_frame

    def fake_build_vocabulary(texts: Any) -> dict[str, int]:
        calls.append("build_vocabulary")
        captured["vocabulary_texts"] = list(texts)
        clock["now"] += 5.0
        return dict(TINY_VOCABULARY)

    def fake_save_vocabulary(vocabulary: dict[str, int], path: Any) -> None:
        calls.append("save_vocabulary")
        captured["vocabulary_path"] = path

    real_create_data_loaders = experiment.create_data_loaders

    def recording_create_data_loaders(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("create_data_loaders")
        loaders = real_create_data_loaders(*args, **kwargs)
        captured["loaders"] = loaders
        return loaders

    def fake_fit_model(**kwargs: Any) -> list[dict[str, float]]:
        calls.append("fit_model")
        captured["fit_kwargs"] = kwargs
        clock["now"] += 7.5
        return [dict(entry) for entry in STUB_HISTORY]

    def fake_collect_predictions(
            model: Any, data_loader: Any, device: Any
    ) -> dict[str, np.ndarray]:
        calls.append("collect_predictions")
        captured["predictions_loader"] = data_loader
        clock["now"] += 50.0
        return {
            "labels": np.array([0, 1, 0, 1], dtype=np.int64),
            "predictions": np.array([0, 1, 1, 1], dtype=np.int64),
            "probabilities": np.full((num_test, 2), 0.5, dtype=np.float32),
            "logits": np.zeros((num_test, 2), dtype=np.float32),
        }

    def fake_calculate_classification_metrics(
            labels: np.ndarray, predictions: np.ndarray
    ) -> dict[str, Any]:
        calls.append("calculate_classification_metrics")
        clock["now"] += 10.0
        return dict(STUB_CLASSIFICATION)

    def fake_measure_inference_time(
            model: Any, data_loader: Any, device: Any
    ) -> dict[str, Any]:
        calls.append("measure_inference_time")
        captured["timing_loader"] = data_loader
        clock["now"] += 25.0
        return {
            "total_seconds": 0.01,
            "examples": num_test,
            "seconds_per_example": 0.0025,
        }

    def fake_count_trainable_parameters(model: Any) -> int:
        calls.append("count_trainable_parameters")
        return 123

    real_build_prediction_records = experiment.build_prediction_records

    def recording_build_prediction_records(
            *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        calls.append("build_prediction_records")
        records = real_build_prediction_records(*args, **kwargs)
        captured["record_texts"] = list(records["text"])
        return records

    def fake_evaluate_negation_subsets(records: pd.DataFrame) -> dict[str, Any]:
        calls.append("evaluate_negation_subsets")
        return {
            "all": {"examples": num_test},
            "with_negation": {"examples": 2},
            "without_negation": {"examples": 2},
        }

    def fake_select_error_examples(
            records: pd.DataFrame, max_examples: int
    ) -> pd.DataFrame:
        calls.append("select_error_examples")
        captured["max_error_examples"] = max_examples
        return records[~records["is_correct"]]

    saved_metrics: list[Any] = []
    saved_records: list[Any] = []
    saved_payloads: dict[str, Any] = {}

    def fake_save_metrics(metrics: dict[str, Any], path: Any) -> None:
        calls.append("save_metrics")
        saved_metrics.append(path)
        saved_payloads[str(path)] = metrics
        clock["now"] += 3.0

    def fake_save_prediction_records(records: pd.DataFrame, path: Any) -> None:
        calls.append("save_prediction_records")
        saved_records.append(path)

    monkeypatch.setattr(experiment, "load_imdb_data", fake_load_imdb_data)
    monkeypatch.setattr(experiment, "create_data_splits", fake_create_data_splits)
    monkeypatch.setattr(experiment, "build_vocabulary", fake_build_vocabulary)
    monkeypatch.setattr(experiment, "save_vocabulary", fake_save_vocabulary)
    monkeypatch.setattr(
        experiment, "create_data_loaders", recording_create_data_loaders
    )
    monkeypatch.setattr(experiment, "fit_model", fake_fit_model)
    monkeypatch.setattr(experiment, "collect_predictions", fake_collect_predictions)
    monkeypatch.setattr(
        experiment,
        "calculate_classification_metrics",
        fake_calculate_classification_metrics,
    )
    monkeypatch.setattr(
        experiment, "measure_inference_time", fake_measure_inference_time
    )
    monkeypatch.setattr(
        experiment, "count_trainable_parameters", fake_count_trainable_parameters
    )
    monkeypatch.setattr(
        experiment, "build_prediction_records", recording_build_prediction_records
    )
    monkeypatch.setattr(
        experiment, "evaluate_negation_subsets", fake_evaluate_negation_subsets
    )
    monkeypatch.setattr(
        experiment, "select_error_examples", fake_select_error_examples
    )
    monkeypatch.setattr(experiment, "save_metrics", fake_save_metrics)
    monkeypatch.setattr(
        experiment, "save_prediction_records", fake_save_prediction_records
    )

    return {
        "calls": calls,
        "captured": captured,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "saved_metrics": saved_metrics,
        "saved_records": saved_records,
        "saved_payloads": saved_payloads,
        "tmp_path": tmp_path,
    }


def run_stubbed_experiment(extra_args: list[str] | None = None) -> dict[str, Any]:
    args = experiment.parse_arguments(
        ["--model", "bow", "--device", "cpu", "--epochs", "3"]
        + (extra_args or [])
    )
    return experiment.run_experiment(args)


def test_run_experiment_returns_expected_keys(stubbed_run: dict[str, Any]) -> None:
    result = run_stubbed_experiment()
    assert set(result) == {"history", "metrics", "metadata", "paths"}
    assert result["history"] == STUB_HISTORY
    assert result["metrics"]["classification"] == STUB_CLASSIFICATION
    assert result["metrics"]["trainable_parameters"] == 123


def test_run_experiment_best_epoch_from_minimum_validation_loss(
        stubbed_run: dict[str, Any],
) -> None:
    result = run_stubbed_experiment()
    assert result["metrics"]["best_epoch"] == 2
    assert result["metrics"]["best_validation_loss"] == pytest.approx(0.4)


def test_run_experiment_builds_vocabulary_from_training_text_only(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    expected = list(stubbed_run["train_frame"]["text"])
    assert stubbed_run["captured"]["vocabulary_texts"] == expected


def test_run_experiment_fits_on_train_and_validation_only(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    loaders = stubbed_run["captured"]["loaders"]
    fit_kwargs = stubbed_run["captured"]["fit_kwargs"]
    assert fit_kwargs["train_loader"] is loaders["train"]
    assert fit_kwargs["validation_loader"] is loaders["validation"]
    assert loaders["test"] not in fit_kwargs.values()


def test_run_experiment_evaluates_test_set_after_fit_model(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    calls = stubbed_run["calls"]
    loaders = stubbed_run["captured"]["loaders"]
    assert calls.index("fit_model") < calls.index("collect_predictions")
    assert calls.index("fit_model") < calls.index("measure_inference_time")
    assert stubbed_run["captured"]["predictions_loader"] is loaders["test"]
    assert stubbed_run["captured"]["timing_loader"] is loaders["test"]


def test_run_experiment_prediction_records_use_test_texts_in_order(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    expected = list(stubbed_run["test_frame"]["text"])
    assert stubbed_run["captured"]["record_texts"] == expected


def test_run_experiment_output_paths_use_run_prefix(
        stubbed_run: dict[str, Any],
) -> None:
    result = run_stubbed_experiment(["--binary-bow"])
    for path in result["paths"].values():
        assert "bow_binary" in path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert result["paths"]["metrics"].endswith("bow_binary_metrics.json")
    assert result["paths"]["predictions"].endswith(
        "bow_binary_test_predictions.csv"
    )


def test_run_experiment_custom_prefix_is_sanitized(
        stubbed_run: dict[str, Any],
) -> None:
    result = run_stubbed_experiment(["--output-prefix", "run one/../x"])
    for path in result["paths"].values():
        filename = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        assert filename.startswith("run_one")


def test_run_experiment_saves_all_artifacts(stubbed_run: dict[str, Any]) -> None:
    run_stubbed_experiment()
    # History, metrics and metadata JSON plus predictions and errors CSV.
    assert len(stubbed_run["saved_metrics"]) == 3
    assert len(stubbed_run["saved_records"]) == 2
    assert stubbed_run["calls"].count("save_vocabulary") == 1


def test_run_experiment_selects_at_most_ten_error_examples(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    assert stubbed_run["captured"]["max_error_examples"] == 10


def test_run_experiment_creates_directories_under_tmp_path(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    tmp_path = stubbed_run["tmp_path"]
    assert (tmp_path / "results" / "metrics").is_dir()
    assert (tmp_path / "results" / "predictions").is_dir()
    assert (tmp_path / "checkpoints").is_dir()


# ---------------------------------------------------------------------------
# Training-time measurement (training_seconds)
# ---------------------------------------------------------------------------


def test_run_experiment_times_fit_model_with_perf_counter(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    calls = stubbed_run["calls"]
    fit_index = calls.index("fit_model")
    # On CPU, perf_counter is read immediately before and after fit_model.
    assert calls[fit_index - 1] == "perf_counter"
    assert calls[fit_index + 1] == "perf_counter"


def test_run_experiment_training_seconds_covers_fit_model_only(
        stubbed_run: dict[str, Any],
) -> None:
    # The stubbed fit_model advances the fake clock by exactly 7.5 seconds;
    # data loading, vocabulary building, prediction collection, metric
    # calculation, inference timing, and saving advance it by much more.
    result = run_stubbed_experiment()
    assert result["metrics"]["training_seconds"] == pytest.approx(7.5)


def test_run_experiment_metrics_contain_non_negative_float_training_seconds(
        stubbed_run: dict[str, Any],
) -> None:
    result = run_stubbed_experiment()
    training_seconds = result["metrics"]["training_seconds"]
    assert isinstance(training_seconds, float)
    assert training_seconds >= 0.0
    json.dumps(training_seconds)


def test_run_experiment_saved_metrics_match_returned_training_seconds(
        stubbed_run: dict[str, Any],
) -> None:
    result = run_stubbed_experiment()
    saved = stubbed_run["saved_payloads"][result["paths"]["metrics"]]
    assert saved is result["metrics"]
    assert saved["training_seconds"] == result["metrics"]["training_seconds"]
    assert result["metadata"]["training_seconds"] == pytest.approx(
        result["metrics"]["training_seconds"]
    )


def test_run_experiment_synchronizes_cuda_around_fit_model(
        stubbed_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = experiment.parse_arguments(
        ["--model", "bow", "--device", "cuda", "--epochs", "3"]
    )
    result = experiment.run_experiment(args)
    calls = stubbed_run["calls"]
    fit_index = calls.index("fit_model")
    # Synchronize, read start time, fit, synchronize, read final time.
    assert calls[fit_index - 2:fit_index] == ["cuda_synchronize", "perf_counter"]
    assert calls[fit_index + 1:fit_index + 3] == ["cuda_synchronize", "perf_counter"]
    assert result["metrics"]["training_seconds"] == pytest.approx(7.5)


def test_run_experiment_does_not_synchronize_cuda_on_cpu(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    assert "cuda_synchronize" not in stubbed_run["calls"]


def test_run_experiment_fit_model_completes_before_test_evaluation(
        stubbed_run: dict[str, Any],
) -> None:
    run_stubbed_experiment()
    calls = stubbed_run["calls"]
    fit_index = calls.index("fit_model")
    assert fit_index < calls.index("collect_predictions")
    assert fit_index < calls.index("calculate_classification_metrics")
    assert fit_index < calls.index("measure_inference_time")
