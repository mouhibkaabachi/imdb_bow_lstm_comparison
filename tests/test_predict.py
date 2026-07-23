"""Unit tests for predict.py using tiny vocabularies and tmp_path artifacts.

No test reads data/IMDB_Dataset.csv, trains a model, or writes outside
tmp_path. Checkpoints are created directly from small untrained models in
the same format as src.training.save_checkpoint.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

import predict
from src.models import BoWFeedForwardClassifier, LSTMSentimentClassifier
from src.preprocessing import save_vocabulary

TINY_VOCABULARY = {
    "<PAD>": 0,
    "<UNK>": 1,
    "bad": 2,
    "good": 3,
    "movie": 4,
    "not": 5,
}


def save_tiny_checkpoint(
        path: Path, model: torch.nn.Module, epoch: int = 3
) -> None:
    """Write a checkpoint file in the exact save_checkpoint format."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": epoch,
            "metrics": {"validation_loss": 0.5},
            "extra": {},
        },
        path,
    )


def make_bow_model(seed: int = 0) -> BoWFeedForwardClassifier:
    """Create a small deterministic BoW classifier."""
    torch.manual_seed(seed)
    return BoWFeedForwardClassifier(
        vocab_size=len(TINY_VOCABULARY), hidden_dim=4, output_dim=2, dropout=0.0
    )


def make_lstm_model(seed: int = 0) -> LSTMSentimentClassifier:
    """Create a small deterministic LSTM classifier."""
    torch.manual_seed(seed)
    return LSTMSentimentClassifier(
        vocab_size=len(TINY_VOCABULARY),
        embedding_dim=4,
        hidden_dim=4,
        output_dim=2,
        padding_idx=0,
        dropout=0.0,
    )


def write_artifacts(
        tmp_path: Path, model: torch.nn.Module, epoch: int = 3
) -> tuple[Path, Path]:
    """Write a checkpoint and vocabulary into tmp_path and return the paths."""
    checkpoint_path = tmp_path / "model_best.pt"
    vocabulary_path = tmp_path / "vocabulary.json"
    save_tiny_checkpoint(checkpoint_path, model, epoch=epoch)
    save_vocabulary(TINY_VOCABULARY, vocabulary_path)
    return checkpoint_path, vocabulary_path


# ---------------------------------------------------------------------------
# parse_arguments
# ---------------------------------------------------------------------------


def test_parse_arguments_bow_command() -> None:
    args = predict.parse_arguments(
        ["--model", "bow", "--text", "This movie was excellent."]
    )
    assert args.model == "bow"
    assert args.text == "This movie was excellent."
    assert args.checkpoint is None
    assert args.vocabulary is None
    assert args.device == "auto"
    assert args.binary_bow is False
    assert args.hidden_dim == 64
    assert args.dropout == pytest.approx(0.3)
    assert args.embedding_dim == 64
    assert args.max_sequence_length == predict.MAX_SEQUENCE_LENGTH


def test_parse_arguments_lstm_command() -> None:
    args = predict.parse_arguments(
        [
            "--model", "lstm",
            "--text", "This movie was not good.",
            "--hidden-dim", "32",
            "--embedding-dim", "16",
            "--max-sequence-length", "50",
            "--device", "cpu",
        ]
    )
    assert args.model == "lstm"
    assert args.text == "This movie was not good."
    assert args.hidden_dim == 32
    assert args.embedding_dim == 16
    assert args.max_sequence_length == 50
    assert args.device == "cpu"


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_parse_arguments_rejects_empty_text(text: str) -> None:
    with pytest.raises(SystemExit):
        predict.parse_arguments(["--model", "bow", "--text", text])


@pytest.mark.parametrize(
    "option", ["--hidden-dim", "--embedding-dim", "--max-sequence-length"]
)
@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_parse_arguments_rejects_invalid_positive_int(
        option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        predict.parse_arguments(
            ["--model", "bow", "--text", "fine", option, value]
        )


@pytest.mark.parametrize("value", ["-0.1", "1.0", "1.5"])
def test_parse_arguments_rejects_invalid_dropout(value: str) -> None:
    with pytest.raises(SystemExit):
        predict.parse_arguments(
            ["--model", "bow", "--text", "fine", "--dropout", value]
        )


def test_parse_arguments_rejects_binary_bow_with_lstm() -> None:
    with pytest.raises(SystemExit):
        predict.parse_arguments(
            ["--model", "lstm", "--text", "fine", "--binary-bow"]
        )


def test_parse_arguments_rejects_unsupported_model() -> None:
    with pytest.raises(SystemExit):
        predict.parse_arguments(["--model", "transformer", "--text", "fine"])


# ---------------------------------------------------------------------------
# resolve_artifact_paths
# ---------------------------------------------------------------------------


def test_resolve_artifact_paths_default_bow_count() -> None:
    args = predict.parse_arguments(["--model", "bow", "--text", "fine"])
    checkpoint_path, vocabulary_path = predict.resolve_artifact_paths(args)
    assert checkpoint_path == predict.CHECKPOINTS_DIR / "bow_count_best.pt"
    assert vocabulary_path == (
        predict.RESULTS_DIR / "metrics" / "bow_count_vocabulary.json"
    )


def test_resolve_artifact_paths_default_bow_binary() -> None:
    args = predict.parse_arguments(
        ["--model", "bow", "--text", "fine", "--binary-bow"]
    )
    checkpoint_path, vocabulary_path = predict.resolve_artifact_paths(args)
    assert checkpoint_path == predict.CHECKPOINTS_DIR / "bow_binary_best.pt"
    assert vocabulary_path == (
        predict.RESULTS_DIR / "metrics" / "bow_binary_vocabulary.json"
    )


def test_resolve_artifact_paths_default_lstm() -> None:
    args = predict.parse_arguments(["--model", "lstm", "--text", "fine"])
    checkpoint_path, vocabulary_path = predict.resolve_artifact_paths(args)
    assert checkpoint_path == predict.CHECKPOINTS_DIR / "lstm_best.pt"
    assert vocabulary_path == (
        predict.RESULTS_DIR / "metrics" / "lstm_vocabulary.json"
    )


def test_resolve_artifact_paths_custom_paths_used_together(
        tmp_path: Path,
) -> None:
    custom_checkpoint = tmp_path / "custom.pt"
    custom_vocabulary = tmp_path / "custom_vocab.json"
    args = predict.parse_arguments(
        [
            "--model", "bow",
            "--text", "fine",
            "--checkpoint", str(custom_checkpoint),
            "--vocabulary", str(custom_vocabulary),
        ]
    )
    checkpoint_path, vocabulary_path = predict.resolve_artifact_paths(args)
    assert checkpoint_path == custom_checkpoint
    assert vocabulary_path == custom_vocabulary


@pytest.mark.parametrize("option", ["--checkpoint", "--vocabulary"])
def test_resolve_artifact_paths_rejects_single_custom_path(
        tmp_path: Path, option: str
) -> None:
    args = predict.parse_arguments(
        ["--model", "bow", "--text", "fine", option, str(tmp_path / "one")]
    )
    with pytest.raises(ValueError):
        predict.resolve_artifact_paths(args)


# ---------------------------------------------------------------------------
# create_inference_model
# ---------------------------------------------------------------------------


def test_create_inference_model_returns_bow_classifier() -> None:
    model = predict.create_inference_model(
        "bow", vocabulary_size=len(TINY_VOCABULARY), hidden_dim=4, dropout=0.0
    )
    assert isinstance(model, BoWFeedForwardClassifier)
    assert model.fc2.out_features == 2


def test_create_inference_model_returns_lstm_classifier() -> None:
    model = predict.create_inference_model(
        "lstm",
        vocabulary_size=len(TINY_VOCABULARY),
        hidden_dim=4,
        dropout=0.0,
        embedding_dim=8,
    )
    assert isinstance(model, LSTMSentimentClassifier)
    assert model.embedding.embedding_dim == 8
    assert model.fc.out_features == 2


def test_create_inference_model_rejects_unsupported_name() -> None:
    with pytest.raises(ValueError):
        predict.create_inference_model(
            "transformer", vocabulary_size=10, hidden_dim=4, dropout=0.0
        )


# ---------------------------------------------------------------------------
# prepare_model_input
# ---------------------------------------------------------------------------


def test_prepare_model_input_bow_shape_and_dtype() -> None:
    (features,) = predict.prepare_model_input(
        "good movie", "bow", TINY_VOCABULARY, device="cpu"
    )
    assert features.shape == (1, len(TINY_VOCABULARY))
    assert features.dtype == torch.float32


def test_prepare_model_input_binary_bow_values() -> None:
    (features,) = predict.prepare_model_input(
        "good good movie",
        "bow",
        TINY_VOCABULARY,
        device="cpu",
        binary_bow=True,
    )
    # Presence/absence only: even the repeated token stays at 1.0.
    assert features[0, TINY_VOCABULARY["good"]].item() == pytest.approx(1.0)
    assert features[0, TINY_VOCABULARY["movie"]].item() == pytest.approx(1.0)
    assert set(features.unique().tolist()) <= {0.0, 1.0}


def test_prepare_model_input_lstm_shapes_and_dtypes() -> None:
    sequences, lengths = predict.prepare_model_input(
        "not good",
        "lstm",
        TINY_VOCABULARY,
        device="cpu",
        max_sequence_length=10,
    )
    assert sequences.shape == (1, 10)
    assert sequences.dtype == torch.long
    assert lengths.shape == (1,)
    assert lengths.dtype == torch.long
    assert lengths.item() == 2
    # Right-padded with the <PAD> ID after the real tokens.
    assert sequences[0, 2:].tolist() == [TINY_VOCABULARY["<PAD>"]] * 8


def test_prepare_model_input_lstm_effective_length_after_truncation() -> None:
    sequences, lengths = predict.prepare_model_input(
        "good bad movie not good bad",
        "lstm",
        TINY_VOCABULARY,
        device="cpu",
        max_sequence_length=4,
    )
    assert sequences.shape == (1, 4)
    assert lengths.item() == 4


def test_prepare_model_input_lstm_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError):
        predict.prepare_model_input(
            "   ", "lstm", TINY_VOCABULARY, device="cpu"
        )


def test_prepare_model_input_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError):
        predict.prepare_model_input(
            "fine", "transformer", TINY_VOCABULARY, device="cpu"
        )


# ---------------------------------------------------------------------------
# predict_text
# ---------------------------------------------------------------------------

EXPECTED_RESULT_KEYS = {
    "text",
    "predicted_label",
    "predicted_sentiment",
    "negative_probability",
    "positive_probability",
    "contains_negation",
}


def run_bow_prediction(text: str) -> dict[str, Any]:
    """Run predict_text with a deterministic small BoW model."""
    model = make_bow_model()
    return predict.predict_text(
        text, model, "bow", TINY_VOCABULARY, device="cpu"
    )


def test_predict_text_output_keys() -> None:
    result = run_bow_prediction("good movie")
    assert set(result) == EXPECTED_RESULT_KEYS


def test_predict_text_probabilities_sum_to_one() -> None:
    result = run_bow_prediction("good movie")
    total = result["negative_probability"] + result["positive_probability"]
    assert total == pytest.approx(1.0)


def test_predict_text_sentiment_matches_label() -> None:
    result = run_bow_prediction("good movie")
    assert result["predicted_label"] in (0, 1)
    expected = "positive" if result["predicted_label"] == 1 else "negative"
    assert result["predicted_sentiment"] == expected


def test_predict_text_preserves_original_text() -> None:
    text = "  This Movie was NOT good!!  "
    result = run_bow_prediction(text)
    assert result["text"] == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [("this was not good", True), ("this was good", False)],
)
def test_predict_text_includes_negation_detection(
        text: str, expected: bool
) -> None:
    result = run_bow_prediction(text)
    assert result["contains_negation"] is expected


def test_predict_text_works_with_lstm_model() -> None:
    model = make_lstm_model()
    result = predict.predict_text(
        "not good movie",
        model,
        "lstm",
        TINY_VOCABULARY,
        device="cpu",
        max_sequence_length=10,
    )
    assert set(result) == EXPECTED_RESULT_KEYS
    assert result["contains_negation"] is True


# ---------------------------------------------------------------------------
# load_model_for_inference
# ---------------------------------------------------------------------------


def test_load_model_for_inference_restores_parameters(tmp_path: Path) -> None:
    trained = make_bow_model(seed=1)
    checkpoint_path = tmp_path / "bow_best.pt"
    save_tiny_checkpoint(checkpoint_path, trained, epoch=5)

    fresh = make_bow_model(seed=2)
    metadata = predict.load_model_for_inference(fresh, checkpoint_path, "cpu")

    assert metadata["epoch"] == 5
    assert not fresh.training  # eval mode
    for name, parameter in fresh.state_dict().items():
        assert torch.equal(parameter, trained.state_dict()[name]), name


def test_load_model_for_inference_missing_checkpoint(tmp_path: Path) -> None:
    model = make_bow_model()
    with pytest.raises(FileNotFoundError):
        predict.load_model_for_inference(
            model, tmp_path / "does_not_exist.pt", "cpu"
        )


# ---------------------------------------------------------------------------
# run_prediction
# ---------------------------------------------------------------------------


def make_run_args(
        checkpoint_path: Path,
        vocabulary_path: Path,
        model: str = "bow",
        text: str = "good movie",
) -> Any:
    """Build a parsed Namespace pointing at tmp_path artifacts."""
    return predict.parse_arguments(
        [
            "--model", model,
            "--text", text,
            "--checkpoint", str(checkpoint_path),
            "--vocabulary", str(vocabulary_path),
            "--device", "cpu",
            "--hidden-dim", "4",
            "--dropout", "0.0",
            "--embedding-dim", "4",
        ]
    )


def test_run_prediction_returns_extended_fields(tmp_path: Path) -> None:
    checkpoint_path, vocabulary_path = write_artifacts(
        tmp_path, make_bow_model(), epoch=3
    )
    args = make_run_args(checkpoint_path, vocabulary_path)
    result = predict.run_prediction(args)

    assert result["model"] == "bow"
    assert result["device"] == "cpu"
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["vocabulary_path"] == str(vocabulary_path)
    assert result["checkpoint_epoch"] == 3
    assert EXPECTED_RESULT_KEYS <= set(result)


def test_run_prediction_lstm_end_to_end(tmp_path: Path) -> None:
    checkpoint_path, vocabulary_path = write_artifacts(
        tmp_path, make_lstm_model()
    )
    args = make_run_args(
        checkpoint_path, vocabulary_path, model="lstm", text="not good movie"
    )
    result = predict.run_prediction(args)
    assert result["model"] == "lstm"
    assert result["contains_negation"] is True


def test_run_prediction_missing_checkpoint(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "vocabulary.json"
    save_vocabulary(TINY_VOCABULARY, vocabulary_path)
    args = make_run_args(tmp_path / "missing.pt", vocabulary_path)
    with pytest.raises(FileNotFoundError):
        predict.run_prediction(args)


def test_run_prediction_missing_vocabulary(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bow_best.pt"
    save_tiny_checkpoint(checkpoint_path, make_bow_model())
    args = make_run_args(checkpoint_path, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        predict.run_prediction(args)


@pytest.mark.parametrize(
    "vocabulary",
    [
        {"<PAD>": 1, "<UNK>": 0, "good": 2},  # swapped special IDs
        {"<UNK>": 1, "good": 2},  # missing <PAD>
        {"<PAD>": 0, "good": 2},  # missing <UNK>
    ],
)
def test_run_prediction_invalid_special_tokens(
        tmp_path: Path, vocabulary: dict[str, int]
) -> None:
    checkpoint_path = tmp_path / "bow_best.pt"
    vocabulary_path = tmp_path / "vocabulary.json"
    save_tiny_checkpoint(checkpoint_path, make_bow_model())
    save_vocabulary(vocabulary, vocabulary_path)
    args = make_run_args(checkpoint_path, vocabulary_path)
    with pytest.raises(ValueError):
        predict.run_prediction(args)


def test_run_prediction_never_reads_imdb_csv(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas

    def fail_read_csv(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("run_prediction must not read any CSV file.")

    monkeypatch.setattr(pandas, "read_csv", fail_read_csv)
    checkpoint_path, vocabulary_path = write_artifacts(
        tmp_path, make_bow_model()
    )
    args = make_run_args(checkpoint_path, vocabulary_path)
    result = predict.run_prediction(args)
    assert result["predicted_label"] in (0, 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_prints_short_summary(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path, vocabulary_path = write_artifacts(
        tmp_path, make_bow_model()
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict.py",
            "--model", "bow",
            "--text", "good movie",
            "--checkpoint", str(checkpoint_path),
            "--vocabulary", str(vocabulary_path),
            "--device", "cpu",
            "--hidden-dim", "4",
            "--dropout", "0.0",
        ],
    )
    predict.main()
    output = capsys.readouterr().out

    assert "Model: bow" in output
    assert "Predicted sentiment:" in output
    assert "Negative probability: 0." in output
    assert "Positive probability: 0." in output
    assert "Negation detected: False" in output
    # Short summary only: no tensors or raw checkpoint contents.
    assert "tensor" not in output
    assert "state_dict" not in output
    assert len(output.splitlines()) == 5
