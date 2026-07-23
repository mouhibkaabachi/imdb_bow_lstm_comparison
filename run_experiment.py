"""Command-line entry point for one full IMDB sentiment experiment.

Runs the complete scientific workflow for a single model:

    python run_experiment.py --model bow
    python run_experiment.py --model lstm

The workflow seeds all random number generators, splits the data,
builds the vocabulary from the training texts only, trains with early
stopping on the validation loss, restores the best checkpoint, and only
then evaluates once on the held-out test split. All artifacts (history,
metrics, metadata, predictions, error examples, vocabulary, checkpoint)
are written under deterministic, run-prefixed file names.
"""

import argparse
import datetime
import platform
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import (
    CHECKPOINTS_DIR,
    DATA_PATH,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data import create_data_splits, load_imdb_data, validate_class_distribution
from src.datasets import BoWDataset, SequenceDataset
from src.evaluation import (
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
from src.models import BoWFeedForwardClassifier, LSTMSentimentClassifier
from src.preprocessing import build_vocabulary, save_vocabulary
from src.training import fit_model, set_random_seeds

MODEL_CHOICES = ("bow", "lstm")
DEVICE_CHOICES = ("auto", "cpu", "cuda")
MAX_ERROR_EXAMPLES = 10

_PREFIX_ALLOWED_PATTERN = re.compile(r"[^A-Za-z0-9_-]")


def _positive_int(value: str) -> int:
    """argparse type: an integer strictly greater than zero."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return number


def _non_negative_int(value: str) -> int:
    """argparse type: an integer greater than or equal to zero."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return number


def _positive_float(value: str) -> float:
    """argparse type: a float strictly greater than zero."""
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive number, got {value!r}"
        )
    return number


def _non_negative_float(value: str) -> float:
    """argparse type: a float greater than or equal to zero."""
    number = float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number, got {value!r}"
        )
    return number


def _dropout_float(value: str) -> float:
    """argparse type: a float in the half-open interval [0, 1)."""
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a dropout value in [0, 1), got {value!r}"
        )
    return number


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for one experiment run.

    Args:
        args: optional argument list (for testing); None uses sys.argv.

    Returns:
        A validated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate one IMDB sentiment model (bag-of-words "
            "baseline or LSTM) with a fixed train/validation/test protocol."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="which model to train: 'bow' or 'lstm'",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=8,
        help="maximum number of training epochs (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=64,
        help="mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=0.001,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=_positive_int,
        default=64,
        help="hidden layer size (default: 64)",
    )
    parser.add_argument(
        "--dropout",
        type=_dropout_float,
        default=0.3,
        help="dropout probability in [0, 1) (default: 0.3)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=_positive_int,
        default=64,
        help="embedding size, LSTM only (default: 64)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=_positive_int,
        default=MAX_SEQUENCE_LENGTH,
        help=f"sequence length after padding/truncation (default: {MAX_SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--binary-bow",
        action="store_true",
        help="use presence/absence bag-of-words features, BoW only",
    )
    parser.add_argument(
        "--patience",
        type=_positive_int,
        default=2,
        help="early-stopping patience in epochs (default: 2)",
    )
    parser.add_argument(
        "--min-delta",
        type=_non_negative_float,
        default=0.0,
        help="minimum validation-loss improvement (default: 0.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=0,
        help="DataLoader worker processes (default: 0)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="computation device (default: auto)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help=f"path to the IMDB CSV (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="optional custom prefix for all output file names",
    )

    namespace = parser.parse_args(args)
    if namespace.model == "lstm" and namespace.binary_bow:
        parser.error("--binary-bow is only valid with --model bow")
    return namespace


def select_device(device_option: str) -> torch.device:
    """Resolve a device option string to a concrete torch.device.

    "auto" prefers CUDA when available and falls back to CPU. "cuda"
    raises RuntimeError when CUDA is unavailable instead of silently
    running on the CPU.
    """
    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but no CUDA "
                "device is available. Use --device cpu or --device auto."
            )
        return torch.device("cuda")
    if device_option == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(
        f"Unknown device option {device_option!r}; expected one of "
        f"{DEVICE_CHOICES}."
    )


def sanitize_run_prefix(prefix: str) -> str:
    """Reduce a prefix to letters, digits, underscores and hyphens.

    Every other character (including path separators and dots) is
    replaced with an underscore, so a prefix like "../evil" cannot
    escape the results directories.

    Raises:
        ValueError: if nothing usable remains after sanitization.
    """
    sanitized = _PREFIX_ALLOWED_PATTERN.sub("_", prefix).strip("_-")
    if not sanitized:
        raise ValueError(
            f"Output prefix {prefix!r} contains no usable characters "
            "(letters, digits, underscores or hyphens)."
        )
    return sanitized


def determine_run_prefix(args: argparse.Namespace) -> str:
    """Return the run prefix for output files, sanitizing custom ones."""
    if args.output_prefix is not None:
        return sanitize_run_prefix(args.output_prefix)
    if args.model == "bow":
        return "bow_binary" if args.binary_bow else "bow_count"
    return "lstm"


def create_data_loaders(
        model_name: str,
        train_dataframe: Any,
        validation_dataframe: Any,
        test_dataframe: Any,
        vocabulary: dict[str, int],
        batch_size: int,
        max_sequence_length: int,
        binary_bow: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create train/validation/test DataLoaders for the chosen model.

    Only the training loader shuffles; validation and test order is
    preserved so predictions align with the split DataFrames. The
    vocabulary must already be built from the training texts.

    Returns:
        {"train": ..., "validation": ..., "test": ...}.

    Raises:
        ValueError: if model_name is not "bow" or "lstm".
    """
    def make_dataset(dataframe: Any) -> Dataset:
        texts = list(dataframe["text"])
        labels = list(dataframe["label"])
        if model_name == "bow":
            return BoWDataset(texts, labels, vocabulary, binary=binary_bow)
        if model_name == "lstm":
            return SequenceDataset(
                texts, labels, vocabulary, max_length=max_sequence_length
            )
        raise ValueError(
            f"Unsupported model name {model_name!r}; expected one of "
            f"{MODEL_CHOICES}."
        )

    loaders: dict[str, DataLoader] = {}
    splits = {
        "train": (train_dataframe, True),
        "validation": (validation_dataframe, False),
        "test": (test_dataframe, False),
    }
    for split_name, (dataframe, shuffle) in splits.items():
        loaders[split_name] = DataLoader(
            make_dataset(dataframe),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def create_model_and_optimizer(
        model_name: str,
        vocabulary_size: int,
        hidden_dim: int,
        dropout: float,
        learning_rate: float,
        embedding_dim: int = 64,
        padding_idx: int = 0,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Instantiate the requested classifier and its Adam optimizer.

    Both models use output_dim=2 and share the same default learning
    rate so the comparison stays fair.

    Raises:
        ValueError: if model_name is not "bow" or "lstm".
    """
    if model_name == "bow":
        model: nn.Module = BoWFeedForwardClassifier(
            vocab_size=vocabulary_size,
            hidden_dim=hidden_dim,
            output_dim=2,
            dropout=dropout,
        )
    elif model_name == "lstm":
        model = LSTMSentimentClassifier(
            vocab_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            padding_idx=padding_idx,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unsupported model name {model_name!r}; expected one of "
            f"{MODEL_CHOICES}."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def build_run_metadata(
        args: argparse.Namespace,
        device: torch.device,
        vocabulary_size: int,
        split_sizes: dict[str, int],
        class_distributions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a JSON-serializable description of one experiment run."""
    return {
        "model": args.model,
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "embedding_dim": args.embedding_dim,
            "max_sequence_length": args.max_sequence_length,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "vocabulary_size": vocabulary_size,
        "split_sizes": split_sizes,
        "class_distributions": class_distributions,
        "data_path": str(args.data_path),
        "binary_bow": args.binary_bow,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": (
            "The test split is evaluated exactly once, after training and "
            "model selection on the validation split are complete."
        ),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full train/validate/test workflow for one model.

    The vocabulary is built from the training texts only, checkpoint
    selection uses the validation loss, and the test split is touched
    exactly once after fit_model has restored the best checkpoint.

    Returns:
        {"history": ..., "metrics": ..., "metadata": ..., "paths": ...}.
    """
    set_random_seeds(RANDOM_SEED)
    device = select_device(args.device)

    metrics_dir = RESULTS_DIR / "metrics"
    predictions_dir = RESULTS_DIR / "predictions"
    for directory in (RESULTS_DIR, CHECKPOINTS_DIR, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_prefix = determine_run_prefix(args)
    paths: dict[str, Path] = {
        "vocabulary": metrics_dir / f"{run_prefix}_vocabulary.json",
        "checkpoint": CHECKPOINTS_DIR / f"{run_prefix}_best.pt",
        "history": metrics_dir / f"{run_prefix}_history.json",
        "metrics": metrics_dir / f"{run_prefix}_metrics.json",
        "metadata": metrics_dir / f"{run_prefix}_metadata.json",
        "predictions": predictions_dir / f"{run_prefix}_test_predictions.csv",
        "error_examples": predictions_dir / f"{run_prefix}_error_examples.csv",
    }

    dataframe = load_imdb_data(args.data_path)
    train_dataframe, validation_dataframe, test_dataframe = create_data_splits(
        dataframe, seed=RANDOM_SEED
    )
    split_sizes = {
        "train": len(train_dataframe),
        "validation": len(validation_dataframe),
        "test": len(test_dataframe),
    }
    class_distributions = {
        "train": validate_class_distribution(train_dataframe),
        "validation": validate_class_distribution(validation_dataframe),
        "test": validate_class_distribution(test_dataframe),
    }

    # Training texts only: validation and test must not leak into the vocabulary.
    vocabulary = build_vocabulary(train_dataframe["text"])
    save_vocabulary(vocabulary, paths["vocabulary"])

    loaders = create_data_loaders(
        model_name=args.model,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        vocabulary=vocabulary,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        binary_bow=args.binary_bow,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model, optimizer = create_model_and_optimizer(
        model_name=args.model,
        vocabulary_size=len(vocabulary),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
    )
    criterion = nn.CrossEntropyLoss()

    # Time only the fit_model training stage. CUDA kernels run
    # asynchronously, so synchronize before reading the clock on GPU runs.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = time.perf_counter()
    history = fit_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=args.epochs,
        checkpoint_path=paths["checkpoint"],
        patience=args.patience,
        min_delta=args.min_delta,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = float(time.perf_counter() - training_start)

    # fit_model has restored the best validation checkpoint; the test
    # split is evaluated here for the first and only time.
    predictions = collect_predictions(model, loaders["test"], device)
    classification = calculate_classification_metrics(
        predictions["labels"], predictions["predictions"]
    )
    inference_time = measure_inference_time(model, loaders["test"], device)
    trainable_parameters = count_trainable_parameters(model)

    records = build_prediction_records(
        texts=list(test_dataframe["text"]),
        labels=predictions["labels"],
        predictions=predictions["predictions"],
        probabilities=predictions["probabilities"],
    )
    negation_subsets = evaluate_negation_subsets(records)
    error_examples = select_error_examples(records, max_examples=MAX_ERROR_EXAMPLES)

    best_entry = min(history, key=lambda entry: entry["validation_loss"])
    metrics = {
        "classification": classification,
        "negation_subsets": negation_subsets,
        "inference_time": inference_time,
        "trainable_parameters": trainable_parameters,
        "training_seconds": training_seconds,
        "best_validation_loss": best_entry["validation_loss"],
        "best_epoch": best_entry["epoch"],
    }
    metadata = build_run_metadata(
        args, device, len(vocabulary), split_sizes, class_distributions
    )
    # The metrics JSON is the authoritative record; the copy here is for
    # convenience when reading run metadata on its own.
    metadata["training_seconds"] = training_seconds

    save_metrics({"history": history}, paths["history"])
    save_metrics(metrics, paths["metrics"])
    save_metrics(metadata, paths["metadata"])
    save_prediction_records(records, paths["predictions"])
    save_prediction_records(error_examples, paths["error_examples"])

    return {
        "history": history,
        "metrics": metrics,
        "metadata": metadata,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def main() -> None:
    """Parse arguments, run the experiment, and print a short summary."""
    args = parse_arguments()
    result = run_experiment(args)
    classification = result["metrics"]["classification"]
    print(f"Model: {result['metadata']['model']}")
    print(f"Device: {result['metadata']['device']}")
    print(f"Test accuracy: {classification['accuracy']:.4f}")
    print(f"Test F1: {classification['f1']:.4f}")
    print(f"Metrics saved to: {result['paths']['metrics']}")
    print(f"Predictions saved to: {result['paths']['predictions']}")


if __name__ == "__main__":
    main()
