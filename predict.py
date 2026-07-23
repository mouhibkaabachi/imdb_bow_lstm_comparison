"""Command-line inference for a trained IMDB sentiment model.

Loads an existing checkpoint and vocabulary produced by run_experiment.py
and classifies a single review text:

    python predict.py --model bow --text "This movie was excellent."
    python predict.py --model lstm --text "This movie was not good."

The script never trains, never updates parameters, and never reads the
IMDB CSV. The architecture options (--hidden-dim, --dropout,
--embedding-dim) must match the configuration used during training so
that the checkpoint's state dict fits the freshly created model.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from config import CHECKPOINTS_DIR, MAX_SEQUENCE_LENGTH, RESULTS_DIR
from run_experiment import (
    DEVICE_CHOICES,
    MODEL_CHOICES,
    _dropout_float,
    _positive_int,
    select_device,
)
from src.models import BoWFeedForwardClassifier, LSTMSentimentClassifier
from src.preprocessing import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    contains_negation,
    encode_text,
    load_vocabulary,
    pad_or_truncate,
    text_to_bow,
)
from src.training import load_checkpoint

SENTIMENT_NAMES = {0: "negative", 1: "positive"}


def _non_empty_text(value: str) -> str:
    """argparse type: a string containing at least one non-space character."""
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "expected a non-empty text, got an empty or whitespace-only string"
        )
    return value


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments for one prediction.

    Args:
        args: optional argument list (for testing); None uses sys.argv.

    Returns:
        A validated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Classify the sentiment of one review text with a trained "
            "bag-of-words baseline or LSTM checkpoint."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="which trained model to use: 'bow' or 'lstm'",
    )
    parser.add_argument(
        "--text",
        required=True,
        type=_non_empty_text,
        help="the review text to classify (must be non-empty)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="custom checkpoint path (requires --vocabulary as well)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="custom vocabulary JSON path (requires --checkpoint as well)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="computation device (default: auto)",
    )
    parser.add_argument(
        "--binary-bow",
        action="store_true",
        help="use presence/absence bag-of-words features, BoW only",
    )
    parser.add_argument(
        "--hidden-dim",
        type=_positive_int,
        default=64,
        help="hidden layer size used during training (default: 64)",
    )
    parser.add_argument(
        "--dropout",
        type=_dropout_float,
        default=0.3,
        help="dropout probability in [0, 1) used during training (default: 0.3)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=_positive_int,
        default=64,
        help="embedding size used during training, LSTM only (default: 64)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=_positive_int,
        default=MAX_SEQUENCE_LENGTH,
        help=f"sequence length after padding/truncation (default: {MAX_SEQUENCE_LENGTH})",
    )

    namespace = parser.parse_args(args)
    if namespace.model == "lstm" and namespace.binary_bow:
        parser.error("--binary-bow is only valid with --model bow")
    return namespace


def resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Determine which checkpoint and vocabulary files to load.

    Custom paths must be supplied together because a checkpoint is only
    valid with the vocabulary of the same training run. When neither is
    supplied, the run_experiment.py naming convention is used.

    Returns:
        (checkpoint_path, vocabulary_path); no files are created.

    Raises:
        ValueError: if only one of --checkpoint/--vocabulary is supplied.
    """
    if args.checkpoint is not None and args.vocabulary is not None:
        return args.checkpoint, args.vocabulary
    if args.checkpoint is not None or args.vocabulary is not None:
        raise ValueError(
            "--checkpoint and --vocabulary must be supplied together, "
            "because a checkpoint is only compatible with the vocabulary "
            "from the same training run."
        )

    if args.model == "bow":
        run_prefix = "bow_binary" if args.binary_bow else "bow_count"
    else:
        run_prefix = "lstm"
    checkpoint_path = CHECKPOINTS_DIR / f"{run_prefix}_best.pt"
    vocabulary_path = RESULTS_DIR / "metrics" / f"{run_prefix}_vocabulary.json"
    return checkpoint_path, vocabulary_path


def create_inference_model(
        model_name: str,
        vocabulary_size: int,
        hidden_dim: int,
        dropout: float,
        embedding_dim: int = 64,
        padding_idx: int = 0,
) -> nn.Module:
    """Instantiate the requested classifier for inference (no optimizer).

    The constructor arguments mirror run_experiment.create_model_and_optimizer
    so a training checkpoint loads without shape mismatches.

    Raises:
        ValueError: if model_name is not "bow" or "lstm".
    """
    if model_name == "bow":
        return BoWFeedForwardClassifier(
            vocab_size=vocabulary_size,
            hidden_dim=hidden_dim,
            output_dim=2,
            dropout=dropout,
        )
    if model_name == "lstm":
        return LSTMSentimentClassifier(
            vocab_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            padding_idx=padding_idx,
            dropout=dropout,
        )
    raise ValueError(
        f"Unsupported model name {model_name!r}; expected one of "
        f"{MODEL_CHOICES}."
    )


def prepare_model_input(
        text: str,
        model_name: str,
        vocabulary: dict[str, int],
        device: torch.device | str,
        binary_bow: bool = False,
        max_sequence_length: int = 200,
) -> tuple[torch.Tensor, ...]:
    """Convert one text into the batched model input tensors.

    Returns:
        For "bow": (features,) with shape (1, vocabulary_size), float32.
        For "lstm": (sequences, lengths) with shapes
        (1, max_sequence_length) and (1,), both torch.long.

    Raises:
        ValueError: if model_name is unsupported, or if an LSTM text
            produces zero tokens.
    """
    if model_name == "bow":
        vector = text_to_bow(text, vocabulary, binary=binary_bow)
        features = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        return (features.to(device),)
    if model_name == "lstm":
        encoded = encode_text(text, vocabulary)
        effective_length = min(len(encoded), max_sequence_length)
        if effective_length == 0:
            raise ValueError(
                f"Text {text!r} produced zero tokens; the LSTM needs at "
                "least one token."
            )
        padded = pad_or_truncate(
            encoded, max_length=max_sequence_length, pad_id=vocabulary[PAD_TOKEN]
        )
        sequences = torch.tensor(padded, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([effective_length], dtype=torch.long)
        return sequences.to(device), lengths.to(device)
    raise ValueError(
        f"Unsupported model name {model_name!r}; expected one of "
        f"{MODEL_CHOICES}."
    )


def predict_text(
        text: str,
        model: nn.Module,
        model_name: str,
        vocabulary: dict[str, int],
        device: torch.device | str,
        binary_bow: bool = False,
        max_sequence_length: int = 200,
) -> dict[str, Any]:
    """Classify one text and return a JSON-serializable result.

    Softmax is applied to the raw logits only for reporting the class
    probabilities; probabilities are returned with full precision.
    """
    model.eval()
    model_input = prepare_model_input(
        text=text,
        model_name=model_name,
        vocabulary=vocabulary,
        device=device,
        binary_bow=binary_bow,
        max_sequence_length=max_sequence_length,
    )
    with torch.no_grad():
        if model_name == "bow":
            (features,) = model_input
            logits = model(features)
        else:
            sequences, lengths = model_input
            logits = model(sequences, lengths)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_label = int(probabilities.argmax().item())
    return {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_sentiment": SENTIMENT_NAMES[predicted_label],
        "negative_probability": float(probabilities[0].item()),
        "positive_probability": float(probabilities[1].item()),
        "contains_negation": contains_negation(text),
    }


def load_model_for_inference(
        model: nn.Module,
        checkpoint_path: Path | str,
        device: torch.device | str,
) -> dict[str, Any]:
    """Load trained weights into a model and prepare it for inference.

    Only the model parameters are restored (optimizer=None); the model
    is moved to the device and set to eval mode.

    Returns:
        The checkpoint metadata dictionary from load_checkpoint.

    Raises:
        FileNotFoundError: if checkpoint_path does not exist.
    """
    metadata = load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.to(device)
    model.eval()
    return metadata


def run_prediction(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full inference workflow for one text.

    Returns:
        The predict_text dictionary extended with the model name, the
        device, both artifact paths and (if available) the checkpoint
        epoch. All values are JSON-serializable.

    Raises:
        FileNotFoundError: if the checkpoint or vocabulary file is missing.
        ValueError: if the vocabulary special tokens are invalid.
    """
    device = select_device(args.device)
    checkpoint_path, vocabulary_path = resolve_artifact_paths(args)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model "
            f"first with 'python run_experiment.py --model {args.model}'."
        )
    if not Path(vocabulary_path).exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocabulary_path}. Train the model "
            f"first with 'python run_experiment.py --model {args.model}'."
        )

    vocabulary = load_vocabulary(vocabulary_path)
    if vocabulary.get(PAD_TOKEN) != PAD_ID or vocabulary.get(UNK_TOKEN) != UNK_ID:
        raise ValueError(
            f"Invalid vocabulary at {vocabulary_path}: expected "
            f"{PAD_TOKEN!r} with ID {PAD_ID} and {UNK_TOKEN!r} with ID "
            f"{UNK_ID}."
        )

    model = create_inference_model(
        model_name=args.model,
        vocabulary_size=len(vocabulary),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
        padding_idx=vocabulary[PAD_TOKEN],
    )
    checkpoint_metadata = load_model_for_inference(model, checkpoint_path, device)

    result = predict_text(
        text=args.text,
        model=model,
        model_name=args.model,
        vocabulary=vocabulary,
        device=device,
        binary_bow=args.binary_bow,
        max_sequence_length=args.max_sequence_length,
    )
    result["model"] = args.model
    result["device"] = str(device)
    result["checkpoint_path"] = str(checkpoint_path)
    result["vocabulary_path"] = str(vocabulary_path)
    if "epoch" in checkpoint_metadata:
        result["checkpoint_epoch"] = checkpoint_metadata["epoch"]
    return result


def main() -> None:
    """Parse arguments, run one prediction, and print a short summary."""
    args = parse_arguments()
    result = run_prediction(args)
    print(f"Model: {result['model']}")
    print(f"Predicted sentiment: {result['predicted_sentiment']}")
    print(f"Negative probability: {result['negative_probability']:.4f}")
    print(f"Positive probability: {result['positive_probability']:.4f}")
    print(f"Negation detected: {result['contains_negation']}")


if __name__ == "__main__":
    main()
