"""Generate report figures and a summary table from verified experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

MODELS = {
    "Count BoW": "bow_count",
    "Binary BoW": "bow_binary",
    "LSTM": "lstm",
}


def load_json(path: Path) -> dict:
    """Load one JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_results() -> tuple[dict, dict]:
    """Load histories and final metrics for all experiments."""
    histories = {}
    metrics = {}

    for display_name, prefix in MODELS.items():
        histories[display_name] = load_json(
            METRICS_DIR / f"{prefix}_history.json"
        )["history"]

        metrics[display_name] = load_json(
            METRICS_DIR / f"{prefix}_metrics.json"
        )

    return histories, metrics


def plot_learning_curves(histories: dict) -> None:
    """Plot training and validation loss and accuracy."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, history in histories.items():
        epochs = [entry["epoch"] for entry in history]

        axes[0].plot(
            epochs,
            [entry["train_loss"] for entry in history],
            marker="o",
            label=f"{model_name}, train",
        )
        axes[0].plot(
            epochs,
            [entry["validation_loss"] for entry in history],
            marker="s",
            linestyle="--",
            label=f"{model_name}, validation",
        )

        axes[1].plot(
            epochs,
            [entry["train_accuracy"] for entry in history],
            marker="o",
            label=f"{model_name}, train",
        )
        axes[1].plot(
            epochs,
            [entry["validation_accuracy"] for entry in history],
            marker="s",
            linestyle="--",
            label=f"{model_name}, validation",
        )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "learning_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_confusion_matrices(metrics: dict) -> None:
    """Plot the test confusion matrix for each model."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))

    image = None

    for axis, (model_name, result) in zip(axes, metrics.items()):
        matrix = np.asarray(
            result["classification"]["confusion_matrix"],
            dtype=int,
        )

        image = axis.imshow(matrix, cmap="Blues")

        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                )

        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], labels=["Negative", "Positive"])
        axis.set_yticks([0, 1], labels=["Negative", "Positive"])

    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.025, pad=0.04)

    figure.suptitle("Test Confusion Matrices")
    figure.subplots_adjust(top=0.82, wspace=0.35)

    figure.savefig(
        FIGURES_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_predictive_performance(metrics: dict) -> None:
    """Compare test accuracy, precision, recall, and F1-score."""
    model_names = list(metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    x_positions = np.arange(len(model_names))
    width = 0.19

    figure, axis = plt.subplots(figsize=(10, 5))

    for index, (metric_name, label) in enumerate(
        zip(metric_names, labels)
    ):
        values = [
            metrics[model]["classification"][metric_name]
            for model in model_names
        ]

        axis.bar(
            x_positions + (index - 1.5) * width,
            values,
            width,
            label=label,
        )

    axis.set_title("Test Performance Comparison")
    axis.set_ylabel("Score")
    axis.set_ylim(0.70, 0.95)
    axis.set_xticks(x_positions, labels=model_names)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_computational_cost(metrics: dict) -> None:
    """Compare training time, inference time, and parameter count."""
    model_names = list(metrics.keys())

    training_times = [
        metrics[model]["training_seconds"]
        for model in model_names
    ]
    inference_times = [
        metrics[model]["inference_time"]["total_seconds"]
        for model in model_names
    ]
    parameter_counts = [
        metrics[model]["trainable_parameters"]
        for model in model_names
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].bar(model_names, training_times)
    axes[0].set_title("Training Time")
    axes[0].set_ylabel("Seconds, logarithmic scale")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(model_names, inference_times)
    axes[1].set_title("Timed Test Inference")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(model_names, parameter_counts)
    axes[2].set_title("Trainable Parameters")
    axes[2].set_ylabel("Parameter count")
    axes[2].grid(axis="y", alpha=0.3)

    for axis in axes:
        axis.tick_params(axis="x", rotation=15)

    figure.suptitle("Computational Cost Comparison")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "computational_cost.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_negation_performance(metrics: dict) -> None:
    """Compare accuracy with and without detected negation."""
    model_names = list(metrics.keys())

    with_negation = [
        metrics[model]["negation_subsets"]["with_negation"]["accuracy"]
        for model in model_names
    ]
    without_negation = [
        metrics[model]["negation_subsets"]["without_negation"]["accuracy"]
        for model in model_names
    ]

    x_positions = np.arange(len(model_names))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.bar(
        x_positions - width / 2,
        with_negation,
        width,
        label="With detected negation",
    )
    axis.bar(
        x_positions + width / 2,
        without_negation,
        width,
        label="Without detected negation",
    )

    axis.set_title("Accuracy by Negation Subset")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.75, 0.95)
    axis.set_xticks(x_positions, labels=model_names)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "negation_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_summary_table(metrics: dict) -> None:
    """Save the principal verified results in one CSV table."""
    rows = []

    for model_name, result in metrics.items():
        classification = result["classification"]
        negation = result["negation_subsets"]

        rows.append(
            {
                "model": model_name,
                "accuracy": classification["accuracy"],
                "precision": classification["precision"],
                "recall": classification["recall"],
                "f1": classification["f1"],
                "accuracy_with_negation": negation[
                    "with_negation"
                ]["accuracy"],
                "accuracy_without_negation": negation[
                    "without_negation"
                ]["accuracy"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result[
                    "best_validation_loss"
                ],
                "training_seconds": result["training_seconds"],
                "inference_seconds": result[
                    "inference_time"
                ]["total_seconds"],
                "seconds_per_example": result[
                    "inference_time"
                ]["seconds_per_example"],
                "trainable_parameters": result[
                    "trainable_parameters"
                ],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        FIGURES_DIR / "model_comparison_summary.csv",
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    """Generate all report-ready outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    histories, metrics = load_results()

    plot_learning_curves(histories)
    plot_confusion_matrices(metrics)
    plot_predictive_performance(metrics)
    plot_computational_cost(metrics)
    plot_negation_performance(metrics)
    save_summary_table(metrics)

    print("Generated report outputs:")
    print(FIGURES_DIR / "learning_curves.png")
    print(FIGURES_DIR / "confusion_matrices.png")
    print(FIGURES_DIR / "performance_comparison.png")
    print(FIGURES_DIR / "computational_cost.png")
    print(FIGURES_DIR / "negation_comparison.png")
    print(FIGURES_DIR / "model_comparison_summary.csv")


if __name__ == "__main__":
    main()