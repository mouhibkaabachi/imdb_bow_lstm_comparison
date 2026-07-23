"""Unit tests for src/data.py and src/datasets.py using small synthetic data."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data import (
    create_data_splits,
    load_imdb_data,
    validate_class_distribution,
)
from src.datasets import BoWDataset, SequenceDataset
from src.preprocessing import PAD_ID, UNK_ID


def write_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    """Write a small review/sentiment CSV file and return its path."""
    frame = pd.DataFrame(rows, columns=["review", "sentiment"])
    csv_path = path / "imdb_sample.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def make_balanced_dataframe(rows_per_class: int = 50) -> pd.DataFrame:
    """Build a balanced DataFrame in the cleaned text/label format."""
    texts = [f"review number {i}" for i in range(2 * rows_per_class)]
    labels = [0] * rows_per_class + [1] * rows_per_class
    return pd.DataFrame({"text": texts, "label": labels})


def test_load_imdb_data_success(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [("Loved it", "positive"), ("Hated it", "negative")],
    )
    dataframe = load_imdb_data(csv_path)
    assert list(dataframe.columns) == ["text", "label"]
    assert len(dataframe) == 2


def test_load_imdb_data_maps_labels(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        [("Loved it", "positive"), ("Hated it", "negative")],
    )
    dataframe = load_imdb_data(csv_path)
    assert dataframe.loc[dataframe["text"] == "Loved it", "label"].item() == 1
    assert dataframe.loc[dataframe["text"] == "Hated it", "label"].item() == 0


def test_load_imdb_data_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"], "mood": ["positive"]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        load_imdb_data(csv_path)


def test_load_imdb_data_unknown_sentiment(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, [("Fine film", "neutral")])
    with pytest.raises(ValueError, match="Unknown sentiment labels"):
        load_imdb_data(csv_path)


def test_load_imdb_data_missing_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame(
        {"review": [None, "Fine"], "sentiment": ["positive", "negative"]}
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_imdb_data(csv_path)


def test_create_data_splits_is_reproducible() -> None:
    dataframe = make_balanced_dataframe()
    train_a, val_a, test_a = create_data_splits(dataframe, seed=42)
    train_b, val_b, test_b = create_data_splits(dataframe, seed=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_create_data_splits_no_overlap() -> None:
    dataframe = make_balanced_dataframe()
    train_df, val_df, test_df = create_data_splits(dataframe, seed=42)
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)
    assert len(train_df) + len(val_df) + len(test_df) == len(dataframe)


def test_create_data_splits_proportions() -> None:
    dataframe = make_balanced_dataframe(rows_per_class=100)
    train_df, val_df, test_df = create_data_splits(dataframe, seed=42)
    total = len(dataframe)
    assert len(train_df) / total == pytest.approx(0.70, abs=0.02)
    assert len(val_df) / total == pytest.approx(0.15, abs=0.02)
    assert len(test_df) / total == pytest.approx(0.15, abs=0.02)


def test_create_data_splits_preserve_class_balance() -> None:
    dataframe = make_balanced_dataframe(rows_per_class=100)
    for split in create_data_splits(dataframe, seed=42):
        proportions = validate_class_distribution(split)["proportions"]
        assert proportions[0] == pytest.approx(0.5, abs=0.05)
        assert proportions[1] == pytest.approx(0.5, abs=0.05)


def test_create_data_splits_rejects_empty_dataframe() -> None:
    empty = pd.DataFrame({"text": [], "label": []})
    with pytest.raises(ValueError, match="empty"):
        create_data_splits(empty)


def test_validate_class_distribution() -> None:
    dataframe = pd.DataFrame(
        {"text": ["a", "b", "c", "d"], "label": [0, 0, 0, 1]}
    )
    distribution = validate_class_distribution(dataframe)
    assert distribution["counts"] == {0: 3, 1: 1}
    assert distribution["proportions"][0] == pytest.approx(0.75)
    assert distribution["proportions"][1] == pytest.approx(0.25)


# --- PyTorch dataset tests -------------------------------------------------

TINY_VOCABULARY: dict[str, int] = {
    "<PAD>": PAD_ID,
    "<UNK>": UNK_ID,
    "good": 2,
    "movie": 3,
    "bad": 4,
}
TINY_TEXTS: list[str] = ["good movie", "bad movie", "good good movie"]
TINY_LABELS: list[int] = [1, 0, 1]


def test_bow_dataset_length() -> None:
    dataset = BoWDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY)
    assert len(dataset) == 3


def test_bow_dataset_count_values_and_dtype() -> None:
    dataset = BoWDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY)
    features, _ = dataset[2]  # "good good movie"
    assert features.dtype == torch.float32
    assert features.shape == (len(TINY_VOCABULARY),)
    assert features[TINY_VOCABULARY["good"]].item() == pytest.approx(2.0)
    assert features[TINY_VOCABULARY["movie"]].item() == pytest.approx(1.0)
    assert features[TINY_VOCABULARY["bad"]].item() == pytest.approx(0.0)


def test_bow_dataset_binary_values() -> None:
    dataset = BoWDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, binary=True)
    features, _ = dataset[2]  # "good good movie"
    assert features[TINY_VOCABULARY["good"]].item() == pytest.approx(1.0)
    assert features[TINY_VOCABULARY["movie"]].item() == pytest.approx(1.0)
    assert features[TINY_VOCABULARY["bad"]].item() == pytest.approx(0.0)


def test_bow_dataset_label_dtype() -> None:
    dataset = BoWDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY)
    _, label = dataset[0]
    assert label.dtype == torch.long
    assert label.item() == 1


def test_sequence_dataset_right_padding() -> None:
    dataset = SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=5)
    sequence, _, _ = dataset[0]  # "good movie" -> [2, 3] + padding
    assert sequence.shape == (5,)
    assert sequence.tolist() == [2, 3, PAD_ID, PAD_ID, PAD_ID]


def test_sequence_dataset_truncation() -> None:
    dataset = SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=2)
    sequence, _, _ = dataset[2]  # "good good movie" -> [2, 2, 3] truncated
    assert sequence.shape == (2,)
    assert sequence.tolist() == [2, 2]


def test_sequence_dataset_effective_length() -> None:
    dataset = SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=5)
    _, length, _ = dataset[0]  # "good movie" has 2 tokens
    assert length.item() == 2

    truncated = SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=2)
    _, length, _ = truncated[2]  # 3 tokens capped at max_length
    assert length.item() == 2


def test_sequence_dataset_dtypes() -> None:
    dataset = SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=5)
    sequence, length, label = dataset[1]
    assert sequence.dtype == torch.long
    assert length.dtype == torch.long
    assert label.dtype == torch.long
    assert label.item() == 0


def test_datasets_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        BoWDataset(TINY_TEXTS, [1, 0], TINY_VOCABULARY)
    with pytest.raises(ValueError, match="same length"):
        SequenceDataset(TINY_TEXTS, [1, 0], TINY_VOCABULARY, max_length=5)


def test_datasets_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        BoWDataset([], [], TINY_VOCABULARY)
    with pytest.raises(ValueError, match="empty"):
        SequenceDataset([], [], TINY_VOCABULARY, max_length=5)


@pytest.mark.parametrize("max_length", [0, -1])
def test_sequence_dataset_rejects_invalid_max_length(max_length: int) -> None:
    with pytest.raises(ValueError, match="max_length"):
        SequenceDataset(TINY_TEXTS, TINY_LABELS, TINY_VOCABULARY, max_length=max_length)


def test_load_imdb_data_removes_duplicate_reviews(tmp_path):
    csv_path = tmp_path / "duplicate_reviews.csv"

    dataframe = pd.DataFrame(
        {
            "review": [
                "A repeated positive review.",
                "A repeated positive review.",
                "A unique negative review.",
            ],
            "sentiment": [
                "positive",
                "positive",
                "negative",
            ],
        }
    )
    dataframe.to_csv(csv_path, index=False)

    loaded = load_imdb_data(csv_path)

    assert len(loaded) == 2
    assert loaded["text"].nunique() == 2


def test_load_imdb_data_rejects_conflicting_duplicate_labels(tmp_path):
    csv_path = tmp_path / "conflicting_reviews.csv"

    dataframe = pd.DataFrame(
        {
            "review": [
                "The same review text.",
                "The same review text.",
            ],
            "sentiment": [
                "positive",
                "negative",
            ],
        }
    )
    dataframe.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="conflicting sentiment labels"):
        load_imdb_data(csv_path)
