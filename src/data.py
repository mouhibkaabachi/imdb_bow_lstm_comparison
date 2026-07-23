"""Loading and splitting of the IMDB sentiment dataset.

The CSV is expected to have two columns: "review" (the text) and
"sentiment" (either "negative" or "positive"). Labels are mapped to
integers: negative -> 0, positive -> 1.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE

REQUIRED_COLUMNS = ("review", "sentiment")
LABEL_MAPPING = {"negative": 0, "positive": 1}


def load_imdb_data(path: str | Path) -> pd.DataFrame:
    """Load the IMDB CSV and return a clean DataFrame.

    The returned DataFrame has two columns: "text" (str) and
    "label" (int, 0 = negative, 1 = positive).

    Raises:
        FileNotFoundError: if the CSV file does not exist.
        ValueError: if required columns are missing, the dataset is
            empty, values are missing, or sentiment labels are unknown.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {list(REQUIRED_COLUMNS)}."
        )

    if len(dataframe) == 0:
        raise ValueError("The dataset is empty.")

    if dataframe["review"].isna().any():
        raise ValueError("The 'review' column contains missing values.")
    if dataframe["sentiment"].isna().any():
        raise ValueError("The 'sentiment' column contains missing values.")

    unknown_labels = sorted(set(dataframe["sentiment"]) - set(LABEL_MAPPING))
    if unknown_labels:
        raise ValueError(
            f"Unknown sentiment labels: {unknown_labels}. "
            f"Allowed labels: {sorted(LABEL_MAPPING)}."
        )

    clean = pd.DataFrame(
        {
            "text": dataframe["review"].astype(str),
            "label": dataframe["sentiment"].map(LABEL_MAPPING).astype(int),
        }
    )

    # Detect identical reviews associated with contradictory labels.
    label_counts_per_text = clean.groupby("text")["label"].nunique()

    if (label_counts_per_text > 1).any():
        raise ValueError(
            "Identical review texts with conflicting sentiment labels were found."
        )

    # Remove duplicate reviews before creating the data splits.
    # This prevents the same review from appearing in train, validation, and test.
    clean = (
        clean
        .drop_duplicates(subset="text", keep="first")
        .reset_index(drop=True)
    )

    return clean


def create_data_splits(
        dataframe: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into stratified train/validation/test DataFrames.

    The proportions are 70% train, 15% validation and 15% test.
    Stratification on "label" preserves the class balance in each split.
    The test set must only be used for the final evaluation, never for
    hyperparameter selection.

    Raises:
        ValueError: if the DataFrame is empty or too small for a
            stratified three-way split.
    """
    if len(dataframe) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    class_counts = dataframe["label"].value_counts()
    if class_counts.min() < 3:
        raise ValueError(
            "Each class needs at least 3 examples for a stratified "
            "train/validation/test split."
        )

    holdout_size = VALIDATION_SIZE + TEST_SIZE  # 0.30
    train_df, holdout_df = train_test_split(
        dataframe,
        train_size=TRAIN_SIZE,
        test_size=holdout_size,
        stratify=dataframe["label"],
        random_state=seed,
    )
    # Split the 30% holdout in half: 15% validation and 15% test.
    validation_df, test_df = train_test_split(
        holdout_df,
        train_size=0.5,
        test_size=0.5,
        stratify=holdout_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def validate_class_distribution(dataframe: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Return label counts and proportions for a DataFrame.

    Returns a dictionary with two keys:
        "counts": mapping label -> number of rows,
        "proportions": mapping label -> fraction of rows.

    Raises:
        ValueError: if the DataFrame is empty.
    """
    if len(dataframe) == 0:
        raise ValueError("Cannot compute the class distribution of an empty DataFrame.")

    counts = dataframe["label"].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "counts": {int(label): int(count) for label, count in counts.items()},
        "proportions": {int(label): count / total for label, count in counts.items()},
    }
