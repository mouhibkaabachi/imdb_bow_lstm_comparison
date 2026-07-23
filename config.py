"""Project-wide configuration constants for the IMDB BoW vs LSTM comparison."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "IMDB_Dataset.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"

RANDOM_SEED: int = 42

TRAIN_SIZE: float = 0.70
VALIDATION_SIZE: float = 0.15
TEST_SIZE: float = 0.15

MAX_VOCAB_SIZE: int = 10000
MIN_TOKEN_FREQUENCY: int = 2
MAX_SEQUENCE_LENGTH: int = 200

NEGATION_MARKERS: frozenset[str] = frozenset(
    {"not", "no", "never", "n't", "hardly", "without"}
)

assert abs(TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE - 1.0) < 1e-9, (
    "Split proportions must sum to 1.0"
)
