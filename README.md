# IMDB Bag-of-Words vs LSTM Comparison

This individual university project for the course *Python for Natural Language Processing* implements binary sentiment classification of IMDB movie reviews and compares two neural approaches: a Bag-of-Words feedforward network, which discards word order, and an LSTM sequence model, which processes reviews as ordered token sequences. Both models are trained and evaluated under an identical experimental protocol, and the comparison covers predictive performance, behaviour on reviews containing negation, and computational cost (parameter count, total training time, and inference time). No claim is made here about which model performs better; conclusions are deferred until the experiments have been executed and verified.

## Research Question

Does sequence modelling with an LSTM improve IMDB sentiment classification over a bag-of-words feedforward baseline, particularly for reviews containing negation, and is the improvement worth the additional computational cost?

Subquestions:

- Does preserving word order improve classification performance?
- Do the two models differ on reviews containing negation?
- Does any performance difference justify the additional computational cost?

## Scientific Design

- The dataset is split into 70% training, 15% validation, and 15% test.
- Splits are stratified by sentiment label so class balance is preserved in every split.
- All random number generators are seeded with random seed 42.
- The vocabulary is built from the training texts only; validation and test texts never influence it.
- The validation loss drives early stopping and best-checkpoint selection.
- Final predictive metrics are calculated only after the best validation checkpoint has been restored. The test loader may then be traversed separately to measure inference time, without modifying the model or using the results for model selection or hyperparameter tuning.
- Both model families use the same data split and the same vocabulary.

The test set must not be used for hyperparameter selection. It is reserved exclusively for the single final evaluation.

## Models

### A. Bag-of-Words Feedforward Baseline

- Count-based Bag-of-Words features by default.
- Optional binary (presence/absence) Bag-of-Words via `--binary-bow`.
- One hidden linear layer.
- ReLU activation.
- Dropout.
- Two output logits (negative, positive).

### B. LSTM Sequence Model

- Input as token IDs.
- Trainable embedding layer.
- Right padding and truncation to a fixed maximum sequence length.
- Packed padded sequences using the true (pre-padding) sequence lengths.
- One-layer unidirectional LSTM.
- Dropout.
- Two output logits (negative, positive).

Both models are trained with `CrossEntropyLoss` and the Adam optimizer.

## Dataset

The code expects the dataset at:

```text
data/IMDB_Dataset.csv
```

Expected columns:

- `review` — the review text.
- `sentiment` — the sentiment label.

Expected sentiment values:

- `positive`
- `negative`

The dataset is intentionally excluded from version control and must be obtained separately from an appropriate legal source.

### Dataset Source and Licence

The project uses the CSV distribution of the Large Movie Review Dataset. The file contains 50,000 English-language IMDB movie reviews, balanced between 25,000 positive and 25,000 negative examples. The downloaded file was verified using the MD5 checksum `308443a50e5c993e7b8a1cdb95750026`.

The CSV distribution is archived on Zenodo under the Creative Commons Attribution 4.0 International licence. The dataset is associated with the following publication:

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150.

Source:
- Stanford Large Movie Review Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
- Zenodo CSV archive: https://zenodo.org/records/7928582

Before splitting, 418 duplicate rows representing 406 duplicated review texts were identified. No duplicated text had conflicting labels. Duplicate texts were removed programmatically to prevent identical reviews from appearing in multiple data splits. The final experimental dataset contained 49,582 unique reviews.
## Project Structure

```text
config.py                 # Project-wide constants: paths, seed, split sizes, vocabulary limits, negation markers
data/                     # Expected location of IMDB_Dataset.csv (not versioned)
checkpoints/              # Best model checkpoints saved during training
results/metrics/          # Vocabulary, training history, metrics, and metadata (JSON)
results/predictions/      # Test predictions and error-example CSV files
src/data.py               # CSV loading, validation, and stratified train/validation/test splitting
src/preprocessing.py      # Tokenization, vocabulary building, encoding, BoW vectors, negation detection
src/datasets.py           # PyTorch Dataset classes for BoW vectors and padded token sequences
src/models.py             # BoWFeedForwardClassifier and LSTMSentimentClassifier
src/training.py           # Seeding, training loops, early stopping, checkpoint save/load
src/evaluation.py         # Metrics, negation-subset analysis, timing, error examples, persistence
tests/                    # Unit tests for all modules and both entry points
run_experiment.py         # Command-line entry point for one full experiment run
predict.py                # Command-line inference for a single review text
requirements.txt          # Python dependencies
```

## Environment Setup

Python 3.12 is supported by the project.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Equivalent commands can be used on Linux or macOS (activate the environment with `source .venv/bin/activate`).

> **Note:** PyTorch installation may depend on the machine and CUDA configuration. If the generic `pip install` command is unsuitable for your system, consult the official PyTorch installation instructions.

## Running the Tests

Run the full test suite:

```powershell
python -m pytest -v
```

Run individual test files, for example:

```powershell
python -m pytest tests\test_preprocessing.py -v
python -m pytest tests\test_models.py -v
python -m pytest tests\test_run_experiment.py -v
```

Notes:

- The test suite uses small synthetic examples and temporary files.
- The full IMDB CSV is not required for the unit tests.
- Successful test results must be verified in an environment where all dependencies are installed.

## Training the Bag-of-Words Baseline

Default (count-based) configuration:

```powershell
python run_experiment.py --model bow
```

Binary (presence/absence) Bag-of-Words variation:

```powershell
python run_experiment.py --model bow --binary-bow
```

A moderate custom configuration example (not presented as the best configuration):

```powershell
python run_experiment.py --model bow --epochs 10 --batch-size 32 --learning-rate 0.0005 --hidden-dim 128 --dropout 0.4 --patience 3
```

## Training the LSTM

Default configuration:

```powershell
python run_experiment.py --model lstm
```

A moderate custom configuration example (not presented as an optimized or best configuration):

```powershell
python run_experiment.py --model lstm --epochs 10 --batch-size 32 --learning-rate 0.0005 --embedding-dim 128 --hidden-dim 128 --max-sequence-length 300 --patience 3
```

## Device Selection

Both `run_experiment.py` and `predict.py` accept a `--device` option:

```powershell
python run_experiment.py --model bow --device auto
python run_experiment.py --model bow --device cpu
python run_experiment.py --model bow --device cuda
```

- `auto` (the default) selects CUDA when available, otherwise CPU.
- `cuda` explicitly requests a GPU and raises an error if CUDA is unavailable.
- `cpu` remains fully supported.

## Generated Artifacts

Each run writes its artifacts under a deterministic prefix. For the ordinary (count-based) BoW model, the prefix is `bow_count`:

- `checkpoints/bow_count_best.pt`
- `results/metrics/bow_count_vocabulary.json`
- `results/metrics/bow_count_history.json`
- `results/metrics/bow_count_metrics.json`
- `results/metrics/bow_count_metadata.json`
- `results/predictions/bow_count_test_predictions.csv`
- `results/predictions/bow_count_error_examples.csv`

For binary BoW (`--binary-bow`), the prefix `bow_binary` replaces `bow_count`. For the LSTM, the prefix is `lstm`. A custom prefix can be set with `--output-prefix`.

Artifact contents:

- **History** contains all recorded training epochs (losses and accuracies).
- **Metrics** contain the final test metrics, negation-subset metrics, total training time, inference timing, trainable parameter count, best epoch, and best validation loss. The recorded training time measures only the `fit_model` training stage (wall-clock time of the training and validation epochs, including early stopping and checkpoint restoration); it excludes data preparation, vocabulary construction, and the final test evaluation.
- **Metadata** contains reproducibility information (seed, versions, device, split sizes, class distributions) and all hyperparameters.
- **The prediction CSV** contains example-level predictions and class probabilities for the test set.
- **The error-examples CSV** contains selected false positives and false negatives.

## Predicting New Reviews

```powershell
python predict.py --model bow --text "This movie was excellent."
python predict.py --model lstm --text "This movie was not good."
```

Prediction requires the matching checkpoint and vocabulary generated by training the corresponding model. Architecture options (`--hidden-dim`, `--dropout`, `--embedding-dim`) must match the values used during training.

Custom artifacts may be supplied, but only together:

```powershell
python predict.py --model lstm --text "Example review." --checkpoint path\to\model.pt --vocabulary path\to\vocabulary.json
```

The `--checkpoint` and `--vocabulary` paths must be provided together, because a checkpoint is only compatible with the vocabulary from the same training run.

## Reproducibility

- Random seed 42 for Python, NumPy, and PyTorch.
- All hyperparameters and run metadata are saved to JSON.
- Deterministic split generation (stratified, seeded).
- Fixed, deterministic vocabulary ordering.
- The checkpoint is selected by validation loss.
- The full epoch history is retained.
- The dataset and generated large artifacts are excluded from version control.

> **Note:** Exact numerical reproducibility may still vary across PyTorch versions, hardware, CPU/GPU execution, and CUDA library versions.

## Evaluation

Implemented evaluation outputs:

- Accuracy.
- Precision for the positive class (label 1).
- Recall for the positive class (label 1).
- F1-score.
- 2x2 confusion matrix.
- Trainable parameter count.
- Total training time (wall-clock duration of the `fit_model` stage only, excluding data preparation and the final test evaluation).
- Inference time.
- Metrics computed separately on test examples with and without detected negation.
- Deterministic selection of false-positive and false-negative examples.

Negation detection is a simple token-aware heuristic using markers such as:

- `not`
- `no`
- `never`
- `hardly`
- `without`
- contractions ending in `n't` (e.g. "isn't", "wasn't")

This heuristic does not capture every form of linguistic negation.

## Known Limitations

- The Bag-of-Words model ignores word order entirely.
- LSTM inputs are truncated to a configured maximum sequence length (default 200 tokens).
- The comparison does not include pretrained Transformer models.
- The simple negation detection heuristic is incomplete.
- Results depend on dataset quality and the specific data split.
- Computational timing depends on the hardware used.
- The models are designed for English binary sentiment classification only.

## Academic Use and AI Declaration

This repository is an individual university project. All submitted work must be understood and explainable by the student.

Provisional declaration:

> "Generative AI tools were used to support project planning, code drafting, debugging, and language revision. All generated suggestions were reviewed and adapted. The experimental execution, verification of outputs, interpretation of results, and final responsibility for the submitted work remain with the author."

This declaration should be updated before submission so it accurately reflects the actual use of AI.

## Project Status

*All reported experimental results were obtained after removing duplicate reviews, confirming zero overlap between the training, validation, and test splits, and successfully running the complete test suite.*

Final project checklist:

- [x] Install dependencies successfully
- [x] Run the complete unit test suite: 252 tests passed
- [x] Train the count-based BoW model
- [x] Train the binary BoW variation
- [x] Train the LSTM model
- [x] Verify saved checkpoints and metrics
- [x] Remove duplicate reviews and confirm zero split overlap
- [x] Perform negation and error analysis
- [x] Generate final figures and summary table
- [x] Add the verified dataset citation and licence
- [x] Write the final scientific report using only verified results