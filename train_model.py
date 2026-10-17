"""Train and evaluate the fake-news text classification pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
EVALUATION_DIR = ROOT_DIR / "evaluation"
MODEL_PATH = MODEL_DIR / "fake_news_pipeline.joblib"
METRICS_PATH = EVALUATION_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = EVALUATION_DIR / "confusion_matrix.png"
RANDOM_STATE = 42
TEST_SIZE = 0.2
REQUIRED_COLUMNS = {"title", "text", "subject", "date"}
LABEL_NAMES = {0: "Fake", 1: "Real"}


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Read both labeled files and apply the documented data cleaning."""
    source_files = (("Fake.csv", 0), ("True.csv", 1))
    frames = []
    source_rows = {}
    missing_text_rows = {}

    for filename, label in source_files:
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. Place {filename} in the data folder."
            )

        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{filename} is missing required columns: {missing}")

        source_rows[filename] = len(frame)
        missing_text_rows[filename] = int(
            (
                frame["text"].isna()
                | frame["text"].fillna("").astype(str).str.strip().eq("")
            ).sum()
        )

        # Only title and article text are retained; subject and date are excluded.
        cleaned = frame[["title", "text"]].fillna("").astype(str)
        cleaned["title"] = cleaned["title"].str.strip()
        cleaned["text"] = cleaned["text"].str.strip()
        cleaned["text"] = cleaned.apply(
            lambda row: " ".join(
                value for value in (row["title"], row["text"]) if value
            ),
            axis=1,
        )
        cleaned["text"] = cleaned["text"].str.replace(r"\s+", " ", regex=True).str.strip()
        cleaned["label"] = label
        frames.append(cleaned[["text", "label"]])

    combined = pd.concat(frames, ignore_index=True)
    rows_before_empty_removal = len(combined)
    combined = combined.loc[combined["text"].ne("")].copy()

    # Deduplication ignores repeated whitespace and letter casing.
    combined["_dedupe_key"] = (
        combined["text"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    conflicting_duplicates = int(
        (combined.groupby("_dedupe_key")["label"].nunique() > 1).sum()
    )
    if conflicting_duplicates:
        raise ValueError(
            "The same normalized article text has conflicting labels in the two "
            "dataset files. Resolve those labels before training."
        )

    rows_before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset="_dedupe_key", keep="first")
    duplicate_rows_removed = rows_before_deduplication - len(combined)
    combined = combined.drop(columns="_dedupe_key")

    class_counts = {
        LABEL_NAMES[int(label)]: int(count)
        for label, count in combined["label"].value_counts().sort_index().items()
    }
    if combined["label"].nunique() != 2 or combined["label"].value_counts().min() < 2:
        raise ValueError("Both classes need at least two usable rows to train and test.")

    data_summary = {
        "source_files": source_rows,
        "input_columns": ["title", "text", "subject", "date"],
        "label_mapping": {"Fake.csv": 0, "True.csv": 1},
        "excluded_columns": ["subject", "date"],
        "missing_or_blank_article_text_by_file": missing_text_rows,
        "rows_before_empty_text_removal": rows_before_empty_removal,
        "empty_combined_text_rows_removed": rows_before_empty_removal
        - rows_before_deduplication,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_cleaning": len(combined),
        "class_distribution_after_cleaning": class_counts,
    }
    return combined, data_summary


def save_confusion_matrix(matrix) -> None:
    """Save a readable confusion-matrix image for the evaluation folder."""
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="Confusion Matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(figure)


def train() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset, data_summary = load_dataset()

    shuffled = dataset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    train_text, test_text, train_labels, test_labels = train_test_split(
        shuffled["text"],
        shuffled["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=shuffled["label"],
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )
    logging.info(
        "Training on %s rows and evaluating on %s rows...",
        len(train_text),
        len(test_text),
    )
    pipeline.fit(train_text, train_labels)

    predictions = pipeline.predict(test_text)
    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1])
    results = {
        "model": {
            "algorithm": "Logistic Regression",
            "features": "TF-IDF unigrams and bigrams",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(train_text)),
            "testing_rows": int(len(test_text)),
        },
        "dataset": data_summary,
        "metrics": {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "precision": float(
                precision_score(test_labels, predictions, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(test_labels, predictions, pos_label=1, zero_division=0)
            ),
            "f1_score": float(
                f1_score(test_labels, predictions, pos_label=1, zero_division=0)
            ),
            "positive_class": "Real",
            "confusion_matrix_labels": ["Fake", "Real"],
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                test_labels,
                predictions,
                labels=[0, 1],
                target_names=["Fake", "Real"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "interpretation": (
            "These are holdout-set results for this dataset, not a guarantee that "
            "predictions about new real-world articles are factual."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix)

    print(f"Saved model pipeline: {MODEL_PATH}")
    print(f"Saved evaluation report: {METRICS_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print("Classification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )
    print(json.dumps(results["metrics"], indent=2))


if __name__ == "__main__":
    train()
