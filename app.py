"""Flask web application for predictions from the saved model pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "fake_news_pipeline.joblib"
METRICS_PATH = ROOT_DIR / "evaluation" / "metrics.json"
MAX_INPUT_LENGTH = 100_000
LABEL_NAMES = {0: "FAKE NEWS", 1: "REAL NEWS"}

app = Flask(__name__)
logger = logging.getLogger(__name__)


def load_saved_model():
    if not MODEL_PATH.is_file():
        logger.warning("Model artifact is missing; run python train_model.py first.")
        return None
    return joblib.load(MODEL_PATH)


model_pipeline = load_saved_model()
metrics = (
    json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if METRICS_PATH.is_file()
    else None
)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/info")
def model_info():
    return jsonify(
        {
            "model_available": model_pipeline is not None,
            "model": metrics.get("model") if metrics else None,
            "dataset": metrics.get("dataset") if metrics else None,
            "metrics": metrics.get("metrics") if metrics else None,
        }
    )


@app.post("/api/predict")
def predict():
    if model_pipeline is None:
        return (
            jsonify(
                {
                    "error": (
                        "The model has not been trained yet. Run "
                        "`python train_model.py` and restart the app."
                    )
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send the article as a JSON object."}), 400

    article = payload.get("text")
    if not isinstance(article, str):
        return jsonify({"error": "Article text must be a string."}), 400
    article = article.strip()
    if not article:
        return jsonify({"error": "Paste a headline or article before checking."}), 400
    if len(article) > MAX_INPUT_LENGTH:
        return (
            jsonify(
                {
                    "error": (
                        f"Input is too long. Please keep it under "
                        f"{MAX_INPUT_LENGTH:,} characters."
                    )
                }
            ),
            400,
        )

    predicted_label = int(model_pipeline.predict([article])[0])
    return jsonify(
        {
            "label": predicted_label,
            "prediction": LABEL_NAMES[predicted_label],
            "message": (
                "This is a model prediction based on patterns in the training "
                "dataset, not a definitive fact-check."
            ),
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=False)
