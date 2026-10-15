"""Prediction helpers for saved execution artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def predict_rows(model_path: str | Path | Any, rows: pd.DataFrame | dict[str, Any]) -> list[dict]:
    bundle = load_model(model_path) if isinstance(model_path, (str, Path)) else model_path
    frame = pd.DataFrame([rows]) if isinstance(rows, dict) else rows.copy()
    frame = frame.drop(columns=[bundle.target_column], errors="ignore")
    predictions = bundle.predict(frame)
    output = [{"prediction": value.item() if hasattr(value, "item") else value} for value in predictions]
    probabilities = bundle.predict_proba(frame)
    if probabilities is not None:
        values, classes = probabilities
        for item, probability in zip(output, values):
            item["probabilities"] = {str(label): float(score) for label, score in zip(classes, probability)}
    return output


def predict_csv(model_path: str | Path, csv_path: str | Path) -> list[dict]:
    return predict_rows(model_path, pd.read_csv(csv_path))
