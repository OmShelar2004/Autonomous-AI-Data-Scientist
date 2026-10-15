"""Real train/test execution and serializable model bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from tools.feature_engineering import DateFeatureEngineer
from tools.model_candidates import get_model_candidates
from tools.preprocessing import build_preprocessor


@dataclass
class ModelBundle:
    pipeline: Any
    problem_type: str
    target_column: str
    label_encoder: Any = None

    def predict(self, X: pd.DataFrame):
        values = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(np.asarray(values, dtype=int)) if self.label_encoder else values

    def predict_proba(self, X: pd.DataFrame):
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        probabilities = self.pipeline.predict_proba(X)
        classes = self.label_encoder.inverse_transform(self.pipeline.classes_.astype(int)) if self.label_encoder else self.pipeline.classes_
        return probabilities, classes


def _make_pipeline(X_train: pd.DataFrame, estimator) -> Pipeline:
    engineer = DateFeatureEngineer().fit(X_train)
    engineered_train = engineer.transform(X_train)
    return Pipeline([
        ("feature_engineering", engineer),
        ("preprocessing", build_preprocessor(engineered_train)),
        ("model", estimator),
    ])


def _classification_metrics(y_true, y_pred, probabilities, classes) -> dict:
    average = "binary" if len(classes) == 2 else "weighted"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if probabilities is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1])) if len(classes) == 2 else float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def train_candidates(df: pd.DataFrame, target_column: str, problem_type: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    data = df.dropna(subset=[target_column]).copy()
    X = data.drop(columns=[target_column])
    y = data[target_column]
    if X.shape[1] == 0:
        raise ValueError("A dataset needs at least one feature column besides its target.")
    is_classification = problem_type != "regression"
    encoder = LabelEncoder() if is_classification else None
    y_fit = encoder.fit_transform(y) if encoder else y
    stratify = y_fit if is_classification and pd.Series(y_fit).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_fit, test_size=test_size, random_state=random_state, stratify=stratify)
    results = []
    for name, estimator in get_model_candidates(problem_type, random_state).items():
        pipeline = _make_pipeline(X_train, clone(estimator))
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if is_classification:
            probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            metrics = _classification_metrics(y_test, predictions, probabilities, encoder.classes_)
        else:
            mse = float(mean_squared_error(y_test, predictions))
            metrics = {"mae": float(mean_absolute_error(y_test, predictions)), "mse": mse, "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_test, predictions))}
        results.append({"model_name": name, "metrics": metrics, "pipeline": pipeline})
    return {"results": results, "X_train": X_train, "y_train": y_train, "X_all": X, "y_all": y_fit, "problem_type": problem_type, "target_column": target_column, "label_encoder": encoder}


def train_final_model(training: dict, selected_model: str) -> ModelBundle:
    match = next((result for result in training["results"] if result["model_name"] == selected_model), None)
    if match is None:
        raise ValueError(f"'{selected_model}' is not one of the evaluated models.")
    # Fit the selected architecture on all labeled data for the final artifact.
    X = training["X_all"]
    y = training["y_all"]
    final_pipeline = _make_pipeline(X, clone(match["pipeline"].named_steps["model"]))
    final_pipeline.fit(X, y)
    return ModelBundle(final_pipeline, training["problem_type"], training["target_column"], training["label_encoder"])
