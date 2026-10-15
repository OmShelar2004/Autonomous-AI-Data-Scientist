"""Lightweight candidate estimators; optional XGBoost support."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def get_model_candidates(problem_type: str, random_state: int = 42) -> dict:
    if problem_type in {"binary_classification", "multiclass_classification"}:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        }
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=random_state, n_jobs=1, eval_metric="logloss")
        except ImportError:
            pass
        return models
    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        }
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=random_state, n_jobs=1, objective="reg:squarederror")
        except ImportError:
            pass
        return models
    raise ValueError(f"Unsupported problem type: {problem_type}")
