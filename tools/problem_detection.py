"""Infer supervised problem type from target values, not dtype alone."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def detect_problem_type(df: pd.DataFrame, target_column: str) -> dict:
    target = df[target_column].dropna()
    if target.empty:
        raise ValueError("The target column has no non-missing values.")
    classes = int(target.nunique())
    if is_bool_dtype(target) or not is_numeric_dtype(target):
        problem_type = "binary_classification" if classes == 2 else "multiclass_classification"
    else:
        # Few repeated integer-like numeric values are labels; continuous/high-cardinality values are regression.
        integer_like = ((target % 1) == 0).all()
        classification_limit = min(20, max(2, int(len(target) * 0.05)))
        if classes == 2 or (integer_like and classes <= classification_limit):
            problem_type = "binary_classification" if classes == 2 else "multiclass_classification"
        else:
            problem_type = "regression"
    result = {"problem_type": problem_type, "target_column": target_column}
    if problem_type != "regression":
        result["number_of_classes"] = classes
        result["class_values"] = target.unique().tolist()
    return result
