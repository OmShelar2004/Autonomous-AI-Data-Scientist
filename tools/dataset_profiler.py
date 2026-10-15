"""Dataset-agnostic profiling utilities for tabular CSV data."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def _is_datetime_like(series: pd.Series, name: str) -> bool:
    """Conservatively identify date/time columns without treating IDs as dates."""
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series):
        return False
    name_hint = any(token in name.lower() for token in ("date", "time", "timestamp"))
    if not name_hint:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True, format="mixed")
    return float(parsed.notna().mean()) >= 0.8


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-friendly facts only; no dataset-specific assumptions."""
    datetime_columns = [
        column for column in df.columns if _is_datetime_like(df[column], str(column))
    ]
    numeric_columns = [
        column for column in df.columns if is_numeric_dtype(df[column]) and column not in datetime_columns
    ]
    categorical_columns = [
        column for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]
    missing = df.isna().sum()
    stats = df[numeric_columns].describe().round(6).to_dict() if numeric_columns else {}
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_values": {column: int(value) for column, value in missing.items()},
        "missing_percentages": {
            column: round(float(value) * 100 / len(df), 2) if len(df) else 0.0
            for column, value in missing.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
        "numeric_statistics": stats,
    }
