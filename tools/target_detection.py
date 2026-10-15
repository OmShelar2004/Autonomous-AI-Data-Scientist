"""Target-column detection with explicit confidence rather than silent guessing."""

from __future__ import annotations

from typing import Any

import pandas as pd


COMMON_TARGET_NAMES = {
    "target", "label", "outcome", "y", "class", "response", "dependent",
    "price", "sale_price", "sales", "revenue", "profit", "score",
}


def _is_identifier(name: str, series: pd.Series) -> bool:
    lowered = name.lower().replace(" ", "_")
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return (lowered == "id" or lowered.endswith("_id") or "identifier" in lowered) and unique_ratio > 0.5


def detect_target(df: pd.DataFrame, user_selected: str | None = None) -> dict[str, Any]:
    if user_selected:
        if user_selected not in df.columns:
            raise ValueError(f"Target column '{user_selected}' is not in the dataset.")
        return {"target_column": user_selected, "confidence": "user_selected", "reason": "Selected by the user.", "requires_user_selection": False}

    candidates: list[tuple[str, int, str]] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty or _is_identifier(str(column), series):
            continue
        normalized = str(column).lower().replace(" ", "_")
        if normalized in COMMON_TARGET_NAMES or normalized.endswith(("_target", "_label", "_outcome")):
            candidates.append((column, 100, "Column name is a common target name."))
            continue
        cardinality = series.nunique()
        ratio = cardinality / max(len(series), 1)
        # A low-cardinality final column is a useful but deliberately lower-confidence signal.
        if column == df.columns[-1] and 2 <= cardinality <= min(20, max(2, len(series) // 5)) and ratio < 0.2:
            candidates.append((column, 55, "Final column has low cardinality and is not ID-like."))

    if not candidates:
        return {"target_column": None, "confidence": "none", "reason": "No unambiguous target was found; choose one explicitly.", "requires_user_selection": True}
    candidates.sort(key=lambda value: value[1], reverse=True)
    best = candidates[0]
    tied = [candidate for candidate in candidates if candidate[1] == best[1]]
    if best[1] < 80 or len(tied) > 1:
        return {"target_column": None, "confidence": "low", "reason": "Possible targets were found, but selection is ambiguous; choose one explicitly.", "candidates": [candidate[0] for candidate in candidates], "requires_user_selection": True}
    return {"target_column": best[0], "confidence": "high", "reason": best[2], "requires_user_selection": False}
