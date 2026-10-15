"""Rank genuine evaluation results and expose the two recommendations."""

from __future__ import annotations

import pandas as pd


def compare_models(results: list[dict], problem_type: str) -> tuple[pd.DataFrame, list[dict]]:
    rows = [{"model_name": item["model_name"], **item["metrics"]} for item in results]
    table = pd.DataFrame(rows)
    if problem_type == "regression":
        table = table.sort_values(["rmse", "r2"], ascending=[True, False], na_position="last")
    else:
        sort_columns = ["f1"] + (["roc_auc"] if "roc_auc" in table else [])
        table = table.sort_values(sort_columns, ascending=False, na_position="last")
    records = table.to_dict(orient="records")
    return table.reset_index(drop=True), records[:2]
