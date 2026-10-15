"""Small, safe feature transformations shared by training and prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tools.dataset_profiler import _is_datetime_like


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Replace detected datetime columns with stable calendar components."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.datetime_columns_ = [column for column in frame.columns if _is_datetime_like(frame[column], str(column))]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in getattr(self, "datetime_columns_", []):
            if column not in frame:
                frame[column] = pd.NaT
            parsed = pd.to_datetime(frame[column], errors="coerce", dayfirst=True, format="mixed")
            prefix = f"{column}__"
            frame[prefix + "year"] = parsed.dt.year
            frame[prefix + "month"] = parsed.dt.month
            frame[prefix + "day"] = parsed.dt.day
            frame[prefix + "day_of_week"] = parsed.dt.dayofweek
            frame = frame.drop(columns=[column])
        return frame


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for profiling/building a matching preprocessor."""
    return DateFeatureEngineer().fit_transform(X)
