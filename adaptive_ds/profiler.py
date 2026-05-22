from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .schemas import DatasetProfile


def _column_stats(series: pd.Series) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing_ratio": float(series.isna().mean()),
        "cardinality": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if clean.empty:
            stats["mean"] = None
            stats["std"] = None
            stats["skew"] = None
        else:
            stats["mean"] = float(clean.mean())
            stats["std"] = float(clean.std(ddof=0))
            stats["skew"] = float(clean.skew()) if len(clean) > 2 else 0.0
            stats["min"] = float(clean.min())
            stats["max"] = float(clean.max())
    return stats


def build_dataset_profile(df: pd.DataFrame, target_column: str | None) -> DatasetProfile:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns.tolist() if col not in numeric_cols]

    column_stats: dict[str, Mapping[str, Any]] = {
        col: _column_stats(df[col]) for col in df.columns
    }

    target_summary: dict[str, Any] = {"target_column": target_column, "exists": False}
    if target_column and target_column in df.columns:
        target = df[target_column]
        target_summary = {
            "target_column": target_column,
            "exists": True,
            "dtype": str(target.dtype),
            "missing_ratio": float(target.isna().mean()),
            "n_unique": int(target.nunique(dropna=True)),
        }

    return DatasetProfile(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        missing_ratio=float(df.isna().mean().mean()),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        column_stats=dict(column_stats),
        target_summary=target_summary,
    )
