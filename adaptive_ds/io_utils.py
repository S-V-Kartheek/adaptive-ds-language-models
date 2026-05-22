from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV or Parquet.")
