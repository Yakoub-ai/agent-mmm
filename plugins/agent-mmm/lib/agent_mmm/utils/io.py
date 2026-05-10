"""CSV/Parquet loading with schema validation for MMM input data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    elif ext in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)
