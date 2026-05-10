"""VIF computation for channel collinearity diagnosis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute Variance Inflation Factor for each column. Returns {column: vif_value}."""
    X = df[columns].dropna()
    if len(X) < len(columns) + 2:
        return {c: float("nan") for c in columns}
    X_arr = X.values.astype(float)
    results = {}
    for i, col in enumerate(columns):
        try:
            results[col] = variance_inflation_factor(X_arr, i)
        except Exception:
            results[col] = float("nan")
    return results
