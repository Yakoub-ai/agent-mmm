"""Structural break detection using CUSUM on OLS residuals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid


def detect_structural_break(series: pd.Series) -> dict:
    """Run CUSUM-OLS test for structural breaks in a time series.

    Returns dict with 'break_detected' (bool), 'cusum_stat' (float), 'p_value' (float | None).
    """
    y = series.dropna().values.astype(float)
    if len(y) < 20:
        return {"break_detected": False, "cusum_stat": None, "p_value": None, "note": "insufficient data"}
    t = np.arange(len(y)).reshape(-1, 1)
    X = np.column_stack([np.ones(len(y)), t])
    try:
        model = OLS(y, X).fit()
        stat, p_value = breaks_cusumolsresid(model.resid)
        break_detected = bool(p_value < 0.05) if p_value is not None else False
        return {
            "break_detected": break_detected,
            "cusum_stat": float(stat),
            "p_value": float(p_value) if p_value is not None else None,
        }
    except Exception as e:
        return {"break_detected": False, "cusum_stat": None, "p_value": None, "note": str(e)}
