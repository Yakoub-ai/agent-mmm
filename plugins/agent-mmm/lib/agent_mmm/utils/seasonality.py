"""STL seasonality decomposition utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def seasonal_strength(series: pd.Series, period: int = 52) -> float:
    """Compute seasonal strength: ratio of seasonal variance to (seasonal + residual) variance.

    Returns value in [0, 1]. Higher = stronger seasonality.
    """
    if len(series) < 2 * period:
        return float("nan")
    series = series.fillna(series.median())
    try:
        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        var_seasonal = float(np.var(result.seasonal))
        var_resid = float(np.var(result.resid))
        denom = var_seasonal + var_resid
        return var_seasonal / denom if denom > 0 else 0.0
    except Exception:
        return float("nan")
