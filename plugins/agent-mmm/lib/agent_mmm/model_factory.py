"""Builds a pymc-marketing MMM from a MMMSpec + model_config dict."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_mmm.spec import MMMSpec, MMMType
from agent_mmm.utils.io import load_data, parse_dates


def _load_model_config_from_file(path: str | Path) -> dict:
    """Load model_config.json written by prior_engine."""
    with open(path) as f:
        return json.load(f)


def _dict_to_prior(d: dict) -> Any:
    """Convert a plain dict (from model_config.json) to a pymc_extras Prior object.

    Handles nested priors (e.g. likelihood.sigma as sub-prior).
    """
    from pymc_extras.prior import Prior
    dist = d["distribution"]
    kwargs = {k: v for k, v in d.items() if k not in ("distribution", "dims")}
    dims = d.get("dims")

    # Convert list values to np.array for vector params
    for key, val in kwargs.items():
        if isinstance(val, list):
            kwargs[key] = np.array(val)
        elif isinstance(val, dict) and "distribution" in val:
            kwargs[key] = _dict_to_prior(val)

    if dims:
        return Prior(dist, dims=dims, **kwargs)
    return Prior(dist, **kwargs)


def build_model_config_priors(model_config_dict: dict) -> dict:
    """Convert a model_config JSON dict to pymc_extras Prior objects for MMM constructor.

    Skips metadata keys (starting with '_').
    """
    result = {}
    skip_keys = {k for k in model_config_dict if k.startswith("_")}
    for key, value in model_config_dict.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict) and "distribution" in value:
            result[key] = _dict_to_prior(value)
        # else: skip non-prior metadata fields
    return result


def prepare_data(
    spec: MMMSpec,
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare X (features) and y (target) from spec.

    Returns (X, y_series) where:
    - X contains date + channel + control columns
    - y_series is a named Series with name == spec.target_column
    """
    if df is None:
        df = load_data(spec.data_path)
        df = parse_dates(df, spec.date_column)

    channel_cols = spec.channel_columns()
    control_cols = spec.control_columns()
    all_feature_cols = [spec.date_column] + channel_cols + control_cols

    X = df[[c for c in all_feature_cols if c in df.columns]].copy()
    y = df[spec.target_column].copy()
    y_series = pd.Series(y.values, name=spec.target_column)

    return X, y_series


def build_mmm(
    spec: MMMSpec,
    model_config_dict: dict | None = None,
    model_config_path: str | Path | None = None,
) -> Any:
    """Build a pymc-marketing MMM object from spec + model_config.

    Exactly one of model_config_dict or model_config_path must be provided.
    For brownfield, still builds a fresh MMM (InferenceData warm-start applied at fit time).

    Returns: MMM instance (not yet fitted).
    """
    from pymc_marketing.mmm.multidimensional import MMM
    from pymc_marketing.mmm import GeometricAdstock, DelayedAdstock, LogisticSaturation

    if model_config_dict is None and model_config_path is not None:
        raw = _load_model_config_from_file(model_config_path)
        model_config_dict = raw.get("model_config", raw)
    if model_config_dict is None:
        raise ValueError("Provide either model_config_dict or model_config_path")

    model_config = build_model_config_priors(model_config_dict)

    channel_cols = spec.channel_columns()
    control_cols = spec.control_columns()

    # Adstock selection heuristic: use Delayed if any offline channel
    from agent_mmm.utils.channel_classifier import classify_channel
    offline_types = {"tv", "ooh", "print", "audio"}
    has_offline = any(classify_channel(c) in offline_types for c in channel_cols)
    if has_offline:
        adstock = DelayedAdstock(l_max=12)
    else:
        adstock = GeometricAdstock(l_max=8)

    kwargs: dict = dict(
        date_column=spec.date_column,
        channel_columns=channel_cols,
        target_column=spec.target_column,
        adstock=adstock,
        saturation=LogisticSaturation(),
        yearly_seasonality=spec.seasonality.yearly_fourier_modes,
        model_config=model_config,
        adstock_first=True,
    )

    if control_cols:
        kwargs["control_columns"] = control_cols

    # Multi-geo dims (placeholder — multidimensional API uses coords at fit time)
    if spec.geo.is_panel and spec.geo.geo_column:
        kwargs["dims"] = spec.geo.geo_column

    return MMM(**kwargs)
