"""Model construction — a thin façade over the backend layer.

Kept as a stable entry point (`build_mmm`, `prepare_data`) while the actual
translation lives in :mod:`agent_mmm.backends`, so a spec targeting Meridian or
Robyn goes through exactly the same call path as a pymc-marketing one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent_mmm.backends import get_backend
from agent_mmm.backends.base import TranslationResult
from agent_mmm.backends.pymc_backend import (  # re-exported for backwards compatibility
    _dict_to_prior,
    build_model_config_priors,
    prepare_data,
)
from agent_mmm.spec import MMMSpec

__all__ = [
    "build_mmm",
    "compile_spec",
    "prepare_data",
    "build_model_config_priors",
    "_dict_to_prior",
    "load_priors",
]


def load_priors(path: str | Path) -> dict:
    """Load the payload written by :func:`agent_mmm.prior_engine.recommend_priors`."""
    with open(path) as f:
        return json.load(f)


def compile_spec(
    spec: MMMSpec,
    priors: dict | None = None,
    priors_path: str | Path | None = None,
    build: bool = True,
    df: pd.DataFrame | None = None,
) -> TranslationResult:
    """Translate a spec into its target framework.

    For pymc-marketing this returns a built (unfitted) model alongside the code;
    for Meridian and Robyn it returns runnable code plus the data contract the
    framework requires. In every case ``result.unsupported`` lists spec features
    the target framework cannot express — read it before running anything.
    """
    if priors is None and priors_path is not None:
        priors = load_priors(priors_path)
    backend = get_backend(spec.framework)
    return backend.translate(spec, priors=priors, build=build, df=df)


def build_mmm(
    spec: MMMSpec,
    model_config_dict: dict | None = None,
    model_config_path: str | Path | None = None,
    priors: dict | None = None,
    df: pd.DataFrame | None = None,
) -> Any:
    """Build an unfitted pymc-marketing ``MMM`` from a spec.

    ``model_config_dict`` / ``model_config_path`` are accepted for backwards
    compatibility and are treated as the ``model_config`` section of a prior
    payload.
    """
    if priors is None:
        if model_config_dict is not None:
            priors = {"model_config": model_config_dict}
        elif model_config_path is not None:
            raw = load_priors(model_config_path)
            priors = raw if "model_config" in raw else {"model_config": raw}

    from agent_mmm.backends.pymc_backend import PyMCMarketingBackend

    return PyMCMarketingBackend().build(spec, priors=priors, df=df)
