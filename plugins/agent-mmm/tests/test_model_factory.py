"""Tests for model_factory.py — model construction (no MCMC)."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.spec import MMMSpec, TargetUnit, TargetUnitKind, ChannelMeta, ControlMeta
from agent_mmm.model_factory import (
    build_model_config_priors, prepare_data, _dict_to_prior, _load_model_config_from_file
)

DATA_DIR = Path(__file__).parent / "data"


def _make_spec(channels=None, controls=None) -> MMMSpec:
    return MMMSpec(
        mmm_type="greenfield",
        company_name="Test",
        industry="retail",
        region="UK",
        data_path=str(DATA_DIR / "synthetic_weekly.csv"),
        target_unit=TargetUnit(kind=TargetUnitKind.monetary, label="GBP", currency_code="GBP"),
        channels=[ChannelMeta(column=c) for c in (channels or ["spend_sem", "spend_social"])],
        controls=[ControlMeta(column=c) for c in (controls or [])],
    )


def test_prepare_data_returns_correct_shapes():
    spec = _make_spec()
    X, y = prepare_data(spec)
    assert "date" in X.columns
    assert "spend_sem" in X.columns
    assert len(X) == 104
    assert len(y) == 104
    assert y.name == "y"  # default target_column


def test_prepare_data_y_series_name_matches_target_column():
    spec = _make_spec()
    spec.target_column = "y"
    _, y = prepare_data(spec)
    assert y.name == "y", "y Series name must match target_column for pymc-marketing"


def test_dict_to_prior_normal():
    pytest.importorskip("pymc_extras")
    prior = _dict_to_prior({"distribution": "Normal", "mu": 0.5, "sigma": 0.5})
    from pymc_extras.prior import Prior
    assert isinstance(prior, Prior)


def test_dict_to_prior_beta_with_arrays():
    pytest.importorskip("pymc_extras")
    prior = _dict_to_prior({
        "distribution": "Beta",
        "alpha": [2.0, 3.0],
        "beta": [4.0, 5.0],
        "dims": "channel",
    })
    from pymc_extras.prior import Prior
    assert isinstance(prior, Prior)


def test_dict_to_prior_nested_student_t():
    pytest.importorskip("pymc_extras")
    prior = _dict_to_prior({
        "distribution": "StudentT",
        "nu": 5,
        "sigma": {"distribution": "HalfNormal", "sigma": 0.5},
    })
    from pymc_extras.prior import Prior
    assert isinstance(prior, Prior)


def test_build_model_config_priors_skips_metadata():
    pytest.importorskip("pymc_extras")
    cfg = {
        "adstock_alpha": {"distribution": "Beta", "alpha": [2.0], "beta": [4.0], "dims": "channel"},
        "_metadata": {"channel_order": ["spend_sem"]},
        "_channel_order": ["spend_sem"],
    }
    result = build_model_config_priors(cfg)
    assert "adstock_alpha" in result
    assert "_metadata" not in result
    assert "_channel_order" not in result


def test_build_mmm_constructs_without_error():
    """Integration test: build MMM from spec + minimal model_config."""
    try:
        from pymc_marketing.mmm.multidimensional import MMM
    except ImportError:
        pytest.skip("pymc-marketing not installed")

    spec = _make_spec()
    # Minimal model_config dict
    cfg = {
        "adstock_alpha": {"distribution": "Beta", "alpha": [2.0, 2.0], "beta": [4.0, 4.0], "dims": "channel"},
        "saturation_lam": {"distribution": "Gamma", "alpha": [4.0, 4.0], "beta": [1.0, 1.0], "dims": "channel"},
        "saturation_beta": {"distribution": "HalfNormal", "sigma": [0.5, 0.5], "dims": "channel"},
        "intercept": {"distribution": "Normal", "mu": 0.5, "sigma": 0.5},
        "gamma_fourier": {"distribution": "Laplace", "mu": 0.0, "b": 0.3, "dims": "fourier_mode"},
        "likelihood": {"distribution": "StudentT", "nu": 5,
                       "sigma": {"distribution": "HalfNormal", "sigma": 0.5}},
    }
    from agent_mmm.model_factory import build_mmm
    model = build_mmm(spec, model_config_dict=cfg)
    assert model is not None
