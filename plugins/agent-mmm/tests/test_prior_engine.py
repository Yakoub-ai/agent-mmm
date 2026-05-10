"""Tests for prior_engine.py and channel_classifier.py."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.utils.channel_classifier import classify_channel, classify_channels
from agent_mmm.spec import MMMSpec, TargetUnit, TargetUnitKind, ChannelMeta
from agent_mmm.prior_engine import recommend_priors, compute_spend_shares


# --- Channel classifier tests ---

def test_classify_sem():
    assert classify_channel("spend_sem") == "sem"
    assert classify_channel("cost_paid_search") == "sem"
    assert classify_channel("google_ads_spend") == "sem"


def test_classify_social():
    assert classify_channel("spend_social") == "social"
    assert classify_channel("facebook_cost") == "social"


def test_classify_meta_over_social():
    assert classify_channel("spend_meta") == "meta"
    assert classify_channel("instagram_spend") == "meta"


def test_classify_tv():
    assert classify_channel("tv_spend") == "tv"
    assert classify_channel("television_cost") == "tv"


def test_classify_ooh():
    assert classify_channel("ooh_budget") == "ooh"
    assert classify_channel("outdoor_spend") == "ooh"


def test_classify_youtube():
    assert classify_channel("youtube_spend") == "youtube"


def test_classify_unknown_falls_back():
    result = classify_channel("channel_xyz_unknown")
    assert result == "digital_display"


def test_classify_channels_dict():
    result = classify_channels(["spend_sem", "spend_tv", "spend_ooh"])
    assert result["spend_sem"] == "sem"
    assert result["spend_tv"] == "tv"
    assert result["spend_ooh"] == "ooh"


# --- Spend shares test ---

def test_compute_spend_shares():
    df = pd.DataFrame({
        "spend_a": [100.0, 200.0],
        "spend_b": [300.0, 400.0],
    })
    shares = compute_spend_shares(df, ["spend_a", "spend_b"])
    assert abs(shares["spend_a"] - 300 / 1000) < 0.001
    assert abs(shares["spend_b"] - 700 / 1000) < 0.001
    assert abs(sum(shares.values()) - 1.0) < 0.001


def test_compute_spend_shares_all_zero():
    df = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    shares = compute_spend_shares(df, ["a", "b"])
    assert shares["a"] == pytest.approx(0.5)
    assert shares["b"] == pytest.approx(0.5)


# --- Full prior engine tests ---

DATA_DIR = Path(__file__).parent / "data"


def _make_spec(channels=None) -> MMMSpec:
    if channels is None:
        channels = ["spend_sem", "spend_social", "spend_tv"]
    return MMMSpec(
        mmm_type="greenfield",
        company_name="Test",
        industry="insurance",
        region="Sweden",
        data_path=str(DATA_DIR / "synthetic_weekly.csv"),
        target_unit=TargetUnit(kind=TargetUnitKind.acquisition, label="policy"),
        channels=[ChannelMeta(column=c) for c in channels],
    )


def test_prior_engine_runs(tmp_path):
    spec = _make_spec()
    result = recommend_priors(spec, base=str(tmp_path))
    assert result["n_channels"] == 3
    assert "model_config" in result
    assert "per_channel_audit" in result


def test_prior_engine_correct_channel_count(tmp_path):
    spec = _make_spec()
    result = recommend_priors(spec, base=str(tmp_path))
    cfg = result["model_config"]
    assert len(cfg["adstock_alpha"]["alpha"]) == 3
    assert len(cfg["saturation_lam"]["alpha"]) == 3
    assert len(cfg["saturation_beta"]["sigma"]) == 3


def test_prior_engine_spend_shares_sum_to_one(tmp_path):
    spec = _make_spec()
    result = recommend_priors(spec, base=str(tmp_path))
    shares = [ch["spend_share_pct"] for ch in result["per_channel_audit"]]
    assert abs(sum(shares) - 100.0) < 0.5


def test_prior_engine_sem_has_low_alpha_mu(tmp_path):
    spec = _make_spec(["spend_sem"])
    result = recommend_priors(spec, base=str(tmp_path))
    ch = result["per_channel_audit"][0]
    assert ch["channel_type"] == "sem"
    assert ch["alpha_mu"] < 0.25, "SEM should have low carryover"


def test_prior_engine_tv_has_high_alpha_mu(tmp_path):
    spec = _make_spec(["spend_tv"])
    result = recommend_priors(spec, base=str(tmp_path))
    ch = result["per_channel_audit"][0]
    assert ch["channel_type"] == "tv"
    assert ch["alpha_mu"] > 0.4, "TV should have high carryover"


def test_prior_engine_writes_artifacts(tmp_path):
    spec = _make_spec()
    recommend_priors(spec, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "priors" / "model_config.json").exists()
    assert (tmp_path / "mmm-workspace" / "priors" / "prior_audit_report.md").exists()


def test_prior_engine_all_five_channels(tmp_path):
    channels = ["spend_sem", "spend_social", "spend_display", "spend_tv", "spend_ooh"]
    spec = _make_spec(channels)
    result = recommend_priors(spec, base=str(tmp_path))
    assert result["n_channels"] == 5
    types = {ch["column"]: ch["channel_type"] for ch in result["per_channel_audit"]}
    assert types["spend_sem"] == "sem"
    assert types["spend_ooh"] == "ooh"
    assert types["spend_tv"] == "tv"
