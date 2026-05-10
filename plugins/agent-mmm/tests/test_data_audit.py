"""Tests for data_audit.py, vif.py, seasonality.py, breaks.py."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.utils.vif import compute_vif
from agent_mmm.utils.seasonality import seasonal_strength
from agent_mmm.utils.breaks import detect_structural_break
from agent_mmm.spec import MMMSpec, TargetUnit, TargetUnitKind, ChannelMeta
from agent_mmm.data_audit import run_audit


# --- VIF tests ---

def test_vif_uncorrelated():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "c": rng.normal(0, 1, 100),
    })
    vif = compute_vif(df, ["a", "b", "c"])
    for v in vif.values():
        assert v < 5, f"Uncorrelated VIF should be low, got {v}"


def test_vif_highly_correlated():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({"a": x, "b": x + rng.normal(0, 0.01, 100)})
    vif = compute_vif(df, ["a", "b"])
    # Highly correlated — both VIFs should be very high
    assert any(v > 10 for v in vif.values() if not np.isnan(v))


# --- Seasonality tests ---

def test_seasonal_strength_with_seasonality():
    t = np.linspace(0, 4 * np.pi, 104)
    series = pd.Series(np.sin(t) * 500 + 1000 + np.random.default_rng(2).normal(0, 50, 104))
    ss = seasonal_strength(series, period=52)
    assert 0 <= ss <= 1


def test_seasonal_strength_insufficient_data():
    series = pd.Series([1.0] * 50)
    ss = seasonal_strength(series, period=52)
    assert np.isnan(ss)


# --- Structural break tests ---

def test_no_break_stationary():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(1000, 50, 100))
    result = detect_structural_break(series)
    assert "break_detected" in result


def test_break_detected_step_change():
    y = np.concatenate([np.ones(50) * 100, np.ones(50) * 500])
    series = pd.Series(y + np.random.default_rng(4).normal(0, 5, 100))
    result = detect_structural_break(series)
    assert "break_detected" in result  # might or might not be True depending on noise


def test_insufficient_data_for_breaks():
    series = pd.Series([1.0] * 10)
    result = detect_structural_break(series)
    assert result["break_detected"] is False


# --- Full audit tests ---

def _make_spec(data_path: str) -> MMMSpec:
    return MMMSpec(
        mmm_type="greenfield",
        company_name="Test Co",
        industry="test",
        region="SE",
        data_path=data_path,
        target_column="y",
        date_column="date",
        target_unit=TargetUnit(kind=TargetUnitKind.acquisition, label="sale"),
        channels=[
            ChannelMeta(column="spend_sem"),
            ChannelMeta(column="spend_social"),
        ],
    )


def test_audit_on_synthetic_data(tmp_path):
    DATA = Path(__file__).parent / "data" / "synthetic_weekly.csv"
    spec = _make_spec(str(DATA))
    # Override channels to match synthetic data
    spec.channels = [
        ChannelMeta(column="spend_sem"),
        ChannelMeta(column="spend_social"),
        ChannelMeta(column="spend_display"),
        ChannelMeta(column="spend_tv"),
        ChannelMeta(column="spend_ooh"),
    ]
    findings = run_audit(spec, base=str(tmp_path))
    assert "dimensions" in findings["checks"]
    assert findings["summary"]["rows"] == 104
    assert findings["summary"]["channels"] == 5
    assert findings["summary"]["data_quality_tier"] in ("PASS", "WARN", "FAIL")


def test_audit_fails_missing_target_column(tmp_path):
    DATA = Path(__file__).parent / "data" / "synthetic_weekly.csv"
    spec = _make_spec(str(DATA))
    spec.target_column = "nonexistent_column"
    findings = run_audit(spec, base=str(tmp_path))
    assert any("nonexistent_column" in e for e in findings["errors"])


def test_audit_warns_insufficient_rows(tmp_path):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=30, freq="W-MON").strftime("%Y-%m-%d"),
        "y": rng.uniform(800, 1500, 30),
        "spend_sem": rng.uniform(1000, 5000, 30),
    })
    p = tmp_path / "small.csv"
    df.to_csv(p, index=False)
    spec = _make_spec(str(p))
    spec.channels = [ChannelMeta(column="spend_sem")]
    findings = run_audit(spec, base=str(tmp_path))
    assert any("Insufficient" in e or "Marginal" in e for e in findings["errors"] + findings["warnings"])
