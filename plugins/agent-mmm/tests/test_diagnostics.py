"""Tests for diagnostics.py — uses synthetic InferenceData, no real MCMC."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.diagnostics import (
    run_diagnostics, _check_convergence, _check_overfit, _check_prior_pull,
    RHAT_THRESHOLD, ESS_THRESHOLD, OVERFIT_GAP_THRESHOLD,
)


def _make_fake_idata(rhat_val=1.01, ess_val=600, n_divergences=0):
    """Build a synthetic ArviZ InferenceData for testing."""
    try:
        import arviz as az
        import xarray as xr
        import numpy as np

        n_chains, n_draws = 2, 200
        rng = np.random.default_rng(42)

        # Posterior with two parameters
        posterior = xr.Dataset({
            "adstock_alpha": (["chain", "draw", "channel"], rng.beta(2, 4, (n_chains, n_draws, 2))),
            "saturation_lam": (["chain", "draw", "channel"], rng.gamma(4, 1, (n_chains, n_draws, 2))),
        })

        # Sample stats
        diverging = np.zeros((n_chains, n_draws), dtype=bool)
        if n_divergences > 0:
            diverging[0, :n_divergences] = True
        sample_stats = xr.Dataset({
            "diverging": (["chain", "draw"], diverging),
        })

        return az.InferenceData(posterior=posterior, sample_stats=sample_stats)
    except ImportError:
        return None


# --- Convergence tests ---

def test_convergence_no_idata():
    result = _check_convergence(None)
    assert result["available"] is False


def test_convergence_with_fake_idata():
    idata = _make_fake_idata()
    if idata is None:
        pytest.skip("arviz not installed")
    result = _check_convergence(idata)
    assert result["available"] is True
    assert "max_rhat" in result
    assert "min_ess_bulk" in result
    assert "n_divergences" in result


def test_convergence_detects_divergences():
    idata = _make_fake_idata(n_divergences=5)
    if idata is None:
        pytest.skip("arviz not installed")
    result = _check_convergence(idata)
    assert result.get("n_divergences", 0) >= 5


# --- Overfit tests ---

def test_overfit_no_cv():
    result = _check_overfit(0.85, None)
    assert result["available"] is False
    assert result["gap"] is None


def test_overfit_pass():
    result = _check_overfit(0.85, {"r2_cv": 0.78})
    assert result["available"] is True
    assert result["gap"] == pytest.approx(0.07)
    assert result["overfit"] is False


def test_overfit_fail():
    result = _check_overfit(0.90, {"r2_cv": 0.60})
    assert result["gap"] > OVERFIT_GAP_THRESHOLD
    assert result["overfit"] is True


def test_overfit_threshold_boundary():
    result = _check_overfit(0.80, {"r2_cv": 0.60})
    assert result["gap"] == pytest.approx(0.20)
    assert result["overfit"] is False  # 0.20 is exactly at threshold, not over


# --- Prior pull tests ---

def test_prior_pull_no_idata():
    result = _check_prior_pull(None)
    assert result == {}


def test_prior_pull_with_idata():
    idata = _make_fake_idata()
    if idata is None:
        pytest.skip("arviz not installed")
    result = _check_prior_pull(idata)
    assert isinstance(result, dict)
    for k, v in result.items():
        if not k.startswith("_"):
            assert "posterior_mean" in v
            assert "possible_pull" in v


# --- Full run_diagnostics tests ---

def test_run_diagnostics_no_idata_no_metrics(tmp_path):
    findings = run_diagnostics(
        run_id="test-run-001",
        idata_path=None,
        metrics_path=None,
        base=str(tmp_path),
    )
    assert "checks" in findings
    assert "summary" in findings
    assert (tmp_path / "mmm-workspace" / "runs" / "test-run-001" / "diagnostics.json").exists()


def test_run_diagnostics_with_cv_pass(tmp_path):
    findings = run_diagnostics(
        run_id="test-run-pass",
        idata_path=None,
        metrics_path=None,
        cv_metrics={"r2_cv": None},
        base=str(tmp_path),
    )
    assert findings["summary"]["tier"] in ("PASS", "WARN", "FAIL")


def test_run_diagnostics_overfit_detected(tmp_path):
    # Write a fake metrics.json
    import json
    run_dir = tmp_path / "mmm-workspace" / "runs" / "test-overfit"
    run_dir.mkdir(parents=True)
    metrics_file = run_dir / "metrics.json"
    json.dump({"r2_insample": 0.92, "run_id": "test-overfit"}, open(metrics_file, "w"))

    findings = run_diagnostics(
        run_id="test-overfit",
        metrics_path=str(metrics_file),
        cv_metrics={"r2_cv": 0.60},
        base=str(tmp_path),
    )
    # Should detect overfit gap
    ov = findings["checks"]["overfit"]
    assert ov["gap"] == pytest.approx(0.32)
    assert ov["overfit"] is True
    assert any("Overfit" in e for e in findings["errors"])
