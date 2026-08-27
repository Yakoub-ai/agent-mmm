"""Tests for diagnostics.py — uses synthetic InferenceData, no real MCMC."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.diagnostics import (
    run_diagnostics, check_convergence, check_overfit, check_prior_contraction,
    check_baseline, decompose,
    RHAT_FAIL, RHAT_TARGET, ESS_THRESHOLD, OVERFIT_GAP_THRESHOLD,
)

# Older names kept as aliases in the module.
_check_convergence = check_convergence
_check_overfit = check_overfit
_check_prior_pull = check_prior_contraction
RHAT_THRESHOLD = RHAT_FAIL


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

        # ArviZ 1.x replaced InferenceData with xarray's DataTree.
        return xr.DataTree.from_dict(
            {"posterior": posterior, "sample_stats": sample_stats}
        )
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
    assert any("gap" in e.lower() for e in findings["errors"])


# --- Decomposition tests -------------------------------------------------- #

def _idata_with_decomposition(n_dates=52, intercept=2.0, media=(1.0, 0.5), scale=100.0):
    """DataTree whose components have a known, checkable decomposition.

    ``intercept_contribution`` deliberately has no date dimension, which is how
    pymc-marketing actually stores a constant intercept — the case that used to
    understate the baseline by a factor of n_periods.
    """
    xr = pytest.importorskip("xarray")
    import numpy as np

    n_chains, n_draws, n_ch = 2, 50, len(media)
    posterior = xr.Dataset({
        "intercept_contribution": (
            ["chain", "draw"], np.full((n_chains, n_draws), intercept)
        ),
        "channel_contribution": (
            ["chain", "draw", "date", "channel"],
            np.broadcast_to(
                np.asarray(media), (n_chains, n_draws, n_dates, n_ch)
            ).copy(),
        ),
    })
    constant_data = xr.Dataset({"target_scale": ((), np.array(scale))})
    return xr.DataTree.from_dict({"posterior": posterior, "constant_data": constant_data})


def test_decompose_broadcasts_a_time_constant_intercept():
    idata = _idata_with_decomposition(n_dates=52, intercept=2.0, media=(1.0, 0.5), scale=100.0)
    dec = decompose(idata)
    # Intercept applies every period: 2.0 * 52 * scale.
    assert dec["totals"]["intercept"] == pytest.approx(2.0 * 52 * 100.0)
    assert dec["totals"]["media"] == pytest.approx(1.5 * 52 * 100.0)


def test_decompose_shares_sum_to_one():
    dec = decompose(_idata_with_decomposition())
    assert sum(v for v in dec["shares"].values() if v is not None) == pytest.approx(1.0)


def test_decompose_applies_the_target_scale():
    unscaled = decompose(_idata_with_decomposition(scale=1.0))
    scaled = decompose(_idata_with_decomposition(scale=100.0))
    assert scaled["grand_total"] == pytest.approx(unscaled["grand_total"] * 100)


def test_baseline_and_attribution_agree_on_the_same_denominator():
    from agent_mmm.diagnostics import check_attribution_plausibility

    idata = _idata_with_decomposition()
    dec = decompose(idata)
    baseline = check_baseline(idata, decomposition=dec)
    attribution = check_attribution_plausibility(idata, decomposition=dec)
    assert baseline["baseline_share"] + attribution["media_share"] == pytest.approx(1.0, abs=1e-6)


def test_negative_baseline_is_a_blocking_error(tmp_path):
    idata = _idata_with_decomposition(intercept=-5.0, media=(1.0, 0.5))
    findings = run_diagnostics("neg-baseline", idata=idata, base=str(tmp_path))
    assert any("negative" in e.lower() for e in findings["errors"])
    assert findings["summary"]["baseline_ok"] is False


def test_implied_roas_is_in_target_units_not_normalised_units():
    from agent_mmm.diagnostics import check_attribution_plausibility

    idata = _idata_with_decomposition(n_dates=10, media=(1.0, 0.5), scale=100.0)
    # channel_0 contributes 1.0 * 10 * 100 = 1000 in target units.
    result = check_attribution_plausibility(idata, spend_totals={"0": 500.0, "1": 500.0})
    assert result["implied_roas"]["0"] == pytest.approx(2.0)
