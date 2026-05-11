"""Tests for all stakeholder report generators."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.spec import MMMSpec, TargetUnit, TargetUnitKind, ChannelMeta
from agent_mmm.reports import generate_cmo_report, generate_cfo_report, generate_mops_report, generate_ds_report


def _make_spec(kind: TargetUnitKind = TargetUnitKind.acquisition, value_per_unit=None) -> MMMSpec:
    return MMMSpec(
        mmm_type="greenfield",
        company_name="Acme Insurance",
        industry="insurance",
        region="Sweden",
        data_path="/tmp/data.csv",
        target_unit=TargetUnit(
            kind=kind,
            label="policy" if kind == TargetUnitKind.acquisition else "SEK",
            currency_code="SEK" if kind == TargetUnitKind.monetary else None,
            value_per_unit=value_per_unit,
        ),
        channels=[
            ChannelMeta(column="spend_sem", label="Paid Search"),
            ChannelMeta(column="spend_tv", label="TV"),
        ],
    )


METRICS = {"r2_insample": 0.84, "mape_insample": 0.12, "run_id": "test-001"}
DIAGNOSTICS = {
    "summary": {"tier": "PASS", "rhat_ok": True, "ess_ok": True, "divergences_ok": True, "overfit_ok": True},
    "checks": {
        "convergence": {"available": True, "max_rhat": 1.02, "min_ess_bulk": 520, "n_divergences": 0},
        "overfit": {"available": True, "in_sample_r2": 0.84, "cv_r2": 0.77, "gap": 0.07, "overfit": False},
        "attribution_plausibility": {"available": False},
    },
    "errors": [], "warnings": [],
}


# --- CMO report tests ---

def test_cmo_report_generates(tmp_path):
    spec = _make_spec()
    report = generate_cmo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "Acme Insurance" in report
    assert "policy" in report
    assert "Top 3 Recommendations" in report


def test_cmo_report_writes_file(tmp_path):
    spec = _make_spec()
    generate_cmo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "reports" / "cmo.md").exists()


def test_cmo_report_with_contributions(tmp_path):
    spec = _make_spec()
    contribs = {"spend_sem": 45.0, "spend_tv": 30.0}
    report = generate_cmo_report(spec, "test-001", METRICS, DIAGNOSTICS, contribs, base=str(tmp_path))
    assert "spend_sem" in report or "Paid Search" in report


# --- CFO report tests ---

def test_cfo_report_acquisition_uses_cpa(tmp_path):
    spec = _make_spec(TargetUnitKind.acquisition)
    report = generate_cfo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "CPA" in report
    assert "policy" in report


def test_cfo_report_monetary_uses_roas(tmp_path):
    spec = _make_spec(TargetUnitKind.monetary)
    report = generate_cfo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "ROAS" in report


def test_cfo_report_acquisition_with_value_shows_both(tmp_path):
    spec = _make_spec(TargetUnitKind.acquisition, value_per_unit=2500.0)
    report = generate_cfo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "2,500" in report or "2500" in report
    assert "CPA" in report


def test_cfo_report_writes_file(tmp_path):
    spec = _make_spec()
    generate_cfo_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "reports" / "cfo.md").exists()


# --- MOps report tests ---

def test_mops_report_generates(tmp_path):
    spec = _make_spec()
    report = generate_mops_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "Channel Summary" in report
    assert "spend_sem" in report


def test_mops_report_with_spend_data(tmp_path):
    spec = _make_spec()
    spend = {"spend_sem": 50000, "spend_tv": 200000}
    contribs = {"spend_sem": 35.0, "spend_tv": 40.0}
    report = generate_mops_report(spec, "test-001", METRICS, DIAGNOSTICS, spend, contribs, base=str(tmp_path))
    assert "50,000" in report or "50000" in report


def test_mops_report_writes_file(tmp_path):
    spec = _make_spec()
    generate_mops_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "reports" / "mops.md").exists()


# --- DS report tests ---

def test_ds_report_generates(tmp_path):
    spec = _make_spec()
    report = generate_ds_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "Model Specification" in report
    assert "Reproducibility" in report
    assert "test-001" in report


def test_ds_report_shows_diagnostics(tmp_path):
    spec = _make_spec()
    report = generate_ds_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert "PASS" in report
    assert "1.02" in report  # max_rhat


def test_ds_report_writes_file(tmp_path):
    spec = _make_spec()
    generate_ds_report(spec, "test-001", METRICS, DIAGNOSTICS, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "reports" / "ds.md").exists()
