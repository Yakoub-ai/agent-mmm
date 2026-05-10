"""Tests for controls_engine.py."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.spec import MMMSpec, TargetUnit, TargetUnitKind, ChannelMeta, ControlMeta
from agent_mmm.controls_engine import recommend_controls


def _make_spec(industry: str = "insurance", region: str = "Sweden") -> MMMSpec:
    return MMMSpec(
        mmm_type="greenfield",
        company_name="Test Co",
        industry=industry,
        region=region,
        data_path="/tmp/data.csv",
        target_unit=TargetUnit(kind=TargetUnitKind.acquisition, label="policy"),
        channels=[ChannelMeta(column="spend_sem")],
    )


def test_recommends_calendar_controls(tmp_path):
    spec = _make_spec()
    result = recommend_controls(spec, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "is_christmas_week" in names or "is_q4" in names


def test_always_includes_covid(tmp_path):
    spec = _make_spec()
    result = recommend_controls(spec, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "is_covid_lockdown" in names


def test_insurance_specific_controls(tmp_path):
    spec = _make_spec(industry="insurance")
    result = recommend_controls(spec, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "renewal_cycle_indicator" in names


def test_automotive_specific_controls(tmp_path):
    spec = _make_spec(industry="automotive")
    result = recommend_controls(spec, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "new_model_launch_week" in names


def test_structural_break_adds_dummy(tmp_path):
    spec = _make_spec()
    audit = {"checks": {"structural_break": {"break_detected": True, "p_value": 0.02}}}
    result = recommend_controls(spec, audit_findings=audit, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "structural_break_dummy" in names


def test_no_structural_break_no_dummy(tmp_path):
    spec = _make_spec()
    audit = {"checks": {"structural_break": {"break_detected": False}}}
    result = recommend_controls(spec, audit_findings=audit, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert "structural_break_dummy" not in names


def test_no_duplicates(tmp_path):
    spec = _make_spec()
    result = recommend_controls(spec, base=str(tmp_path))
    names = [r["name"] for r in result["recommendations"]]
    assert len(names) == len(set(names)), "Duplicate control names found"


def test_writes_artifacts(tmp_path):
    spec = _make_spec()
    recommend_controls(spec, base=str(tmp_path))
    assert (tmp_path / "mmm-workspace" / "controls" / "recommended.json").exists()
    assert (tmp_path / "mmm-workspace" / "controls" / "recommendations_report.md").exists()


def test_high_importance_first(tmp_path):
    spec = _make_spec()
    result = recommend_controls(spec, base=str(tmp_path))
    importances = [r.get("importance", "low") for r in result["recommendations"]]
    order = {"high": 0, "medium": 1, "low": 2}
    scores = [order.get(i, 2) for i in importances]
    assert scores == sorted(scores), "Recommendations not sorted by importance"
