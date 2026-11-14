"""Tests for data_advisor.py: audit findings -> prioritised, evidenced tips."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.data_advisor import AREAS, advise, render_tips, run_advisor
from agent_mmm.data_audit import run_audit
from agent_mmm.spec import ChannelMeta, ControlMeta, Experiment, MMMSpec, TargetUnit, TargetUnitKind

DATA = Path(__file__).parent / "data"


def _spec(**kw):
    base = dict(
        mmm_type="greenfield", company_name="T", industry="retail", region="SE",
        data_path=str(DATA / "synthetic_weekly.csv"),
        target_unit=TargetUnit(kind=TargetUnitKind.acquisition, label="sale"),
        channels=[ChannelMeta(column=c) for c in ["spend_sem", "spend_social", "spend_tv"]],
    )
    base.update(kw)
    return MMMSpec(**base)


def _ids(rep):
    return {t.id for t in rep.tips}


def test_every_tip_has_evidence_and_both_actions(tmp_path):
    spec = _spec()
    rep = advise(spec, run_audit(spec, base=tmp_path))
    assert rep.tips
    for t in rep.tips:
        assert t.evidence and t.why and t.now and t.future, t.id
        assert t.area in AREAS


def test_missing_price_and_experiments_are_flagged(tmp_path):
    spec = _spec()
    ids = _ids(advise(spec, run_audit(spec, base=tmp_path)))
    assert {"no-price", "no-experiments", "no-exposure", "no-distribution"} <= ids


def test_price_control_removes_price_tip(tmp_path):
    df = pd.read_csv(DATA / "synthetic_weekly.csv")
    df["avg_price"] = 10 + (df.index % 7)
    spec = _spec(controls=[ControlMeta(column="avg_price", category="pricing")])
    assert "no-price" not in _ids(advise(spec, run_audit(spec, base=tmp_path, df=df), df))


def test_geo_level_data_in_national_spec(tmp_path):
    df = pd.read_csv(DATA / "synthetic_panel.csv")
    spec = _spec(data_path=str(DATA / "synthetic_panel.csv"))
    rep = advise(spec, run_audit(spec, base=tmp_path, df=df), df)
    dup = next(t for t in rep.tips if t.id == "duplicates")
    assert "geo-level" in dup.now and "do not deduplicate" in dup.now
    assert any(o.title == "Model at geo level" for o in rep.opportunities)


def test_unused_columns_are_classified(tmp_path):
    df = pd.read_csv(DATA / "synthetic_weekly.csv")
    df["tv_grps"] = df["spend_tv"] / 100
    df["promo_depth"] = 0.1
    spec = _spec()
    rep = advise(spec, run_audit(spec, base=tmp_path, df=df), df)
    titles = {o.title for o in rep.opportunities}
    assert {"Unused exposure columns", "Unused price columns"} <= titles


def test_incomplete_experiments(tmp_path):
    spec = _spec(experiments=[Experiment(channel="spend_tv", start_date="2023-01-02", end_date="2023-02-27", lift_absolute=500.0)])
    ids = _ids(advise(spec, run_audit(spec, base=tmp_path)))
    assert "experiments-incomplete" in ids and "no-experiments" not in ids


def test_tips_are_sorted_by_priority(tmp_path):
    spec = _spec()
    tips = advise(spec, run_audit(spec, base=tmp_path)).sorted_tips()
    assert [t.priority for t in tips] == sorted(t.priority for t in tips)


def test_scorecard_covers_every_area(tmp_path):
    spec = _spec()
    rep = advise(spec, run_audit(spec, base=tmp_path))
    assert set(rep.scorecard) == set(AREAS)
    assert rep.scorecard["business_drivers"] == "critical"


def test_run_advisor_writes_outputs(tmp_path):
    spec = _spec()
    run_audit(spec, base=tmp_path)
    run_advisor(spec, base=tmp_path)
    out = tmp_path / "mmm-workspace" / "audit"
    md = (out / "data_quality_tips.md").read_text()
    assert "## Scorecard" in md and "Data collection checklist" in md
    data = json.loads((out / "data_quality_tips.json").read_text())
    assert data["tips"][0]["priority_label"] == "critical"


def test_run_advisor_without_audit_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_advisor(_spec(), base=tmp_path)


def test_brownfield_without_known_results(tmp_path):
    spec = _spec(mmm_type="brownfield")
    rep = advise(spec, run_audit(spec, base=tmp_path))
    assert any("previous model" in o.title for o in rep.opportunities)


def test_clean_render_when_no_tips():
    from agent_mmm.data_advisor import AdvisorReport
    assert "No data-quality issues" in render_tips(AdvisorReport(scorecard={a: "good" for a in AREAS}))
