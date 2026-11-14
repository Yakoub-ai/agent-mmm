"""Tests for budget.py: allocation, the guardrail ladder, baselines and recommendations."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent))

from agent_mmm.budget import (
    GUARDRAILS, ChannelCurve, Guardrail, allocate, curves_from_results, evaluate, plan_budget, recommend,
    render_plan, run_budget,
)
from agent_mmm.compare import compare
from test_results_and_compare import CH, _results


def _curve(name, current, scale, sat, draws=50, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([0, .25, .5, .75, .9, 1, 1.1, 1.25, 1.5, 2, 2.5, 3])
    grid = current * m
    k = scale * rng.lognormal(0, noise, size=(draws, 1)) if noise else np.full((draws, 1), scale)
    d = k * (1 - np.exp(-grid / (sat * current)))[None, :]
    return ChannelCurve(name, current, grid, d, d.mean(axis=0))


def test_allocate_moves_money_to_the_steeper_curve():
    # marginal return at current spend: a ≈ 2.7, b ≈ 16 -> money should flow to b
    curves = {"a": _curve("a", 100, 1000, 0.5), "b": _curve("b", 100, 10000, 5.0)}
    bounds = {c: (cv.current * 0.5, cv.current * 1.5) for c, cv in curves.items()}
    alloc, feasible, total = allocate(curves, 200, bounds)
    assert feasible and total == pytest.approx(200)
    assert alloc["b"] > alloc["a"]
    assert float(np.mean(evaluate(curves, alloc))) > 0


def test_allocate_respects_bounds_and_reports_infeasible():
    curves = {"a": _curve("a", 100, 1000, 1.0), "b": _curve("b", 100, 1000, 1.0)}
    bounds = {c: (90.0, 110.0) for c in curves}
    alloc, feasible, total = allocate(curves, 300, bounds)
    assert not feasible and total == pytest.approx(220)
    assert all(90 - 1e-9 <= v <= 110 + 1e-9 for v in alloc.values())


def test_evaluate_is_zero_at_current_spend():
    curves = {"a": _curve("a", 100, 1000, 1.0, noise=0.3)}
    assert np.allclose(evaluate(curves, {"a": 100.0}), 0)


def test_ladder_gains_grow_with_looser_guardrails():
    p = plan_budget(_results("pymc-marketing", roas=(3.0, 1.0, 0.3)))
    gains = [s.gain["mean"] for s in p.scenarios if s.budget_factor == 1.0]
    assert gains == sorted(gains)
    assert len([s for s in p.scenarios if s.budget_factor == 1.0]) == len(GUARDRAILS)


def test_aggressive_level_is_never_recommended():
    p = plan_budget(_results("pymc-marketing", roas=(3.0, 1.0, 0.3), spread=0.01))
    rec = p.recommendations["1"]
    assert rec["guardrail"] != "aggressive"
    assert any("extrapolates" in r or "beyond" in r for r in rec["reasons"])


def test_low_confidence_stops_the_ladder():
    p = plan_budget(_results("pymc-marketing", roas=(1.0, 1.0, 1.0), spread=1.5))
    rec = p.recommendations["1"]
    stopped = [r for r in rec["reasons"] if "chance of beating" in r]
    assert rec["guardrail"] is None or stopped


def test_budget_cut_is_compared_with_scaled_mix_not_today():
    p = plan_budget(_results("pymc-marketing", roas=(3.0, 1.0, 0.3), spread=0.05))
    cut = [s for s in p.scenarios if s.budget_factor == 0.9 and s.guardrail == "large"][0]
    assert cut.gain["mean"] is not None and cut.gain_vs_mix["mean"] > 0
    rec = p.recommendations["0.9"]
    assert rec.get("baseline") == "scaling today's mix" or rec["guardrail"] is None


def test_no_room_levels_are_skipped_not_failed():
    p = plan_budget(_results("pymc-marketing", roas=(3.0, 1.0, 0.3), spread=0.05))
    rec = p.recommendations["1.1"]
    assert any("no room to re-allocate" in r for r in rec["reasons"])
    assert rec["guardrail"] not in (None, "small")


def test_disputed_channels_are_held_to_small_moves():
    a = _results("pymc-marketing", roas=(2.0, 1.0, 0.5), spread=0.05)
    b = _results("meridian", roas=(0.4, 1.0, 0.5), spread=0.05, seed=3)
    p = plan_budget(a, compare([a, b]))
    assert p.disputed == ["tv"]
    for s in p.scenarios:
        if s.budget_factor == 1.0:
            assert abs(s.allocation["tv"]["change_pct"]) <= 0.1 + 1e-6
    released = plan_budget(a, compare([a, b]), release_disputed=True)
    assert released.disputed == []


def test_locked_channel_does_not_move():
    r = _results("pymc-marketing")
    tv = curves_from_results(r)["tv"].current
    p = plan_budget(r, locked={"tv": tv})
    assert all(abs(s.allocation["tv"]["new"] - tv) < 1e-6 for s in p.scenarios if s.feasible)


def test_unconverged_model_is_flagged():
    p = plan_budget(_results("pymc-marketing", rhat=1.3))
    assert any("did not fully converge" in n for n in p.notes)


def test_frontier_is_monotone_for_each_guardrail():
    p = plan_budget(_results("pymc-marketing"))
    for key, fr in p.frontiers.items():
        ok = [x["optimised_gain"] for x in fr if x["feasible"]]
        assert all(b >= a - 1e-6 for a, b in zip(ok, ok[1:])), key


def test_custom_guardrails():
    g = (Guardrail("tight", "Tight (±5%)", 0.95, 1.05),)
    p = plan_budget(_results("pymc-marketing"), guardrails=g, budget_factors=(1.0,))
    assert [s.guardrail for s in p.scenarios] == ["tight"]
    assert p.recommendations["1"]["guardrail"] in ("tight", None)
    assert "Tight (±5%)" in render_plan(p)


def test_render_and_write(tmp_path):
    rs = [_results("pymc-marketing"), _results("meridian", seed=2)]
    out = run_budget(rs, compare(rs), base=tmp_path)
    for run_id in ("run_pymc-marketing", "run_meridian"):
        w = out["written"][run_id]
        text = Path(w["report"]).read_text()
        assert "## The guardrail ladder" in text and "Small adjustments" in text
        json.loads(Path(w["json"]).read_text())
        assert all(Path(f).stat().st_size > 5000 for f in w["figures"])
    assert "Do the models agree" in Path(out["written"]["agreement"]).read_text()
    assert "## Recommendation" in render_plan(out["plans"][0])


def test_no_curves_raises():
    r = _results("pymc-marketing")
    r.curves = {"multipliers": [], "channels": {}}
    with pytest.raises(ValueError):
        plan_budget(r)


def test_saved_plans_round_trip_with_custom_settings(tmp_path):
    from agent_mmm.budget import load_plans
    g = (Guardrail("tight", "Tight (±5%)", 0.95, 1.05),) + GUARDRAILS
    r = _results("pymc-marketing")
    tv = curves_from_results(r)["tv"].current
    run_budget([r], base=tmp_path, guardrails=g, locked={"tv": tv})
    (p,) = load_plans(tmp_path)
    assert p.guardrails[0].key == "tight" and p.locked == {"tv": tv}
    assert render_plan(p) == render_plan(plan_budget(r, guardrails=g, locked={"tv": tv}))


def test_report_includes_budget_section():
    pytest.importorskip("plotly")
    from agent_mmm.charts import build_html
    r = _results("pymc-marketing")
    page = build_html([r], budget_plans=[plan_budget(r)])
    assert "Budget scenarios" in page and "Small adjustments (±10%)" in page
