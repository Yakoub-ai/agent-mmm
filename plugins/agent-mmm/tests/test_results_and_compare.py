"""Tests for results_schema, compare, and the Meridian launcher (no fitting)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm import meridian_runner
from agent_mmm.compare import compare, render_comparison, run_comparison
from agent_mmm.results_schema import (
    DEFAULT_MULTIPLIERS, ModelResults, build_results, fit_metrics, holdout_mask, load_all, summarize,
)

CH = ["tv", "search", "social"]


def _results(framework="a", roas=(2.0, 1.0, 0.5), spread=0.2, rhat=1.0, ess=1000, div=0,
             holdout=13, seed=0, T=104):
    rng = np.random.default_rng(seed)
    S, C = 200, len(CH)
    hmask = holdout_mask(T, 52, holdout)
    n_train = int((~hmask).sum())
    spend = rng.uniform(50, 150, size=(T, C))
    beta = np.array(roas) * rng.lognormal(0, spread, size=(S, C))           # per-draw ROAS
    contrib = spend[~hmask][None, :, :] * beta[:, None, :]                    # (S, Ttr, C)
    base = 1000.0
    expected = base + np.concatenate(
        [contrib.sum(axis=2), (spend[hmask][None] * beta[:, None]).sum(axis=2)], axis=1)
    actual = expected.mean(axis=0) + rng.normal(0, 20, T)
    m = np.array(DEFAULT_MULTIPLIERS)
    curves = {ch: contrib[:, :, j].sum(axis=1)[:, None] * (1 - np.exp(-m))[None, :] / (1 - np.exp(-1))
              for j, ch in enumerate(CH)}
    dates = [f"2022-{(i // 4) % 12 + 1:02d}-{i % 4 * 7 + 1:02d}" for i in range(T)]
    return build_results(
        framework=framework, run_id=f"run_{framework}", target="y", target_label="sale",
        kpi_kind="non_revenue", channels=CH, dates=dates, actual=actual, expected=expected,
        contributions=contrib, spend=spend, holdout=hmask, curve_draws=curves,
        diagnostics={"max_rhat": rhat, "min_ess": ess, "divergences": div},
    )


# --------------------------------------------------------------------------- #
# results_schema
# --------------------------------------------------------------------------- #
def test_holdout_mask_auto_and_explicit():
    m = holdout_mask(104, 52, "auto")
    assert m.sum() == 13 and m[-13:].all()
    assert holdout_mask(104, 52, 0).sum() == 0
    assert holdout_mask(20, 52, "auto").sum() == 3          # capped at 15% of data
    dates = [f"2023-01-{d:02d}" for d in range(1, 31)]
    assert holdout_mask(30, 365, "auto", dates=dates, start="2023-01-25").sum() == 6


def test_holdout_too_large_is_refused():
    assert holdout_mask(20, 52, 15).sum() == 0


def test_build_results_recovers_known_roas():
    r = _results(roas=(2.0, 1.0, 0.5), spread=0.01)
    for ch, true in zip(CH, (2.0, 1.0, 0.5)):
        assert r.channel_summary[ch]["roas"]["mean"] == pytest.approx(true, rel=0.02)
    assert r.window["n_holdout"] == 13 and r.window["n_train"] == 91
    assert r.fit["mape_holdout"] is not None
    # Saturating curve: marginal ROAS below average ROAS.
    assert r.channel_summary["tv"]["mroas"]["mean"] < r.channel_summary["tv"]["roas"]["mean"]


def test_curves_are_thinned_and_monotone():
    r = _results()
    c = r.curves["channels"]["tv"]
    assert len(c["draws"]) <= 100 and len(c["mean"]) == len(DEFAULT_MULTIPLIERS)
    assert all(b >= a for a, b in zip(c["mean"], c["mean"][1:]))
    assert c["spend"][DEFAULT_MULTIPLIERS.index(1.0)] == pytest.approx(r.channel_summary["tv"]["spend"])


def test_round_trip_and_json_safety(tmp_path):
    r = _results()
    r.diagnostics["weird"] = float("nan")
    p = r.save(tmp_path / "runs" / "x" / "results.json")
    assert "NaN" not in p.read_text()
    back = ModelResults.load(p)
    assert back.channel_summary == r.to_dict()["channel_summary"]
    assert [x.run_id for x in load_all(tmp_path / "runs")] == ["run_a"]


def test_wrong_schema_version_rejected(tmp_path):
    p = tmp_path / "r.json"
    d = _results().to_dict()
    d["schema_version"] = "999"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ModelResults.load(p)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_results(framework="x", run_id="x", target="y", target_label="s", kpi_kind="revenue",
                      channels=["a"], dates=["d1", "d2"], actual=np.ones(2), expected=np.ones((5, 2)),
                      contributions=np.ones((5, 3, 1)), spend=np.ones((2, 1)),
                      holdout=np.zeros(2, bool), curve_draws={})


def test_channel_without_spend_has_no_roas():
    r = _results()
    spend = np.array([[np.nan]] * 4)
    res = build_results(framework="x", run_id="x", target="y", target_label="s", kpi_kind="revenue",
                        channels=["organic"], dates=["a", "b", "c", "d"], actual=np.ones(4),
                        expected=np.ones((3, 4)) * 2, contributions=np.ones((3, 4, 1)) * 0.5,
                        spend=spend, holdout=np.zeros(4, bool), curve_draws={})
    assert res.channel_summary["organic"]["spend"] is None and "roas" not in res.channel_summary["organic"]


def test_summary_helpers():
    assert summarize([np.nan])["mean"] is None
    assert fit_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["mape"] == 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def test_agreeing_models():
    c = compare([_results("a", seed=1, spread=0.1), _results("b", seed=2, spread=0.1)])
    assert c["comparable"]
    assert {e["agreement"] for e in c["channels"].values()} <= {"agree-precise", "agree-uncertain"}
    assert c["recommendation"]["model"] is None          # same data -> holdout tie
    assert any(f["kind"] == "forecast" and "tie" in f["finding"] for f in c["findings"])


def test_disagreement_is_flagged():
    c = compare([_results("a", roas=(2.0, 1.0, 0.5), spread=0.05),
                 _results("b", roas=(0.4, 1.0, 0.5), spread=0.05, seed=3)])
    assert c["channels"]["tv"]["agreement"] == "disagree"
    assert any(f["kind"] == "identification" and "tv" in f["finding"] for f in c["findings"])


def test_non_converged_model_is_excluded():
    c = compare([_results("a"), _results("b", rhat=1.2, div=30, seed=4)])
    bad = next(m for m in c["models"] if m["framework"] == "b")
    assert not bad["converged"]
    assert c["recommendation"]["model"] != bad["label"]


def test_different_windows_are_not_comparable():
    c = compare([_results("a"), _results("b", holdout=8)])
    assert not c["comparable"] and c["findings"][0]["kind"] == "blocking"
    assert c["recommendation"]["model"] is None


def test_needs_two_models():
    with pytest.raises(ValueError):
        compare([_results()])


def test_render_and_write(tmp_path):
    c, md = run_comparison([_results("a"), _results("b", seed=5)], base=tmp_path)
    text = md.read_text()
    assert "# Framework comparison" in text and "Holdout MAPE" in text and "| tv |" in text
    assert (tmp_path / "mmm-workspace" / "comparison" / "comparison.json").exists()
    assert render_comparison(c) == text


# --------------------------------------------------------------------------- #
# Meridian launcher (no Meridian needed)
# --------------------------------------------------------------------------- #
def test_launch_requires_meridian_env(tmp_path):
    with pytest.raises(FileNotFoundError, match="mmm-setup"):
        meridian_runner.launch("mmm-workspace/spec.yaml", "r1", base=tmp_path)


def test_launch_uses_argv_not_shell(tmp_path, monkeypatch):
    py = tmp_path / "py"
    py.write_text("")
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw

        class P:
            returncode = 0
        return P()

    monkeypatch.setattr(meridian_runner.subprocess, "run", fake_run)
    out = meridian_runner.launch("mmm-workspace/spec.yaml", "r1", base=tmp_path, python=str(py))
    assert isinstance(seen["argv"], list) and "shell" not in seen["kw"]
    assert seen["argv"][1] == "-P" and "--run-id" in seen["argv"]
    assert out.name == "results.json"


@pytest.mark.parametrize("bad", ["../x", "a/b", ".."])
def test_runner_rejects_path_like_run_ids(bad):
    with pytest.raises(SystemExit):
        meridian_runner.main(["--run-id", bad])


def test_runner_module_imports_without_meridian():
    """The main env imports meridian_runner to call launch(); Meridian must load lazily."""
    assert "meridian.model" not in sys.modules


@pytest.mark.slow
@pytest.mark.requires_pymc
def test_pymc_fit_exports_comparable_results(tmp_path):
    pytest.importorskip("pymc_marketing")
    from agent_mmm.fit_runner import SAMPLER_SMOKE, run_fit
    from agent_mmm.spec import ChannelMeta, MMMSpec, TargetUnit, TargetUnitKind

    data = Path(__file__).parent / "data" / "synthetic_weekly.csv"
    spec = MMMSpec(mmm_type="greenfield", company_name="T", industry="t", region="SE",
                   data_path=str(data), target_unit=TargetUnit(kind=TargetUnitKind.acquisition, label="sale"),
                   channels=[ChannelMeta(column=c) for c in ["spend_sem", "spend_tv"]])
    m = run_fit(spec, sampler_config=dict(SAMPLER_SMOKE, progressbar=False), run_id="t",
                base=tmp_path, holdout="auto", skip_prior_pc=True)
    assert m["holdout_periods"] == 13 and "results_error" not in m
    r = ModelResults.load(m["results_path"])
    assert r.window["n_holdout"] == 13 and r.fit["mape_holdout"] is not None
    tv = r.curves["channels"]["spend_tv"]
    assert tv["mean"][0] == 0 and tv["mean"][-1] > tv["mean"][DEFAULT_MULTIPLIERS.index(1.0)]
