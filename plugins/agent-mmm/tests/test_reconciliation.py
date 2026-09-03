"""Reconciliation must name the *shape* of the disagreement, not just its size."""
import numpy as np
import pandas as pd
import pytest

from agent_mmm.reconciliation import reconcile_spend, render_reconciliation_report


@pytest.fixture
def weeks() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", periods=60, freq="W-MON")


@pytest.fixture
def base(weeks) -> np.ndarray:
    return np.random.default_rng(3).normal(10_000, 800, len(weeks))


def _modelled(weeks, base) -> pd.DataFrame:
    return pd.DataFrame({"date": weeks, "tv_spend": base * 0.5, "social_spend": base * 0.5})


def _finance(weeks, values) -> pd.DataFrame:
    return pd.DataFrame({"date": weeks, "spend": values})


class TestPatterns:
    def test_aligned(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base))
        assert r.pattern == "aligned"
        assert r.ok
        assert r.n_within_tolerance == r.n_periods

    def test_constant_fee_shortfall(self, weeks, base):
        """The ledger carries agency fees the platform export does not."""
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base / 0.85))
        assert r.pattern == "constant_shortfall"
        assert not r.ok
        assert any("agency fees" in w for w in r.warnings)
        assert any("gross or net" in q for q in r.questions)

    def test_constant_excess(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base * 0.85))
        assert r.pattern == "constant_excess"
        assert any("two exports" in q for q in r.questions)

    def test_drifting_gap_points_at_a_missing_channel(self, weeks, base):
        r = reconcile_spend(
            _modelled(weeks, base), _finance(weeks, base + np.linspace(0, 4000, len(weeks)))
        )
        assert r.pattern == "drifting"
        assert any("What changed around" in q for q in r.questions)

    def test_timing_shift_keeps_totals_but_breaks_periods(self, weeks, base):
        """Totals agree, periods do not — worse for an MMM than a level gap,
        because it corrupts carryover directly."""
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, np.roll(base, 1)))
        assert r.pattern == "timing_shift"
        assert abs(r.total_pct) < 0.01
        assert r.n_within_tolerance < r.n_periods
        assert any("delivery" in q for q in r.questions)

    def test_isolated_spikes(self, weeks, base):
        values = base.copy()
        values[10] *= 1.8
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, values))
        assert r.pattern == "isolated_spikes"
        assert any("Rebates" in q for q in r.questions)


class TestThresholds:
    def test_total_gap_beyond_tolerance_fails(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base * 1.2))
        assert not r.ok
        assert any("differs from the ledger" in e for e in r.errors)

    def test_small_gap_within_tolerance_passes(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base * 1.01))
        assert r.ok

    def test_custom_tolerance_is_respected(self, weeks, base):
        tight = reconcile_spend(
            _modelled(weeks, base), _finance(weeks, base * 1.03), max_total_variance=0.01
        )
        assert not tight.ok
        loose = reconcile_spend(
            _modelled(weeks, base), _finance(weeks, base * 1.03), max_total_variance=0.10
        )
        assert loose.ok

    def test_period_level_disagreement_warns_even_when_totals_match(self, weeks, base):
        noisy = base * np.random.default_rng(1).choice([0.8, 1.25], len(weeks))
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, noisy))
        assert any("Period-level disagreement" in w for w in r.warnings)


class TestCoverageGaps:
    def test_ledger_period_missing_from_model_is_an_error(self, weeks, base):
        m = _modelled(weeks[:-5], base[:-5])
        f = _finance(weeks, base)
        r = reconcile_spend(m, f)
        assert len(r.missing_in_modelled) == 5
        assert any("media the model cannot see" in e for e in r.errors)

    def test_model_period_missing_from_ledger_warns(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks[:-5], base[:-5]))
        assert len(r.missing_in_finance) == 5
        assert any("full window" in w for w in r.warnings)

    def test_no_overlap_is_an_error(self, weeks, base):
        later = pd.date_range("2030-01-07", periods=len(weeks), freq="W-MON")
        r = reconcile_spend(_modelled(weeks, base), _finance(later, base))
        assert not r.ok
        assert any("share no periods" in e for e in r.errors)


class TestMechanics:
    def test_spend_columns_are_auto_detected(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base))
        assert r.modelled_total == pytest.approx(base.sum())

    def test_explicit_spend_columns_are_respected(self, weeks, base):
        r = reconcile_spend(
            _modelled(weeks, base), _finance(weeks, base * 0.5), modelled_spend_columns=["tv_spend"]
        )
        assert r.modelled_total == pytest.approx(base.sum() * 0.5)
        assert r.ok

    def test_unknown_column_raises(self, weeks, base):
        with pytest.raises(KeyError):
            reconcile_spend(
                _modelled(weeks, base), _finance(weeks, base), modelled_spend_columns=["nope"]
            )

    def test_daily_model_reconciles_against_weekly_ledger(self, base):
        days = pd.date_range("2022-01-03", periods=140, freq="D")
        daily = pd.DataFrame({"date": days, "tv_spend": 100.0})
        weekly = pd.date_range("2022-01-03", periods=20, freq="W-MON")
        ledger = pd.DataFrame({"date": weekly, "spend": 700.0})
        r = reconcile_spend(daily, ledger, freq="W-MON")
        assert r.pattern == "aligned"
        assert r.ok

    def test_always_asks_who_signs_off(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base))
        assert any("signs off" in q for q in r.questions)

    def test_renders_markdown(self, weeks, base):
        r = reconcile_spend(_modelled(weeks, base), _finance(weeks, base / 0.85))
        md = render_reconciliation_report(r)
        assert md.startswith("# Spend Reconciliation")
        assert "constant_shortfall" in md
        assert "## Largest variances" in md

    def test_json_serialisable(self, weeks, base):
        import json

        json.dumps(reconcile_spend(_modelled(weeks, base), _finance(weeks, base)).to_dict())
