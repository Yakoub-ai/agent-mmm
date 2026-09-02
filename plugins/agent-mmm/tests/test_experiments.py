"""Power calculations must be right, and the roadmap must refuse to recommend
a test that cannot detect the effect it is looking for."""
import math

import numpy as np
import pandas as pd
import pytest

from agent_mmm.experiments import (
    ChannelTestCandidate,
    build_roadmap,
    design_geo_holdout,
    duration_noise_factor,
    estimate_geo_cv,
    estimate_period_cv,
    estimate_pre_period_correlation,
    expected_lift_from_share,
    geo_lift_mde,
    holdout_mde,
    prioritise_channels,
    render_roadmap,
    required_duration_for_mde,
    required_geos_for_mde,
    z_score,
)


class TestZScore:
    @pytest.mark.parametrize(
        "p,expected",
        [(0.5, 0.0), (0.80, 0.8416212), (0.90, 1.2815516), (0.95, 1.6448536), (0.975, 1.9599640),
         (0.99, 2.3263479), (0.005, -2.5758293)],
    )
    def test_matches_standard_normal_table(self, p, expected):
        assert z_score(p) == pytest.approx(expected, abs=1e-6)

    def test_symmetry(self):
        assert z_score(0.3) == pytest.approx(-z_score(0.7), abs=1e-9)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_raises(self, p):
        with pytest.raises(ValueError):
            z_score(p)


class TestDurationNoise:
    def test_single_period_is_no_reduction(self):
        assert duration_noise_factor(1) == pytest.approx(1.0)

    def test_independent_periods_give_root_n(self):
        assert duration_noise_factor(9, residual_autocorrelation=0.0) == pytest.approx(1 / 3)

    def test_autocorrelation_limits_the_gain(self):
        """Designing on an independence assumption produces a test that is
        quietly about half as powerful as its plan claimed."""
        independent = duration_noise_factor(8, residual_autocorrelation=0.0)
        realistic = duration_noise_factor(8, residual_autocorrelation=0.3)
        assert realistic > independent
        assert realistic == pytest.approx(math.sqrt((1 + 7 * 0.3) / 8))

    def test_monotone_in_duration(self):
        factors = [duration_noise_factor(n) for n in (1, 2, 4, 8, 16, 32)]
        assert factors == sorted(factors, reverse=True)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_invalid_duration_raises(self, bad):
        with pytest.raises(ValueError):
            duration_noise_factor(bad)

    def test_invalid_autocorrelation_raises(self):
        with pytest.raises(ValueError):
            duration_noise_factor(8, residual_autocorrelation=1.0)


class TestGeoLiftMDE:
    def test_known_value(self):
        # (z_.90 + z_.80) * cv * sqrt(1/n + 1/n), one period, no adjustment
        expected = (z_score(0.90) + z_score(0.80)) * 0.5 * math.sqrt(2 / 20)
        assert geo_lift_mde(20, 20, 0.5) == pytest.approx(expected)

    def test_more_geos_detect_smaller_effects(self):
        assert geo_lift_mde(50, 50, 0.4) < geo_lift_mde(10, 10, 0.4)

    def test_noisier_outcome_needs_bigger_effect(self):
        assert geo_lift_mde(20, 20, 0.8) > geo_lift_mde(20, 20, 0.2)

    def test_pre_period_adjustment_is_a_large_free_win(self):
        plain = geo_lift_mde(20, 20, 0.5)
        adjusted = geo_lift_mde(20, 20, 0.5, pre_period_correlation=0.9)
        assert adjusted == pytest.approx(plain * math.sqrt(1 - 0.81))
        assert adjusted < plain * 0.5

    def test_longer_tests_detect_smaller_effects(self):
        assert geo_lift_mde(20, 20, 0.35, duration_periods=12) < geo_lift_mde(
            20, 20, 0.35, duration_periods=2
        )

    def test_two_sided_costs_power(self):
        assert geo_lift_mde(20, 20, 0.5, one_sided=False) > geo_lift_mde(20, 20, 0.5)

    def test_realistic_design_is_in_a_believable_range(self):
        """20v20 geos, CV 0.35, well-matched pre-period, 8 weeks — published
        geo-lift tooling lands in the same ballpark."""
        mde = geo_lift_mde(20, 20, 0.35, pre_period_correlation=0.9, duration_periods=8)
        assert 0.03 < mde < 0.12

    @pytest.mark.parametrize("kwargs", [
        {"n_treatment": 0, "n_control": 10, "cv": 0.5},
        {"n_treatment": 10, "n_control": 0, "cv": 0.5},
        {"n_treatment": 10, "n_control": 10, "cv": -0.1},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            geo_lift_mde(**kwargs)

    def test_correlation_out_of_range_raises(self):
        with pytest.raises(ValueError):
            geo_lift_mde(10, 10, 0.5, pre_period_correlation=1.0)


class TestRequiredGeos:
    def test_round_trips_with_the_mde_formula(self):
        n_t, n_c = required_geos_for_mde(0.08, 0.4, pre_period_correlation=0.9, duration_periods=8)
        achieved = geo_lift_mde(n_t, n_c, 0.4, pre_period_correlation=0.9, duration_periods=8)
        assert achieved <= 0.08 + 1e-9
        # And one fewer geo per arm would not have been enough.
        if n_t > 1:
            assert geo_lift_mde(n_t - 1, n_c - 1, 0.4, pre_period_correlation=0.9,
                                duration_periods=8) > 0.08

    def test_smaller_targets_need_more_geos(self):
        assert required_geos_for_mde(0.02, 0.4)[0] > required_geos_for_mde(0.10, 0.4)[0]

    def test_unequal_control_ratio(self):
        n_t, n_c = required_geos_for_mde(0.10, 0.4, control_ratio=2.0)
        assert n_c == pytest.approx(n_t * 2, abs=1)

    def test_non_positive_target_raises(self):
        with pytest.raises(ValueError):
            required_geos_for_mde(0.0, 0.4)


class TestHoldoutMDE:
    def test_longer_windows_detect_smaller_effects(self):
        assert holdout_mde(16, 0.2, n_pre_periods=26) < holdout_mde(4, 0.2, n_pre_periods=26)

    def test_no_pre_period_is_rejected_not_scored(self):
        """Without a comparison window no lift is estimable at any sample size.
        Returning a small-looking MDE here would be actively misleading."""
        with pytest.raises(ValueError, match="counterfactual"):
            holdout_mde(8, 0.2, n_pre_periods=0)

    def test_longer_pre_period_helps(self):
        assert holdout_mde(8, 0.2, n_pre_periods=52) < holdout_mde(8, 0.2, n_pre_periods=8)

    def test_required_duration_round_trips(self):
        n = required_duration_for_mde(0.10, 0.25, pre_ratio=2.0)
        assert holdout_mde(n, 0.25, n_pre_periods=int(n * 2)) <= 0.10 + 1e-9

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            holdout_mde(0, 0.2)
        with pytest.raises(ValueError):
            holdout_mde(8, 0.2, n_pre_periods=-1)

    def test_design_without_pre_period_is_blocked_not_raised(self):
        """Inside a roadmap this must degrade to a blocked plan, not an
        exception that takes the whole roadmap down."""
        c = ChannelTestCandidate("tv", spend=1000.0, contribution_share=0.09, geo_testable=False)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3, pre_period_periods=0)
        assert plan.blocked_reason
        assert not plan.viable
        assert math.isinf(plan.power.mde_relative)


class TestEstimators:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        dates = pd.date_range("2022-01-03", periods=80, freq="W-MON")
        sizes = {"lon": 1000.0, "man": 400.0, "bir": 350.0, "gla": 200.0}
        rows = []
        for d in dates:
            for geo, size in sizes.items():
                rows.append({"date": d, "geo": geo, "y": size * rng.normal(1.0, 0.08)})
        return pd.DataFrame(rows)

    def test_geo_cv_reflects_market_size_spread(self, panel):
        cv = estimate_geo_cv(panel, geo_column="geo", target_column="y")
        assert 0.5 < cv < 1.2

    def test_geo_cv_zero_when_all_geos_identical(self):
        df = pd.DataFrame({"geo": ["a", "b", "c"] * 10, "y": [100.0] * 30})
        assert estimate_geo_cv(df, geo_column="geo", target_column="y") == pytest.approx(0.0)

    def test_pre_period_correlation_is_high_for_a_stable_panel(self, panel):
        """Market sizes are persistent, which is exactly why a pre-period
        adjusted design detects effects an unadjusted one cannot."""
        rho = estimate_pre_period_correlation(
            panel, geo_column="geo", target_column="y", date_column="date"
        )
        assert rho > 0.95

    def test_period_cv_removes_trend(self):
        dates = pd.date_range("2022-01-03", periods=60, freq="W-MON")
        trending = pd.DataFrame({"date": dates, "y": np.linspace(100, 300, 60)})
        assert estimate_period_cv(trending, target_column="y", date_column="date") == pytest.approx(
            0.0, abs=1e-6
        )
        raw = estimate_period_cv(trending, target_column="y", date_column="date", detrend=False)
        assert raw > 0.2

    def test_missing_columns_raise(self, panel):
        with pytest.raises(KeyError):
            estimate_geo_cv(panel, geo_column="nope", target_column="y")
        with pytest.raises(KeyError):
            estimate_pre_period_correlation(
                panel, geo_column="geo", target_column="nope", date_column="date"
            )


class TestExpectedLift:
    def test_carryover_discount_is_applied(self):
        """Ignoring in-flight carryover is a standard way to arrive at a test
        designed 15-30% underpowered."""
        assert expected_lift_from_share(0.10) == pytest.approx(0.085)

    def test_partial_holdout_scales_the_expectation(self):
        assert expected_lift_from_share(0.10, holdout_fraction=0.5) == pytest.approx(0.0425)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_share_raises(self, bad):
        with pytest.raises(ValueError):
            expected_lift_from_share(bad)

    def test_invalid_holdout_fraction_raises(self):
        with pytest.raises(ValueError):
            expected_lift_from_share(0.1, holdout_fraction=0.0)


class TestPrioritisation:
    def test_spend_shares_are_computed(self):
        ranked = prioritise_channels([
            ChannelTestCandidate("a", spend=750.0),
            ChannelTestCandidate("b", spend=250.0),
        ])
        assert sum(c.spend_share for c in ranked) == pytest.approx(1.0)

    def test_wide_intervals_mean_high_uncertainty(self):
        vague = ChannelTestCandidate("a", spend=1.0, roas_point=2.0, roas_ci_low=0.5, roas_ci_high=6.0)
        precise = ChannelTestCandidate("b", spend=1.0, roas_point=2.0, roas_ci_low=1.9, roas_ci_high=2.1)
        assert vague.uncertainty > precise.uncertainty

    def test_we_do_not_prioritise_testing_what_we_already_know(self):
        known = ChannelTestCandidate("known", spend=1000.0, roas_point=2.0,
                                     roas_ci_low=1.95, roas_ci_high=2.05)
        unknown = ChannelTestCandidate("unknown", spend=1000.0, roas_point=2.0,
                                       roas_ci_low=0.4, roas_ci_high=7.0)
        ranked = prioritise_channels([known, unknown])
        assert ranked[0].channel == "unknown"

    def test_non_geo_testable_channels_are_penalised(self):
        national = ChannelTestCandidate("tv", spend=1000.0, geo_testable=False)
        local = ChannelTestCandidate("ooh", spend=1000.0, geo_testable=True)
        assert national.feasibility < local.feasibility

    def test_always_on_channels_are_maximally_uncertain_without_a_model(self):
        assert ChannelTestCandidate("brand", spend=1.0, always_on=True).uncertainty > \
               ChannelTestCandidate("promo", spend=1.0, always_on=False).uncertainty

    def test_a_prior_experiment_lowers_priority(self):
        fresh = ChannelTestCandidate("a", spend=1000.0)
        fresh.spend_share = 1.0
        anchored = ChannelTestCandidate("a", spend=1000.0, has_prior_experiment=True)
        anchored.spend_share = 1.0
        assert anchored.priority < fresh.priority


class TestDesign:
    def test_powered_design_is_viable(self):
        c = ChannelTestCandidate("search_brand", spend=900_000, contribution_share=0.12,
                                 roas_point=8.0, roas_ci_low=3.0, roas_ci_high=14.0)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.35, duration_periods=8)
        assert plan.power.powered is True
        assert plan.viable
        assert not plan.blocked_reason

    def test_effect_below_the_floor_blocks_the_test(self):
        """A null from an underpowered test gets read as 'the channel does
        nothing', and the budget moves on bad evidence."""
        c = ChannelTestCandidate("display", spend=300_000, contribution_share=0.01)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.35, duration_periods=8)
        assert plan.power.powered is False
        assert not plan.viable
        assert "null result" in plan.blocked_reason
        assert any("Underpowered" in n for n in plan.power.notes)

    def test_non_geo_testable_falls_back_to_time_holdout(self):
        c = ChannelTestCandidate("tv", spend=2_400_000, contribution_share=0.09, geo_testable=False)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.35)
        assert plan.design == "time_holdout"
        assert "cannot be bought by geo" in plan.blocked_reason
        assert any("No control group" in r for r in plan.risks)

    def test_expected_result_is_stated_before_the_test(self):
        c = ChannelTestCandidate("tv", spend=1000.0, contribution_share=0.08, roas_point=1.4)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3)
        assert "8" in plan.expected_result or "6.8%" in plan.expected_result
        assert plan.falsifies
        assert "less incremental" in plan.falsifies

    def test_no_contribution_means_no_pre_registered_expectation(self):
        c = ChannelTestCandidate("new_channel", spend=1000.0)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3)
        assert plan.power.expected_lift_relative is None
        assert plan.power.powered is None
        assert "cannot be wrong" in plan.expected_result

    def test_execution_steps_warn_against_compensating_spend(self):
        c = ChannelTestCandidate("social", spend=1000.0, contribution_share=0.2)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3)
        assert any("compensate" in s for s in plan.execution)
        assert any("pre-test history" in s for s in plan.execution)

    def test_feeds_back_names_all_three_frameworks(self):
        c = ChannelTestCandidate("social", spend=1000.0, contribution_share=0.2)
        plan = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3)
        for framework in ("pymc-marketing", "Meridian", "Robyn"):
            assert framework in plan.feeds_back_as

    def test_spend_at_risk_scales_with_duration_and_arm_size(self):
        c = ChannelTestCandidate("social", spend=520_000, contribution_share=0.2)
        short = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3, duration_periods=4)
        long = design_geo_holdout(c, n_geos_available=40, geo_cv=0.3, duration_periods=16)
        assert long.spend_at_risk > short.spend_at_risk


class TestRoadmap:
    @pytest.fixture
    def candidates(self) -> list[ChannelTestCandidate]:
        return [
            ChannelTestCandidate("search_brand", spend=900_000, contribution_share=0.12,
                                 roas_point=8.2, roas_ci_low=3.0, roas_ci_high=14.0, always_on=True),
            ChannelTestCandidate("tv", spend=2_400_000, contribution_share=0.09, roas_point=1.4,
                                 roas_ci_low=0.6, roas_ci_high=2.6, geo_testable=False),
            ChannelTestCandidate("social_prospecting", spend=1_200_000, contribution_share=0.07,
                                 roas_point=2.1, roas_ci_low=1.5, roas_ci_high=2.8),
            ChannelTestCandidate("display", spend=300_000, contribution_share=0.01,
                                 roas_point=0.5, roas_ci_low=0.1, roas_ci_high=1.4),
        ]

    def test_plans_are_ranked(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.35)
        assert [p.rank for p in rm.plans] == [1, 2, 3, 4]
        assert rm.plans[0].priority >= rm.plans[-1].priority

    def test_only_viable_plans_are_sequenced(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.35)
        sequenced = {ch for wave in rm.waves for ch in wave}
        assert sequenced == {p.channel for p in rm.viable_plans}
        assert "tv" not in sequenced          # national-only, blocked

    def test_waves_are_serial_by_default(self, candidates):
        """Two geo tests in overlapping markets contaminate each other."""
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.15)
        assert all(len(w) == 1 for w in rm.waves)

    def test_concurrency_can_be_raised_deliberately(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.15, max_concurrent=2)
        assert any(len(w) == 2 for w in rm.waves)

    def test_max_tests_is_respected(self, candidates):
        assert len(build_roadmap(candidates, n_geos_available=40, geo_cv=0.35, max_tests=2).plans) == 2

    def test_underpowered_tests_prompt_for_more_power(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.35)
        assert any("more geos" in q for q in rm.questions)

    def test_too_few_geos_warns(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=3, geo_cv=0.35)
        assert any("geos available" in w for w in rm.warnings)

    def test_always_asks_who_signs_off_the_risk(self, candidates):
        rm = build_roadmap(candidates, n_geos_available=40, geo_cv=0.35)
        assert any("signs it" in q for q in rm.questions)
        assert any("peak trading" in q for q in rm.questions)

    def test_unfitted_channels_prompt_for_an_expectation(self):
        rm = build_roadmap(
            [ChannelTestCandidate("mystery", spend=1000.0)], n_geos_available=40, geo_cv=0.3
        )
        assert any("nothing to contradict" in q for q in rm.questions)

    def test_renders_markdown(self, candidates):
        md = render_roadmap(build_roadmap(candidates, n_geos_available=40, geo_cv=0.35))
        assert md.startswith("# Experiment Roadmap")
        assert "## Sequence" in md
        assert "**What we expect**" in md
        assert "**What would change our mind**" in md
        assert "**How to run it**" in md
        assert "**How it feeds back into the model**" in md

    def test_json_serialisable(self, candidates):
        import json

        json.dumps(build_roadmap(candidates, n_geos_available=40, geo_cv=0.35).to_dict())
