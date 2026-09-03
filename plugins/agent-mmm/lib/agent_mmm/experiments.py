"""Designing the experiments that make an MMM worth believing.

An MMM fitted to observational data reports the correlation structure of a media
plan that was never randomised. It cannot, on its own, tell you whether brand
search causes conversions or merely stands next to them. An experiment can, for
one channel, at one spend level, for one period — and that single anchored point
constrains the whole decomposition.

So the question is never "should we experiment", it is "which test, first, and
what will it actually be able to detect". This module answers that:

* **Power** — the minimum detectable effect for a geo holdout or a time-based
  holdout, given the number of geos, the duration, and how noisy the outcome is.
  Most media tests that get run are incapable of detecting the effect they are
  looking for, and this is knowable in advance, in an afternoon.
* **Priority** — which channel to test first, from money at risk, how uncertain
  we currently are, and how testable the channel actually is.
* **Expectation** — what result the current model predicts, stated *before* the
  test runs, so the test can surprise us. A test with no pre-registered
  expectation cannot fail, and therefore teaches nothing.
* **Sequencing** — a roadmap that does not run two contaminating tests in the
  same geos at the same time.

Every number here is a planning number. It tells you whether a design is worth
running, not what the answer will be.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "PowerResult",
    "ChannelTestCandidate",
    "ExperimentPlan",
    "Roadmap",
    "z_score",
    "duration_noise_factor",
    "geo_lift_mde",
    "required_geos_for_mde",
    "holdout_mde",
    "required_duration_for_mde",
    "estimate_geo_cv",
    "estimate_pre_period_correlation",
    "estimate_period_cv",
    "expected_lift_from_share",
    "prioritise_channels",
    "design_geo_holdout",
    "build_roadmap",
    "render_roadmap",
]


# --------------------------------------------------------------------------- #
# Normal quantiles — kept local so power calculations do not require scipy
# --------------------------------------------------------------------------- #
def z_score(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to about 1e-9 over the range power calculations use, which is far
    beyond the precision of the inputs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #
@dataclass
class PowerResult:
    """What a proposed design can and cannot detect."""

    design: str
    mde_relative: float
    """Minimum detectable lift, as a fraction of the control outcome."""
    expected_lift_relative: Optional[float] = None
    alpha: float = 0.10
    power: float = 0.80
    n_treatment: int = 0
    n_control: int = 0
    duration_periods: int = 0
    cv: float = 0.0
    pre_period_correlation: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def powered(self) -> Optional[bool]:
        """True when the effect we expect is larger than the smallest we could see."""
        if self.expected_lift_relative is None:
            return None
        return self.expected_lift_relative >= self.mde_relative

    @property
    def power_ratio(self) -> Optional[float]:
        if self.expected_lift_relative is None or self.mde_relative == 0:
            return None
        return self.expected_lift_relative / self.mde_relative

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "mde_relative": round(self.mde_relative, 5),
            "expected_lift_relative": (
                round(self.expected_lift_relative, 5) if self.expected_lift_relative is not None else None
            ),
            "powered": self.powered,
            "power_ratio": round(self.power_ratio, 3) if self.power_ratio is not None else None,
            "alpha": self.alpha,
            "power": self.power,
            "n_treatment": self.n_treatment,
            "n_control": self.n_control,
            "duration_periods": self.duration_periods,
            "cv": round(self.cv, 4),
            "pre_period_correlation": self.pre_period_correlation,
            "notes": self.notes,
        }


def _effect_multiplier(alpha: float, power: float, one_sided: bool) -> float:
    tail = alpha if one_sided else alpha / 2
    return z_score(1 - tail) + z_score(power)


def duration_noise_factor(duration_periods: int, residual_autocorrelation: float = 0.3) -> float:
    """How much running longer reduces residual noise.

    Averaging the outcome over ``n`` periods would cut noise by ``1/sqrt(n)`` if
    the periods were independent. They are not: a market that is running hot in
    week one tends to still be running hot in week two. Under AR(1)-like
    residual correlation ``phi`` the effective sample size is
    ``n / (1 + (n-1)*phi)``, so the factor is:

        sqrt((1 + (n - 1) * phi) / n)

    At phi = 0.3, eight weeks buys a 38% noise reduction rather than the 65% an
    independence assumption would promise. Designing on the independence
    assumption is a common way to arrive at a test that is quietly half as
    powerful as its plan said.
    """
    if duration_periods < 1:
        raise ValueError("duration_periods must be at least 1")
    if not 0.0 <= residual_autocorrelation < 1.0:
        raise ValueError("residual_autocorrelation must be in [0, 1)")
    n = duration_periods
    return math.sqrt((1 + (n - 1) * residual_autocorrelation) / n)


def geo_lift_mde(
    n_treatment: int,
    n_control: int,
    cv: float,
    *,
    alpha: float = 0.10,
    power: float = 0.80,
    one_sided: bool = True,
    pre_period_correlation: float = 0.0,
    duration_periods: int = 1,
    residual_autocorrelation: float = 0.3,
) -> float:
    """Minimum detectable relative lift for a geo holdout or geo split.

    ``cv`` is the coefficient of variation of the outcome *across geos* in a
    single period — the cross-sectional noise the test has to see through.

    ``pre_period_correlation`` is the correlation between each geo's pre-period
    and test-period outcome. Using it (CUPED, or a difference-in-differences on
    a matched pre-period) shrinks the residual noise by ``sqrt(1 - rho^2)``. At
    rho = 0.9 that is a 56% reduction in the detectable effect for free, which is
    usually worth far more than adding geos you do not have. Estimate it with
    ``estimate_pre_period_correlation`` rather than assuming it.

    ``duration_periods`` matters because the test statistic is computed on the
    window average, not on one period — see ``duration_noise_factor``.

    One-sided is the default: media tests ask "did it help", and insisting on a
    two-sided test inflates the required effect by about 15% for no gain in what
    the business actually decides.
    """
    if n_treatment < 1 or n_control < 1:
        raise ValueError("both arms need at least one geo")
    if cv < 0:
        raise ValueError("cv must be non-negative")
    if not -1.0 < pre_period_correlation < 1.0:
        raise ValueError("pre_period_correlation must be in (-1, 1)")

    effective_cv = cv * math.sqrt(1 - pre_period_correlation ** 2)
    effective_cv *= duration_noise_factor(duration_periods, residual_autocorrelation)
    return _effect_multiplier(alpha, power, one_sided) * effective_cv * math.sqrt(
        1 / n_treatment + 1 / n_control
    )


def required_geos_for_mde(
    target_mde: float,
    cv: float,
    *,
    alpha: float = 0.10,
    power: float = 0.80,
    one_sided: bool = True,
    pre_period_correlation: float = 0.0,
    duration_periods: int = 1,
    residual_autocorrelation: float = 0.3,
    control_ratio: float = 1.0,
) -> tuple[int, int]:
    """Geos per arm needed to detect ``target_mde``. Returns (treatment, control)."""
    if target_mde <= 0:
        raise ValueError("target_mde must be positive")
    effective_cv = cv * math.sqrt(1 - pre_period_correlation ** 2)
    effective_cv *= duration_noise_factor(duration_periods, residual_autocorrelation)
    k = _effect_multiplier(alpha, power, one_sided)
    # mde = k * cv * sqrt(1/n + 1/(r*n))  =>  n = (k*cv/mde)^2 * (1 + 1/r)
    n_t = ((k * effective_cv / target_mde) ** 2) * (1 + 1 / control_ratio)
    n_treatment = max(1, math.ceil(n_t))
    return n_treatment, max(1, math.ceil(n_treatment * control_ratio))


def holdout_mde(
    n_test_periods: int,
    cv: float,
    *,
    n_pre_periods: int = 0,
    alpha: float = 0.10,
    power: float = 0.80,
    one_sided: bool = True,
) -> float:
    """Minimum detectable relative lift for a time-based (on/off) holdout.

    Here ``cv`` is the period-to-period coefficient of variation of the outcome.
    The pre-period supplies the counterfactual, so its length enters the standard
    error alongside the test window's.

    A pre-period is mandatory. Without one there is nothing to compare the test
    window against, so no lift can be estimated at any sample size — that is a
    design error rather than an underpowered design, and this function rejects it
    rather than returning a number that would look reassuringly small. If the
    comparison is meant to come from elsewhere (unaffected markets or products),
    that is a synthetic-control design, not a pre/post holdout.
    """
    if n_test_periods < 1:
        raise ValueError("n_test_periods must be at least 1")
    if n_pre_periods < 1:
        raise ValueError(
            "a time-based holdout needs a pre-period to serve as the counterfactual; "
            "with n_pre_periods=0 there is nothing to measure the lift against"
        )

    se_factor = math.sqrt(1 / n_test_periods + 1 / n_pre_periods)
    return _effect_multiplier(alpha, power, one_sided) * cv * se_factor


def required_duration_for_mde(
    target_mde: float,
    cv: float,
    *,
    alpha: float = 0.10,
    power: float = 0.80,
    one_sided: bool = True,
    pre_ratio: float = 2.0,
) -> int:
    """Test-window length needed to detect ``target_mde`` in a time-based holdout."""
    if target_mde <= 0:
        raise ValueError("target_mde must be positive")
    k = _effect_multiplier(alpha, power, one_sided)
    # mde = k*cv*sqrt(1/n + 1/(pre_ratio*n))
    n = ((k * cv / target_mde) ** 2) * (1 + 1 / pre_ratio)
    return max(1, math.ceil(n))


def estimate_geo_cv(
    df: pd.DataFrame, *, geo_column: str, target_column: str, date_column: Optional[str] = None
) -> float:
    """Cross-geo coefficient of variation of per-geo mean outcome.

    This is the noise a geo test has to overcome. Computed on geo means so that
    a geo with more observations does not look less variable than it is.
    """
    if geo_column not in df.columns or target_column not in df.columns:
        raise KeyError("geo_column and target_column must both be present")
    per_geo = df.groupby(geo_column)[target_column].mean()
    if len(per_geo) < 2 or per_geo.mean() == 0:
        return 0.0
    return float(per_geo.std(ddof=1) / abs(per_geo.mean()))


def estimate_pre_period_correlation(
    df: pd.DataFrame,
    *,
    geo_column: str,
    target_column: str,
    date_column: str,
    split_date: Optional[str] = None,
) -> float:
    """Correlation between each geo's outcome before and after a split point.

    This is the single most important input to geo-test power, and it is
    measurable from the panel you already have: split the history in two, take
    each geo's mean outcome in each half, and correlate them across geos. Market
    sizes are persistent, so this is usually high (0.9+), which is precisely why
    a pre-period-adjusted design detects effects an unadjusted one cannot.
    """
    from .discovery import parse_date_column

    for col in (geo_column, target_column, date_column):
        if col not in df.columns:
            raise KeyError(f"'{col}' not in dataframe")

    parsed, _, _ = parse_date_column(df[date_column])
    work = pd.DataFrame({
        "geo": df[geo_column].to_numpy(),
        "date": parsed.to_numpy(),
        "y": pd.to_numeric(df[target_column], errors="coerce").to_numpy(),
    }).dropna()
    if work.empty:
        return 0.0

    cut = pd.Timestamp(split_date) if split_date else work["date"].quantile(0.5)
    pre = work[work["date"] < cut].groupby("geo")["y"].mean()
    post = work[work["date"] >= cut].groupby("geo")["y"].mean()
    joined = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna()
    if len(joined) < 3 or joined["pre"].std(ddof=1) == 0 or joined["post"].std(ddof=1) == 0:
        return 0.0
    return float(joined["pre"].corr(joined["post"]))


def estimate_period_cv(
    df: pd.DataFrame, *, target_column: str, date_column: str, detrend: bool = True
) -> float:
    """Period-to-period coefficient of variation, optionally after removing trend.

    Trend inflates raw variance and would make a time-based test look far less
    powerful than it is, because the counterfactual accounts for trend anyway.
    """
    from .discovery import parse_date_column

    parsed, _, _ = parse_date_column(df[date_column])
    series = (
        pd.DataFrame({"d": parsed, "y": pd.to_numeric(df[target_column], errors="coerce")})
        .dropna()
        .sort_values("d")["y"]
        .to_numpy()
    )
    if len(series) < 3 or series.mean() == 0:
        return 0.0
    if detrend and len(series) >= 4:
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        residual = series - (slope * x + intercept)
        return float(np.std(residual, ddof=1) / abs(series.mean()))
    return float(np.std(series, ddof=1) / abs(series.mean()))


def expected_lift_from_share(
    contribution_share: float, holdout_fraction: float = 1.0, *, carryover_recovery: float = 0.85
) -> float:
    """Lift the model predicts a holdout will produce, as a fraction of the outcome.

    If a channel is believed to drive 6% of sales and we switch off all of it,
    a perfect test would show a 6% drop — except that carryover from before the
    test keeps working for part of the window. ``carryover_recovery`` (< 1)
    reflects that a short test measures less than the full steady-state effect.
    Ignoring it is the standard way a test gets designed 15-30% underpowered.
    """
    if not 0 <= contribution_share <= 1:
        raise ValueError("contribution_share must be a fraction between 0 and 1")
    if not 0 < holdout_fraction <= 1:
        raise ValueError("holdout_fraction must be in (0, 1]")
    return contribution_share * holdout_fraction * carryover_recovery


# --------------------------------------------------------------------------- #
# Prioritisation
# --------------------------------------------------------------------------- #
@dataclass
class ChannelTestCandidate:
    """A channel considered for testing, with the evidence for its priority."""

    channel: str
    spend: float
    spend_share: float = 0.0
    contribution_share: Optional[float] = None
    roas_point: Optional[float] = None
    roas_ci_low: Optional[float] = None
    roas_ci_high: Optional[float] = None
    always_on: bool = False
    geo_testable: bool = True
    has_prior_experiment: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def uncertainty(self) -> float:
        """How little we know, on a 0-1 scale.

        Interval width relative to the point estimate, capped. A channel whose
        89% interval spans 0.5 to 5.0 has not been measured, and that is exactly
        the channel a test is worth spending on.
        """
        if self.roas_ci_low is not None and self.roas_ci_high is not None and self.roas_point:
            width = (self.roas_ci_high - self.roas_ci_low) / abs(self.roas_point)
            return float(min(1.0, width / 3.0))
        # No fitted model yet: always-on channels are the least identified.
        return 0.9 if self.always_on else 0.6

    @property
    def feasibility(self) -> float:
        """How practical a clean test is, on a 0-1 scale."""
        score = 1.0
        if not self.geo_testable:
            # National-only buying (linear TV in many markets) leaves only
            # time-based designs, which are much weaker.
            score *= 0.4
        if self.always_on:
            # Turning off an always-on channel has organisational cost and
            # brand risk, so these tests get blocked more often than they fail.
            score *= 0.7
        return score

    @property
    def value_of_information(self) -> float:
        """Money at risk times how little we know about it."""
        return self.spend_share * self.uncertainty

    @property
    def priority(self) -> float:
        score = self.value_of_information * self.feasibility
        if self.has_prior_experiment:
            # Already anchored: a second test on the same channel buys less than
            # a first test on an unanchored one.
            score *= 0.4
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": round(self.spend, 2),
            "spend_share": round(self.spend_share, 4),
            "uncertainty": round(self.uncertainty, 3),
            "feasibility": round(self.feasibility, 3),
            "value_of_information": round(self.value_of_information, 4),
            "priority": round(self.priority, 4),
            "always_on": self.always_on,
            "geo_testable": self.geo_testable,
            "has_prior_experiment": self.has_prior_experiment,
            "roas_point": self.roas_point,
            "roas_ci_low": self.roas_ci_low,
            "roas_ci_high": self.roas_ci_high,
            "notes": self.notes,
        }


def prioritise_channels(candidates: Iterable[ChannelTestCandidate]) -> list[ChannelTestCandidate]:
    """Rank channels by what a test on each would be worth."""
    items = list(candidates)
    total = sum(c.spend for c in items)
    for c in items:
        c.spend_share = c.spend / total if total else 0.0
    return sorted(items, key=lambda c: -c.priority)


# --------------------------------------------------------------------------- #
# Designs
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentPlan:
    """One proposed experiment, complete enough to hand to whoever runs it."""

    channel: str
    design: str
    rank: int = 0
    priority: float = 0.0

    # Design
    n_treatment_geos: int = 0
    n_control_geos: int = 0
    duration_periods: int = 0
    pre_period_periods: int = 0
    holdout_fraction: float = 1.0
    spend_at_risk: float = 0.0

    # What it can see, and what we think it will see
    power: Optional[PowerResult] = None
    expected_result: str = ""
    falsifies: str = ""

    execution: list[str] = field(default_factory=list)
    feeds_back_as: str = ""
    risks: list[str] = field(default_factory=list)
    blocked_reason: str = ""

    @property
    def viable(self) -> bool:
        return not self.blocked_reason and bool(self.power and self.power.powered is not False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "design": self.design,
            "rank": self.rank,
            "priority": round(self.priority, 4),
            "n_treatment_geos": self.n_treatment_geos,
            "n_control_geos": self.n_control_geos,
            "duration_periods": self.duration_periods,
            "pre_period_periods": self.pre_period_periods,
            "holdout_fraction": self.holdout_fraction,
            "spend_at_risk": round(self.spend_at_risk, 2),
            "power": self.power.to_dict() if self.power else None,
            "expected_result": self.expected_result,
            "falsifies": self.falsifies,
            "execution": self.execution,
            "feeds_back_as": self.feeds_back_as,
            "risks": self.risks,
            "blocked_reason": self.blocked_reason,
            "viable": self.viable,
        }


def design_geo_holdout(
    candidate: ChannelTestCandidate,
    *,
    n_geos_available: int,
    geo_cv: float,
    duration_periods: int = 8,
    pre_period_periods: int = 12,
    holdout_fraction: float = 1.0,
    treatment_share: float = 0.5,
    pre_period_correlation: float = 0.9,
    residual_autocorrelation: float = 0.3,
    alpha: float = 0.10,
    power: float = 0.80,
    periods_per_year: int = 52,
) -> ExperimentPlan:
    """Design a geo holdout for one channel and say honestly whether it can work."""
    plan = ExperimentPlan(
        channel=candidate.channel,
        design="geo_holdout",
        priority=candidate.priority,
        duration_periods=duration_periods,
        pre_period_periods=pre_period_periods,
        holdout_fraction=holdout_fraction,
    )

    if not candidate.geo_testable:
        plan.design = "time_holdout"
        plan.blocked_reason = (
            f"{candidate.channel} cannot be bought by geo, so a geo design is unavailable. A "
            "time-based on/off test is the fallback, and it is much weaker: it cannot separate the "
            "switch-off from anything else that changed in the same weeks."
        )

    n_treatment = max(1, int(round(n_geos_available * treatment_share)))
    n_control = max(1, n_geos_available - n_treatment)
    plan.n_treatment_geos = n_treatment
    plan.n_control_geos = n_control

    expected = (
        expected_lift_from_share(candidate.contribution_share, holdout_fraction)
        if candidate.contribution_share is not None
        else None
    )

    if candidate.geo_testable:
        mde = geo_lift_mde(
            n_treatment, n_control, geo_cv,
            alpha=alpha, power=power,
            pre_period_correlation=pre_period_correlation,
            duration_periods=duration_periods,
            residual_autocorrelation=residual_autocorrelation,
        )
        design_label = "geo_holdout"
    else:
        if pre_period_periods < 1:
            plan.blocked_reason = (
                f"{candidate.channel} cannot be geo-tested and no pre-period was specified. A "
                "time-based holdout with nothing to compare against cannot estimate a lift at any "
                "sample size."
            )
            plan.power = PowerResult(
                design="time_holdout",
                mde_relative=float("inf"),
                expected_lift_relative=expected,
                alpha=alpha,
                power=power,
                duration_periods=duration_periods,
                cv=geo_cv,
                notes=["No pre-period: there is no counterfactual to measure the lift against."],
            )
            _write_expectation(plan, candidate, expected, float("inf"))
            _write_execution(plan, candidate, duration_periods, pre_period_periods)
            return plan
        mde = holdout_mde(
            duration_periods, geo_cv, n_pre_periods=pre_period_periods, alpha=alpha, power=power
        )
        design_label = "time_holdout"

    result = PowerResult(
        design=design_label,
        mde_relative=mde,
        expected_lift_relative=expected,
        alpha=alpha,
        power=power,
        n_treatment=n_treatment,
        n_control=n_control,
        duration_periods=duration_periods,
        cv=geo_cv,
        pre_period_correlation=pre_period_correlation if candidate.geo_testable else 0.0,
    )

    if candidate.geo_testable and pre_period_correlation > 0:
        result.notes.append(
            f"Assumes a pre-period covariate correlated at {pre_period_correlation:.2f} — measure "
            f"it with `estimate_pre_period_correlation` rather than trusting this default. Without "
            f"that adjustment the detectable effect is "
            f"{geo_lift_mde(n_treatment, n_control, geo_cv, alpha=alpha, power=power, duration_periods=duration_periods, residual_autocorrelation=residual_autocorrelation):.1%}, "
            f"not {mde:.1%} — so building the matched pre-period is not optional."
        )

    if result.powered is False:
        needed_t, needed_c = required_geos_for_mde(
            expected, geo_cv, alpha=alpha, power=power,
            pre_period_correlation=pre_period_correlation,
            duration_periods=duration_periods,
            residual_autocorrelation=residual_autocorrelation,
        )
        result.notes.append(
            f"Underpowered: the model expects a {expected:.1%} lift but the smallest effect this "
            f"design could detect is {mde:.1%}. Detecting {expected:.1%} needs about {needed_t} "
            f"treatment and {needed_c} control geos, against {n_geos_available} available."
        )
        plan.blocked_reason = plan.blocked_reason or (
            "As designed this test would most likely return a null result whether or not the "
            "channel works, which is worse than not running it: a null gets read as 'the channel "
            "does nothing' and the budget moves on bad evidence."
        )

    plan.power = result
    plan.spend_at_risk = candidate.spend * holdout_fraction * (
        duration_periods / periods_per_year
    ) * (n_treatment / max(1, n_geos_available))

    _write_expectation(plan, candidate, expected, mde)
    _write_execution(plan, candidate, duration_periods, pre_period_periods)
    return plan


def _write_expectation(
    plan: ExperimentPlan,
    candidate: ChannelTestCandidate,
    expected: Optional[float],
    mde: float,
) -> None:
    """State the prediction before the test runs, so the test can fail."""
    if expected is None:
        plan.expected_result = (
            f"No fitted contribution for {candidate.channel} yet, so there is no pre-registered "
            f"expectation. This design can detect a lift of {mde:.1%} or larger. Write down the "
            "expected effect before the test starts — a test with no prediction cannot be wrong, "
            "and therefore teaches nothing."
        )
        plan.falsifies = "Nothing, until an expected effect is recorded."
        return

    roas = candidate.roas_point
    roas_text = f" (implied ROAS {roas:.2f})" if roas else ""
    plan.expected_result = (
        f"If the current model is right, holding out {plan.holdout_fraction:.0%} of "
        f"{candidate.channel} should reduce the outcome in treated geos by about "
        f"{expected:.1%}{roas_text} relative to control, over {plan.duration_periods} periods. "
        f"The design detects {mde:.1%} or larger, so the expected effect is "
        f"{'comfortably above' if expected >= 2 * mde else 'above' if expected >= mde else 'below'} "
        "the detection floor."
    )
    plan.falsifies = (
        f"A measured lift near zero (below {mde:.1%}) would say {candidate.channel} is materially "
        f"less incremental than the model believes — the model's estimate would have to come down "
        f"towards the test. A lift well above {expected:.1%} would say the model is crediting some "
        f"of {candidate.channel}'s effect to another channel or to the baseline, most likely one "
        "it correlates with."
    )


def _write_execution(
    plan: ExperimentPlan, candidate: ChannelTestCandidate, duration: int, pre_periods: int
) -> None:
    if plan.design == "geo_holdout":
        plan.execution = [
            f"Split {plan.n_treatment_geos + plan.n_control_geos} geos into "
            f"{plan.n_treatment_geos} treatment and {plan.n_control_geos} control. Stratify on "
            "outcome level and on the channel's own spend so the arms are balanced on both, and "
            "randomise within strata rather than hand-picking markets.",
            f"Record {pre_periods} periods of pre-test history for every geo before touching the "
            "buy. This is the covariate that makes the test powerful; it cannot be reconstructed "
            "afterwards.",
            f"Suppress {plan.holdout_fraction:.0%} of {candidate.channel} in treatment geos for "
            f"{duration} periods. Change nothing else — no creative refresh, no promotion, no "
            "budget shift into other channels to 'compensate', which is the most common way a "
            "media test is destroyed.",
            "Verify suppression in-flight: pull delivery by geo in week one and confirm the "
            "treatment geos actually went dark. Partial suppression looks exactly like a weak "
            "effect.",
            "Hold the plan for the full window even if early numbers look bad. Stopping early on a "
            "noisy read is how a test becomes a story.",
            f"Analyse as a difference-in-differences against the pre-period, or CUPED-adjust on the "
            f"pre-period outcome. Report the lift with its standard error, not a p-value alone — "
            "the MMM needs the uncertainty to calibrate against.",
        ]
        plan.risks = [
            "Spillover: adjacent geos and national media contaminate the control arm. Prefer "
            "geographically separated markets and check for cross-border delivery.",
            "Carryover from before the test keeps working during it, so a short window understates "
            "the true effect. This is already discounted in the expected result.",
            "A concurrent promotion or competitor action in one arm will be read as media effect.",
        ]
    else:
        plan.execution = [
            f"Establish {pre_periods} periods of stable pre-test history with no other planned "
            "changes.",
            f"Switch {candidate.channel} off nationally for {duration} periods.",
            "Document everything else happening in the window — promotions, competitor activity, "
            "seasonality, PR. A national on/off test has no control group, so the counterfactual "
            "is entirely an argument, and every unrecorded change weakens it.",
            "Analyse against a seasonally-adjusted counterfactual (a synthetic control from "
            "unaffected products or markets is stronger than a naive pre/post).",
        ]
        plan.risks = [
            "No control group: anything else that moved in the window is confounded with the test.",
            "A national switch-off has real revenue and brand cost, and is hard to get approved.",
            "Seasonality can easily exceed the effect being measured.",
        ]

    plan.feeds_back_as = (
        f"Feed the measured lift and its standard error into the model as a calibration "
        f"constraint on {candidate.channel} — `add_lift_test_measurements` in pymc-marketing, an "
        "ROI prior in Meridian, `calibration_input` in Robyn. Refit and check that the channel's "
        "posterior moved towards the test and that the other channels' contributions did not "
        "quietly absorb the difference."
    )


# --------------------------------------------------------------------------- #
# Roadmap
# --------------------------------------------------------------------------- #
@dataclass
class Roadmap:
    plans: list[ExperimentPlan] = field(default_factory=list)
    waves: list[list[str]] = field(default_factory=list)
    total_spend_at_risk: float = 0.0
    geo_cv: float = 0.0
    n_geos_available: int = 0
    warnings: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    @property
    def viable_plans(self) -> list[ExperimentPlan]:
        return [p for p in self.plans if p.viable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plans": [p.to_dict() for p in self.plans],
            "waves": self.waves,
            "total_spend_at_risk": round(self.total_spend_at_risk, 2),
            "geo_cv": round(self.geo_cv, 4),
            "n_geos_available": self.n_geos_available,
            "warnings": self.warnings,
            "questions": self.questions,
        }


def build_roadmap(
    candidates: Iterable[ChannelTestCandidate],
    *,
    n_geos_available: int,
    geo_cv: float,
    duration_periods: int = 8,
    pre_period_periods: int = 12,
    max_concurrent: int = 1,
    max_tests: int = 6,
    periods_per_year: int = 52,
    **design_kwargs: Any,
) -> Roadmap:
    """Prioritise channels, design a test for each, and sequence them into waves.

    ``max_concurrent`` defaults to 1 because two geo tests running at once in
    overlapping markets contaminate each other, and the second one's control arm
    is no longer clean. Raise it only when the tests use disjoint geo sets.
    """
    ranked = prioritise_channels(candidates)
    roadmap = Roadmap(geo_cv=geo_cv, n_geos_available=n_geos_available)

    # Below roughly 10 markets per arm, the estimate is dominated by which markets
    # happened to land in which arm rather than by the treatment.
    MIN_GEOS_PER_ARM = 10
    if n_geos_available < 2 * MIN_GEOS_PER_ARM:
        roadmap.warnings.append(
            f"Only {n_geos_available} geos available, giving about {n_geos_available // 2} per "
            f"arm. Geo designs need roughly {MIN_GEOS_PER_ARM} per arm to average out "
            "cross-market noise; below that the result is mostly measuring which markets "
            "happened to be picked. Consider a matched-market design or a longer window."
        )

    for rank, candidate in enumerate(ranked[:max_tests], start=1):
        plan = design_geo_holdout(
            candidate,
            n_geos_available=n_geos_available,
            geo_cv=geo_cv,
            duration_periods=duration_periods,
            pre_period_periods=pre_period_periods,
            periods_per_year=periods_per_year,
            **design_kwargs,
        )
        plan.rank = rank
        roadmap.plans.append(plan)

    roadmap.total_spend_at_risk = sum(p.spend_at_risk for p in roadmap.viable_plans)

    viable = [p.channel for p in roadmap.viable_plans]
    roadmap.waves = [viable[i : i + max_concurrent] for i in range(0, len(viable), max_concurrent)]

    blocked = [p for p in roadmap.plans if p.blocked_reason]
    if blocked:
        roadmap.warnings.append(
            f"{len(blocked)} of {len(roadmap.plans)} proposed tests cannot be run as designed: "
            + "; ".join(f"{p.channel} ({p.design})" for p in blocked)
        )

    unpowered = [p for p in roadmap.plans if p.power and p.power.powered is False]
    if unpowered:
        roadmap.questions.append(
            "For "
            + ", ".join(p.channel for p in unpowered)
            + ": can we get more geos, run for longer, or hold out a larger share of spend? Any of "
            "the three buys power, and running the test as currently designed does not."
        )

    no_expectation = [p for p in roadmap.plans if p.power and p.power.expected_lift_relative is None]
    if no_expectation:
        roadmap.questions.append(
            "No fitted contribution exists yet for "
            + ", ".join(p.channel for p in no_expectation)
            + ". Fit the model first, or record an explicit prior expectation — otherwise the test "
            "result has nothing to contradict."
        )

    roadmap.questions.append(
        "What is the maximum revenue the business will accept putting at risk, and who signs it "
        "off? Every design below trades revenue for information, and that trade is a business "
        "decision, not a modelling one."
    )
    roadmap.questions.append(
        "Is there a period in the calendar where a holdout is unacceptable (peak trading, a "
        "launch, a seasonal spike)? Test windows have to avoid those, and that usually constrains "
        "the roadmap more than statistics does."
    )
    return roadmap


def render_roadmap(roadmap: Roadmap) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Experiment Roadmap")
    a("")
    a(f"{len(roadmap.viable_plans)} of {len(roadmap.plans)} proposed tests are viable as designed. "
      f"Total revenue at risk across viable tests: {roadmap.total_spend_at_risk:,.0f}.")
    a("")
    a(f"* Geos available: {roadmap.n_geos_available}")
    a(f"* Cross-geo coefficient of variation: {roadmap.geo_cv:.2f}")
    a("")
    a("An MMM without an experiment is an argument from correlation. Each test below converts one "
      "channel from an assumption into a measurement, and the ranking is by how much that "
      "conversion is worth: money at risk, times how little we currently know, times how cleanly "
      "the channel can be tested.")
    a("")

    if roadmap.waves:
        a("## Sequence")
        a("")
        for i, wave in enumerate(roadmap.waves, start=1):
            a(f"{i}. **Wave {i}** — {', '.join(wave)}")
        a("")
        a("Waves run in sequence, not in parallel: two geo tests in overlapping markets contaminate "
          "each other's control arm.")
        a("")

    for plan in roadmap.plans:
        status = "BLOCKED" if plan.blocked_reason else "READY"
        a(f"## {plan.rank}. {plan.channel} — {plan.design} [{status}]")
        a("")
        if plan.blocked_reason:
            a(f"> **Not runnable as designed.** {plan.blocked_reason}")
            a("")

        p = plan.power
        a("| | |")
        a("|---|---|")
        a(f"| Design | {plan.design} |")
        if plan.design == "geo_holdout":
            a(f"| Geos | {plan.n_treatment_geos} treatment vs {plan.n_control_geos} control |")
        a(f"| Duration | {plan.duration_periods} periods (plus {plan.pre_period_periods} pre) |")
        a(f"| Holdout | {plan.holdout_fraction:.0%} of the channel |")
        a(f"| Revenue at risk | {plan.spend_at_risk:,.0f} |")
        if p:
            a(f"| Detectable effect | {p.mde_relative:.1%} |")
            if p.expected_lift_relative is not None:
                a(f"| Expected effect | {p.expected_lift_relative:.1%} |")
                a(f"| Powered | {'yes' if p.powered else 'no'} |")
        a("")

        a("**What we expect**")
        a("")
        a(plan.expected_result)
        a("")
        a("**What would change our mind**")
        a("")
        a(plan.falsifies)
        a("")

        if p and p.notes:
            a("**Power notes**")
            a("")
            for n in p.notes:
                a(f"* {n}")
            a("")

        a("**How to run it**")
        a("")
        for i, step in enumerate(plan.execution, start=1):
            a(f"{i}. {step}")
        a("")

        if plan.risks:
            a("**Risks**")
            a("")
            for r in plan.risks:
                a(f"* {r}")
            a("")

        a("**How it feeds back into the model**")
        a("")
        a(plan.feeds_back_as)
        a("")

    if roadmap.warnings:
        a("## Warnings")
        a("")
        for w in roadmap.warnings:
            a(f"* {w}")
        a("")

    if roadmap.questions:
        a("## Questions to settle before committing")
        a("")
        for q in roadmap.questions:
            a(f"* {q}")
        a("")

    return "\n".join(lines)
