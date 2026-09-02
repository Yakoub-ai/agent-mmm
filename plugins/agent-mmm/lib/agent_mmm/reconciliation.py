"""Reconciling modelled spend against the finance ledger.

Platform exports and the general ledger disagree, always, and the size of the
disagreement is diagnostic:

* a small, stable gap is usually agency fees, ad-serving costs or VAT;
* a gap that grows is usually a channel missing from the export;
* a gap in one direction in one period is usually a rebate, a credit or a
  restatement;
* a gap that flips sign is usually a date-basis mismatch — the platform reports
  on delivery, finance on invoice.

Whichever it is, an MMM built on media numbers that finance does not recognise
will not survive its first review with the CFO, and it should not: if the model
cannot account for the money, its ROAS is measuring something other than the
money.

This module does not fix the difference. It measures it, classifies its shape,
and says what to ask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "PeriodVariance",
    "ReconciliationReport",
    "reconcile_spend",
    "render_reconciliation_report",
]


@dataclass
class PeriodVariance:
    period: str
    modelled: float
    finance: float

    @property
    def difference(self) -> float:
        return self.modelled - self.finance

    @property
    def pct(self) -> float:
        return self.difference / self.finance if self.finance else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "modelled": round(self.modelled, 2),
            "finance": round(self.finance, 2),
            "difference": round(self.difference, 2),
            "pct": round(self.pct, 4) if np.isfinite(self.pct) else None,
        }


@dataclass
class ReconciliationReport:
    modelled_total: float = 0.0
    finance_total: float = 0.0
    n_periods: int = 0
    n_within_tolerance: int = 0
    tolerance: float = 0.02
    periods: list[PeriodVariance] = field(default_factory=list)
    worst: list[PeriodVariance] = field(default_factory=list)
    missing_in_finance: list[str] = field(default_factory=list)
    missing_in_modelled: list[str] = field(default_factory=list)
    pattern: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    @property
    def total_difference(self) -> float:
        return self.modelled_total - self.finance_total

    @property
    def total_pct(self) -> float:
        return self.total_difference / self.finance_total if self.finance_total else float("nan")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelled_total": round(self.modelled_total, 2),
            "finance_total": round(self.finance_total, 2),
            "total_difference": round(self.total_difference, 2),
            "total_pct": round(self.total_pct, 4) if np.isfinite(self.total_pct) else None,
            "n_periods": self.n_periods,
            "n_within_tolerance": self.n_within_tolerance,
            "tolerance": self.tolerance,
            "pattern": self.pattern,
            "worst": [p.to_dict() for p in self.worst],
            "missing_in_finance": self.missing_in_finance,
            "missing_in_modelled": self.missing_in_modelled,
            "errors": self.errors,
            "warnings": self.warnings,
            "questions": self.questions,
            "ok": self.ok,
        }


def _classify_pattern(variances: list[PeriodVariance], tolerance: float) -> str:
    """Name the shape of the disagreement — the shape tells you the cause."""
    finite = [v for v in variances if np.isfinite(v.pct)]
    if not finite:
        return "unknown"

    pcts = np.array([v.pct for v in finite])
    within = np.abs(pcts) <= tolerance
    if within.all():
        return "aligned"

    outside = pcts[~within]
    if len(outside) == 0:
        return "aligned"

    # A constant proportional gap: fees or tax, applied uniformly.
    if np.all(outside < 0) and float(np.std(pcts)) < 0.02:
        return "constant_shortfall"
    if np.all(outside > 0) and float(np.std(pcts)) < 0.02:
        return "constant_excess"

    # A gap that grows with time: a channel that was added and never mapped.
    idx = np.arange(len(pcts))
    if len(pcts) >= 6:
        slope = float(np.polyfit(idx, pcts, 1)[0])
        if abs(slope) * len(pcts) > 3 * tolerance:
            return "drifting"

    # Sign flips concentrated in adjacent periods: a timing/date-basis problem.
    signs = np.sign(pcts[~within])
    if len(signs) >= 4 and len(set(signs.tolist())) > 1:
        # Do the deviations roughly cancel? That is timing, not a missing channel.
        if abs(float(np.sum(outside))) < 0.4 * float(np.sum(np.abs(outside))):
            return "timing_shift"

    if (~within).sum() <= max(1, int(0.1 * len(pcts))):
        return "isolated_spikes"
    return "unstructured"


_PATTERN_GUIDANCE: dict[str, tuple[str, str]] = {
    "aligned": (
        "Modelled spend and the ledger agree within tolerance in every period.",
        "",
    ),
    "constant_shortfall": (
        "Modelled spend is consistently below the ledger by a near-constant proportion.",
        "This is the signature of agency fees, ad-serving or tech costs, or VAT sitting in the "
        "ledger but not in the platform export. Decide deliberately whether the model should see "
        "gross or net spend — and use the same basis in the ROAS you report, or every number is "
        "off by the fee rate.",
    ),
    "constant_excess": (
        "Modelled spend is consistently above the ledger by a near-constant proportion.",
        "Usually double-counting: the same campaigns appearing in two exports, or a platform "
        "reporting in a different currency or including tax the ledger excludes.",
    ),
    "drifting": (
        "The gap grows steadily across the window.",
        "Almost always a channel that started mid-window and was never added to the export, or a "
        "renamed campaign silently dropping out of the taxonomy mapping. Find the period the drift "
        "starts and look at what launched.",
    ),
    "timing_shift": (
        "The gap alternates sign and roughly cancels out over the window.",
        "A date-basis mismatch: the platform reports on delivery, the ledger on invoice or "
        "payment. The totals are right and the periods are wrong, which is worse for an MMM than a "
        "level difference — it corrupts carryover estimation directly. Reconcile on delivery date.",
    ),
    "isolated_spikes": (
        "A small number of periods disagree sharply while the rest align.",
        "Rebates, credits, make-goods or a restated invoice. Identify each one; do not smooth them "
        "away.",
    ),
    "unstructured": (
        "The differences have no consistent shape.",
        "This usually means the two sources are not measuring the same thing at all — different "
        "scope, different entities, or a broken join. Re-check the join keys before anything else.",
    ),
    "unknown": ("Not enough overlapping periods to characterise the difference.", ""),
}


def reconcile_spend(
    modelled: pd.DataFrame,
    finance: pd.DataFrame,
    *,
    date_column: str = "date",
    modelled_spend_columns: Optional[list[str]] = None,
    finance_spend_column: str = "spend",
    tolerance: float = 0.02,
    freq: str = "W-MON",
    max_total_variance: float = 0.05,
) -> ReconciliationReport:
    """Compare period-level modelled spend against the finance ledger.

    ``tolerance`` is the per-period band treated as agreement; ``max_total_variance``
    is the whole-window gap above which the reconciliation fails outright.
    """
    from .discovery import parse_date_column

    report = ReconciliationReport(tolerance=tolerance)

    if date_column not in modelled.columns:
        raise KeyError(f"'{date_column}' not in modelled frame")
    if date_column not in finance.columns:
        raise KeyError(f"'{date_column}' not in finance frame")
    if finance_spend_column not in finance.columns:
        raise KeyError(f"'{finance_spend_column}' not in finance frame")

    if modelled_spend_columns is None:
        modelled_spend_columns = [
            c for c in modelled.columns
            if c != date_column and pd.api.types.is_numeric_dtype(modelled[c])
        ]
    missing = [c for c in modelled_spend_columns if c not in modelled.columns]
    if missing:
        raise KeyError(f"modelled spend columns not found: {missing}")
    if not modelled_spend_columns:
        report.errors.append("No numeric spend columns to reconcile.")
        return report

    period = _freq_to_period(freq)

    m_dates, _, _ = parse_date_column(modelled[date_column])
    m = pd.DataFrame({
        "period": m_dates.dt.to_period(period).dt.start_time,
        "modelled": modelled[modelled_spend_columns].apply(
            lambda c: pd.to_numeric(c, errors="coerce")
        ).fillna(0.0).sum(axis=1),
    }).dropna(subset=["period"]).groupby("period", as_index=False)["modelled"].sum()

    f_dates, _, _ = parse_date_column(finance[date_column])
    f = pd.DataFrame({
        "period": f_dates.dt.to_period(period).dt.start_time,
        "finance": pd.to_numeric(finance[finance_spend_column], errors="coerce").fillna(0.0),
    }).dropna(subset=["period"]).groupby("period", as_index=False)["finance"].sum()

    joined = m.merge(f, on="period", how="outer", indicator=True).sort_values("period")
    report.missing_in_finance = [
        str(p.date()) for p in joined.loc[joined["_merge"] == "left_only", "period"]
    ]
    report.missing_in_modelled = [
        str(p.date()) for p in joined.loc[joined["_merge"] == "right_only", "period"]
    ]

    both = joined[joined["_merge"] == "both"]
    report.modelled_total = float(m["modelled"].sum())
    report.finance_total = float(f["finance"].sum())
    report.n_periods = len(both)

    report.periods = [
        PeriodVariance(period=str(r.period.date()), modelled=float(r.modelled), finance=float(r.finance))
        for r in both.itertuples()
    ]
    report.n_within_tolerance = sum(
        1 for v in report.periods if np.isfinite(v.pct) and abs(v.pct) <= tolerance
    )
    report.worst = sorted(
        [v for v in report.periods if np.isfinite(v.pct)],
        key=lambda v: -abs(v.pct),
    )[:10]
    report.pattern = _classify_pattern(report.periods, tolerance)

    _assess(report, max_total_variance)
    return report


def _assess(report: ReconciliationReport, max_total_variance: float) -> None:
    if report.n_periods == 0:
        report.errors.append(
            "The two sources share no periods. Either the date columns are on different bases or "
            "the join is wrong — there is nothing to reconcile until that is fixed."
        )
        return

    if np.isfinite(report.total_pct) and abs(report.total_pct) > max_total_variance:
        report.errors.append(
            f"Total modelled spend differs from the ledger by {report.total_pct:+.1%} "
            f"({report.total_difference:+,.0f}). Above {max_total_variance:.0%} the model is not "
            "describing the money the business actually spent, and every ROAS derived from it "
            "inherits the error."
        )

    share_ok = report.n_within_tolerance / report.n_periods
    if share_ok < 0.9:
        report.warnings.append(
            f"Only {share_ok:.0%} of periods agree within {report.tolerance:.0%}. Period-level "
            "disagreement corrupts carryover estimation even when the totals match."
        )

    if report.missing_in_modelled:
        report.errors.append(
            f"{len(report.missing_in_modelled)} period(s) have ledger spend but none in the model "
            f"(first: {report.missing_in_modelled[0]}). That is media the model cannot see."
        )
    if report.missing_in_finance:
        report.warnings.append(
            f"{len(report.missing_in_finance)} period(s) have modelled spend but nothing in the "
            f"ledger (first: {report.missing_in_finance[0]}). Check the ledger extract covers the "
            "full window."
        )

    headline, guidance = _PATTERN_GUIDANCE.get(report.pattern, ("", ""))
    if guidance:
        report.warnings.append(f"{headline} {guidance}")

    if report.pattern == "constant_shortfall":
        report.questions.append(
            "Does the ledger include agency fees, ad-serving or tax that the platform export "
            "excludes? Should the model see gross or net media cost?"
        )
    elif report.pattern == "constant_excess":
        report.questions.append(
            "Is any campaign counted in two exports, or is a platform reporting in a different "
            "currency or tax basis than the ledger?"
        )
    elif report.pattern == "drifting":
        first_bad = next(
            (v.period for v in report.periods if np.isfinite(v.pct) and abs(v.pct) > report.tolerance),
            None,
        )
        report.questions.append(
            f"What changed around {first_bad}? A new channel, a renamed campaign or a new agency "
            "would all produce this drift."
        )
    elif report.pattern == "timing_shift":
        report.questions.append(
            "Is the ledger on an invoice or payment basis while the platform reports on delivery? "
            "The model needs delivery dates — carryover is a claim about when the impression "
            "landed, not when it was paid for."
        )
    elif report.pattern == "isolated_spikes":
        worst = report.worst[:3]
        report.questions.append(
            "What happened in "
            + ", ".join(f"{v.period} ({v.pct:+.0%})" for v in worst)
            + "? Rebates and make-goods should be recorded, not smoothed."
        )

    report.questions.append(
        "Who signs off that these media numbers are the ones finance recognises? Get that "
        "agreement before the model is built, not when the results are presented."
    )


def _freq_to_period(freq: str) -> str:
    f = freq.upper()
    if f.startswith("W"):
        return "W"
    if f.startswith("M"):
        return "M"
    if f.startswith("Q"):
        return "Q"
    return "D"


def render_reconciliation_report(report: ReconciliationReport) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Spend Reconciliation")
    a("")
    a(f"**{'PASS' if report.ok else 'FAIL'}** — modelled {report.modelled_total:,.0f} vs ledger "
      f"{report.finance_total:,.0f} ({report.total_pct:+.2%})")
    a("")
    headline, guidance = _PATTERN_GUIDANCE.get(report.pattern, ("", ""))
    a(f"**Pattern: `{report.pattern}`** — {headline}")
    if guidance:
        a("")
        a(guidance)
    a("")
    a(f"* Periods compared: {report.n_periods}")
    a(f"* Within {report.tolerance:.0%} tolerance: {report.n_within_tolerance} "
      f"({report.n_within_tolerance / report.n_periods:.0%})" if report.n_periods else "")
    a("")

    if report.errors:
        a("## Errors")
        a("")
        for e in report.errors:
            a(f"* {e}")
        a("")

    if report.worst:
        a("## Largest variances")
        a("")
        a("| Period | Modelled | Ledger | Difference | % |")
        a("|---|---:|---:|---:|---:|")
        for v in report.worst:
            a(f"| {v.period} | {v.modelled:,.0f} | {v.finance:,.0f} | {v.difference:+,.0f} | "
              f"{v.pct:+.1%} |")
        a("")

    if report.warnings:
        a("## Warnings")
        a("")
        for w in report.warnings:
            a(f"* {w}")
        a("")

    if report.questions:
        a("## Questions")
        a("")
        for q in report.questions:
            a(f"* {q}")
        a("")

    return "\n".join(lines)
