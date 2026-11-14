"""Data-quality advisor: turn an audit into a prioritised improvement plan.

The audit (``data_audit.run_audit``) says what is wrong with *this* dataset.
The advisor answers the two questions that come next:

1. **What do we do about it now?** — the fix for the model being built.
2. **How do we stop it happening again?** — the change to how data is
   collected, named, stored and logged, so the next refresh is better.

It also looks for **unused value** in the dataset: columns that are present
but not modelled, geo detail that was aggregated away, delivery metrics sitting
next to spend.

Every tip carries the evidence it was derived from. A tip without evidence is
generic advice, and generic advice is ignored.

The advisor never changes data. It reads the audit dict, the spec and
(optionally) the frame, and writes ``data_quality_tips.md`` / ``.json``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agent_mmm.spec import ChannelRole, DataGranularity, MMMSpec, MMMType
from agent_mmm.workspace import ensure_workspace

# Priority: 1 = blocks or biases the model, 2 = widens uncertainty, 3 = hygiene.
P_CRITICAL, P_HIGH, P_LOW = 1, 2, 3
PRIORITY_LABEL = {P_CRITICAL: "critical", P_HIGH: "high", P_LOW: "nice to have"}

AREAS = (
    "history_and_grain",
    "completeness_and_integrity",
    "spend_and_delivery",
    "variation_and_identifiability",
    "business_drivers",
    "experiments",
    "events_and_context",
    "geo",
)
AREA_TITLE = {
    "history_and_grain": "History and granularity",
    "completeness_and_integrity": "Completeness and integrity",
    "spend_and_delivery": "Spend and delivery metrics",
    "variation_and_identifiability": "Variation and identifiability",
    "business_drivers": "Business drivers (controls)",
    "experiments": "Experiments",
    "events_and_context": "Events and context",
    "geo": "Geography",
}

# Column-name hints for unused columns. Order matters: first match wins.
_NAME_HINTS: list[tuple[str, re.Pattern, str]] = [
    ("exposure", re.compile(r"impr|impressions|grp|trp|clicks|views|reach|freq", re.I),
     "a delivery metric — pair it with its spend column as the channel's exposure input"),
    ("price", re.compile(r"price|discount|promo|markdown|coupon", re.I),
     "a price/promotion driver — the most common omitted confounder"),
    ("distribution", re.compile(r"store|acv|distribution|stock|availability|oos", re.I),
     "a distribution/availability driver"),
    ("kpi", re.compile(r"units|orders|transactions|conversions|signups|leads|volume", re.I),
     "an alternative KPI — check it agrees with the target, or model it as a second outcome"),
    ("external", re.compile(r"temp|weather|rain|holiday|covid|competitor|search_trend|gdp|cpi", re.I),
     "an external factor — candidate control"),
]


@dataclass
class Tip:
    id: str
    area: str
    priority: int
    title: str
    evidence: str
    why: str
    now: str
    future: str

    @property
    def priority_label(self) -> str:
        return PRIORITY_LABEL[self.priority]


@dataclass
class Opportunity:
    title: str
    evidence: str
    action: str


@dataclass
class AdvisorReport:
    tips: list[Tip] = field(default_factory=list)
    opportunities: list[Opportunity] = field(default_factory=list)
    scorecard: dict[str, str] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, *args, **kwargs) -> None:
        self.tips.append(Tip(*args, **kwargs))

    def sorted_tips(self) -> list[Tip]:
        return sorted(self.tips, key=lambda t: (t.priority, AREAS.index(t.area), t.id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "scorecard": self.scorecard,
            "tips": [asdict(t) | {"priority_label": t.priority_label} for t in self.sorted_tips()],
            "opportunities": [asdict(o) for o in self.opportunities],
        }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def advise(spec: MMMSpec, findings: dict, df: Optional[pd.DataFrame] = None) -> AdvisorReport:
    """Build tips and opportunities from an audit result. Pure: writes nothing."""
    checks = findings.get("checks", {})
    rep = AdvisorReport(context={
        "mmm_type": spec.mmm_type.value,
        "granularity": spec.granularity.value,
        "audit_tier": findings.get("summary", {}).get("data_quality_tier"),
        "periods": findings.get("summary", {}).get("periods"),
    })
    geo_cols = _geo_candidates(df) if df is not None and not spec.geo.is_panel else []
    _history(spec, checks, rep)
    _integrity(spec, checks, rep, geo_cols)
    _spend_delivery(spec, checks, rep)
    _variation(spec, checks, rep)
    _drivers(spec, checks, rep)
    _experiments(spec, checks, rep)
    _events(spec, checks, rep)
    _geo(spec, checks, df, rep)
    _unused_columns(spec, checks, rep)
    _brownfield(spec, rep)
    rep.scorecard = _scorecard(rep)
    return rep


def run_advisor(
    spec: MMMSpec,
    findings: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    base: str | Path = ".",
) -> AdvisorReport:
    """Load the audit (or use the one given), advise, and write the outputs."""
    ws = ensure_workspace(base)
    audit_dir = Path(ws) / "audit"
    if findings is None:
        path = audit_dir / "audit.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run the data audit first")
        findings = json.loads(path.read_text(encoding="utf-8"))
    rep = advise(spec, findings, df)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "data_quality_tips.json").write_text(
        json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (audit_dir / "data_quality_tips.md").write_text(render_tips(rep), encoding="utf-8")
    return rep


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _history(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    dims = checks.get("dimensions", {})
    periods = dims.get("periods")
    ppy = spec.periods_per_year()
    if periods:
        years = periods / ppy
        if years < 1:
            rep.add("history-short", "history_and_grain", P_CRITICAL,
                    "Less than one year of history",
                    f"{periods} periods (~{years:.1f} years) from {dims.get('date_min')} to {dims.get('date_max')}.",
                    "Seasonality and media cannot be separated without seeing each season at least once.",
                    "Backfill from platform exports (most keep 2-3 years), or use a geo panel to add observations.",
                    "Archive raw platform exports every month. Platforms delete or restate old data; "
                    "an archive you own is the only reliable history.")
        elif years < 2:
            rep.add("history-one-cycle", "history_and_grain", P_HIGH,
                    "Only one seasonal cycle",
                    f"{periods} periods (~{years:.1f} years).",
                    "One cycle lets the model fit seasonality but not validate it out of sample.",
                    "Keep the holdout short and lean on priors for seasonality; say so in the report.",
                    "Schedule a refresh when the second year completes; keep monthly raw snapshots until then.")
    if spec.granularity == DataGranularity.monthly:
        rep.add("grain-monthly", "history_and_grain", P_HIGH,
                "Monthly data hides carryover and flighting",
                "Spec granularity is monthly.",
                "Adstock half-lives are often 1-4 weeks; at monthly grain they collapse into one period "
                "and the model has 12 points a year to learn from.",
                "Model at monthly if that is all there is, with wide priors and grouped channels.",
                "Collect weekly (ISO weeks, Monday start) at source. Weekly is the standard MMM grain.")
    reg = checks.get("date_regularity", {})
    if reg.get("n_gaps"):
        rep.add("date-gaps", "history_and_grain", P_HIGH,
                "Missing periods in the date range",
                f"{reg['n_gaps']} gaps; largest {reg.get('max_gap_days')} days.",
                "Adstock assumes evenly spaced periods; a hole makes carryover jump.",
                "Reindex to a complete calendar and decide per column what the gap means (zero spend vs unknown).",
                "Add a completeness check to each refresh: expected periods vs received, before anything is joined.")


def _integrity(spec: MMMSpec, checks: dict, rep: AdvisorReport, geo_cols: list[str]) -> None:
    dup = checks.get("duplicate_rows", {})
    if dup.get("n_duplicates"):
        now = "Deduplicate or aggregate before modelling; find out which source produced them."
        if geo_cols:
            now = (f"The data looks geo-level ('{geo_cols[0]}' column) but the spec is national. "
                   "Either declare a geo panel or aggregate to national; do not deduplicate.")
        rep.add("duplicates", "completeness_and_integrity", P_CRITICAL,
                "Duplicate rows on the primary key",
                f"{dup['n_duplicates']} duplicates on {dup.get('keys')}.",
                "Duplicated periods double-count spend and sales.",
                now,
                f"Define the model table's key ({', '.join(dup.get('keys', []))}) and reject loads that violate it.")
    completeness = checks.get("completeness", {})
    channel_cols = set(spec.channel_columns())
    spend_gaps = {c: v["pct_missing"] for c, v in completeness.items()
                  if c in channel_cols and v.get("pct_missing", 0) > 0}
    if spend_gaps:
        rep.add("spend-missing", "completeness_and_integrity", P_HIGH,
                "Spend columns have blanks instead of zeros",
                f"Missing %: {spend_gaps}.",
                "A blank is ambiguous — no spend, or no data? Filling blindly invents or deletes activity.",
                "Confirm with the channel owner, then fill true zeros with 0 and treat the rest as missing.",
                "Make every spend feed emit explicit 0 for dark periods, so a blank always means 'data missing'.")
    tgt = completeness.get(spec.target_column, {})
    if tgt.get("pct_missing", 0) > 0:
        rep.add("target-missing", "completeness_and_integrity", P_CRITICAL,
                "Target has missing periods",
                f"{tgt['pct_missing']}% of target values missing.",
                "Every missing target period is a period the model cannot learn from.",
                "Recover from finance or the order system; never fill the target with 0 or interpolation.",
                "Source the target from the system of record (finance / order DB), not a dashboard export.")
    neg = checks.get("negative_spend", {})
    if neg:
        rep.add("negative-spend", "completeness_and_integrity", P_CRITICAL,
                "Negative spend values",
                f"{neg}.",
                "Credits and refunds booked in the wrong period distort the response curve.",
                "Move each credit back to the period the media actually ran.",
                "Record spend by delivery date, not invoice date, and keep credits as a separate column.")
    if checks.get("non_numeric_columns"):
        rep.add("non-numeric", "completeness_and_integrity", P_CRITICAL,
                "Numeric columns exported as text",
                f"{checks['non_numeric_columns']}.",
                "Currency symbols and thousands separators break parsing; coercion silently creates zeros.",
                "Clean at extract and check row totals before and after.",
                "Export raw numbers (no currency formatting, '.' decimal) and store currency in its own column.")


def _spend_delivery(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    paid = spec.channels_with_role(ChannelRole.paid_media, ChannelRole.paid_reach_frequency)
    spend_only = [c.column for c in paid if not c.exposure_column]
    if paid and spend_only:
        rep.add("no-exposure", "spend_and_delivery", P_HIGH if len(spend_only) == len(paid) else P_LOW,
                "Channels with spend but no delivery metric",
                f"No impressions/GRPs/clicks for: {spend_only}.",
                "Spend mixes price and volume: a CPM rise looks like more media. Delivery metrics let the "
                "model separate the two and make cost sanity checks possible.",
                "Proceed on spend, but treat channels with big CPM swings with caution.",
                "Export impressions (digital), GRPs/TRPs (TV, radio) or clicks alongside spend, from the same "
                "source and on the same date basis.")
    cpu = checks.get("cost_per_exposure_unit", {})
    broken = {ch: e for ch, e in cpu.items()
              if (e.get("ratio_p95_p05") or 0) > 5
              or e.get("periods_spend_without_exposure") or e.get("periods_exposure_without_spend")}
    if broken:
        ev = "; ".join(
            f"{ch}: p95/p05 cost ratio {e.get('ratio_p95_p05')}, "
            f"{e.get('periods_spend_without_exposure', 0)} spend-only and "
            f"{e.get('periods_exposure_without_spend', 0)} delivery-only periods"
            for ch, e in broken.items())
        rep.add("spend-delivery-mismatch", "spend_and_delivery", P_CRITICAL,
                "Spend and delivery do not line up",
                ev + ".",
                "This is nearly always a join problem (different date bases or partial feeds), not a market effect.",
                "Re-pull spend and delivery from the same report and re-join on delivery date.",
                "Take spend and delivery from one platform report per channel, never from two systems joined later.")
    rf = [c.column for c in spec.channels_with_role(ChannelRole.paid_reach_frequency)
          if not (c.reach_column and c.frequency_column)]
    if rf:
        rep.add("rf-missing", "spend_and_delivery", P_HIGH,
                "Reach & frequency channels without reach/frequency data",
                f"{rf}.",
                "Without reach and frequency, Meridian's RF model cannot be used and frequency cannot be optimised.",
                "Model them as ordinary paid media for now.",
                "Ask the platform/agency for weekly reach and average frequency (TV, YouTube, Meta all provide it).")


def _variation(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    pairs = checks.get("high_correlation_pairs") or []
    vif = {k: v for k, v in (checks.get("vif") or {}).items() if v and v > 5}
    if pairs or vif:
        ev = []
        if pairs:
            ev.append("correlated pairs " + ", ".join(f"{p['a']}~{p['b']} (r={p['r']})" for p in pairs[:5]))
        if vif:
            ev.append(f"VIF>5: {vif}")
        rep.add("collinear", "variation_and_identifiability", P_CRITICAL if any(v > 10 for v in vif.values()) else P_HIGH,
                "Channels move together",
                "; ".join(ev) + ".",
                "The data cannot split credit between channels that always move together; the split comes from priors.",
                "Group the correlated channels or report them jointly.",
                "Plan variation on purpose: stagger launches, alternate flighting between channels, or run "
                "geo splits. Independent variation is the raw material of MMM.")
    flat = checks.get("flat_variables") or {}
    always_on = [c for c, v in flat.items() if not v.get("constant")]
    if always_on:
        rep.add("flat", "variation_and_identifiability", P_HIGH,
                "Channels with almost no variation",
                f"Low variation (CV<0.15): {always_on}.",
                "A channel that never changes cannot be measured; its coefficient is the prior.",
                "Report these as assumptions, not estimates.",
                "Run pulse tests: 2-4 week spend changes of ±30-50%, or a geo holdout, once a year.")
    budget = checks.get("parameter_budget") or {}
    if budget.get("obs_per_parameter") is not None and budget["obs_per_parameter"] < 10:
        rep.add("param-budget", "variation_and_identifiability", P_HIGH,
                "Too many parameters for the data",
                f"{budget['obs_per_parameter']} observations per parameter "
                f"({budget['n_observations']} obs, ~{budget['n_parameters_approx']} params).",
                "Below ~10 per parameter, priors drive most of the answer.",
                "Group small channels, reduce Fourier terms, drop controls that do not move the target.",
                "Collect geo-level data or a finer grain to add observations without adding parameters.")


def _drivers(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    cov = checks.get("confounder_coverage") or {}
    if cov and not cov.get("price"):
        rep.add("no-price", "business_drivers", P_CRITICAL,
                "No price or promotion data",
                "No pricing control or price channel in the spec.",
                "Promotions are planned alongside media; without them, promotion lift is credited to media.",
                "Build at least a promo-week flag from the trading calendar.",
                "Keep a weekly average-price index and a promotion calendar (dates, depth, mechanic) "
                "as standard inputs.")
    if cov and not cov.get("distribution") and spec.industry.lower() in {"retail", "cpg", "fmcg", "grocery"}:
        rep.add("no-distribution", "business_drivers", P_HIGH,
                "No distribution data in a retail/CPG model",
                f"Industry: {spec.industry}.",
                "Store count, ACV or stock-outs move sales independently of media.",
                "Use a store-count or stock-availability series if any exists.",
                "Add weekly distribution (ACV/store count) and out-of-stock rate to the data feed.")


def _experiments(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    paid = {c.column for c in spec.channels_with_role(ChannelRole.paid_media, ChannelRole.paid_reach_frequency)}
    if not spec.experiments:
        rep.add("no-experiments", "experiments", P_HIGH,
                "No incrementality experiments",
                "spec.experiments is empty.",
                "Without one, several very different attributions fit the data equally well.",
                "Present results as ranges; use /mmm-experiment-plan to pick the first test.",
                "Keep an experiment register: channel, design, dates, geos, lift, standard error, spend during test.")
        return
    exp = checks.get("experiments") or {}
    n, usable = exp.get("n", len(spec.experiments)), exp.get("n_usable_as_constraint", 0)
    if usable < n:
        rep.add("experiments-incomplete", "experiments", P_HIGH,
                "Experiments missing what calibration needs",
                f"{n - usable} of {n} experiments lack lift, standard error or spend during the test.",
                "A lift without its uncertainty and spend cannot calibrate the model.",
                "Recover the standard error and test spend from the original readout.",
                "Make the readout template require: absolute lift, standard error, spend, dates, geos.")
    uncovered = sorted(paid - {e.channel for e in spec.experiments})
    if uncovered:
        rep.add("experiments-coverage", "experiments", P_LOW,
                "Channels never tested",
                f"{uncovered}.",
                "Their effects rest on observational data alone.",
                "Show them with wider ranges than tested channels.",
                "Rotate tests across the largest untested channels, biggest spend first.")


def _events(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    br = checks.get("structural_break") or {}
    td = checks.get("target_distribution") or {}
    outliers = td.get("outlier_dates") or []
    if br.get("break_detected") or outliers:
        ev = []
        if br.get("break_detected"):
            ev.append(f"structural break in the target (p={br.get('p_value')})")
        if outliers:
            ev.append(f"{len(outliers)} extreme periods, e.g. {outliers[:3]}")
        rep.add("events", "events_and_context", P_HIGH,
                "Unexplained shifts and spikes in the target",
                "; ".join(ev) + ".",
                "Anything not in the model is pushed onto whatever channel moved at the same time.",
                "Identify each spike and the break; add indicator variables for the real ones.",
                "Keep a dated event log: launches, price changes, outages, tracking changes, PR, competitor "
                "moves. It is the cheapest high-value data an MMM can have.")
    ss = (checks.get("seasonality") or {}).get("seasonal_strength")
    if ss is not None and ss > 0.5 and (rep.context.get("periods") or 0) < 3 * spec.periods_per_year():
        rep.add("strong-seasonality", "events_and_context", P_LOW,
                "Strong seasonality with limited history",
                f"Seasonal strength {ss}.",
                "When seasons and media peaks coincide, the model struggles to tell them apart.",
                "Use informative seasonality priors; check the decomposition by eye.",
                "Shift some activity off-peak for a season to create contrast.")


def _geo(spec: MMMSpec, checks: dict, df: Optional[pd.DataFrame], rep: AdvisorReport) -> None:
    panel = checks.get("panel") or {}
    if spec.geo.is_panel:
        if not spec.geo.population_column:
            rep.add("geo-population", "geo", P_HIGH,
                    "Geo panel without population",
                    "spec.geo.population_column is not set.",
                    "Meridian needs population for per-capita scaling; without it big geos dominate.",
                    "Add census population per geo (yearly is fine).",
                    "Keep a geo lookup table: geo code, name, population, region — versioned.")
        if panel and not panel.get("rectangular", True):
            rep.add("geo-ragged", "geo", P_CRITICAL,
                    "Geos have different date coverage",
                    f"{panel.get('n_geos')} geos with unequal periods.",
                    "Panel models need every (geo, period) cell.",
                    "Fill explicit zeros for dark media, or drop partial geos.",
                    "Export every geo for every period, including zero rows.")
        return
    if df is None:
        return
    for c in _geo_candidates(df)[:1]:
        rep.opportunities.append(Opportunity(
            "Model at geo level",
            f"Column '{c}' has {int(df[c].nunique())} distinct values, but the spec is national.",
            "A geo panel multiplies observations and gives the model regional contrast to learn from. "
            "Worth it from ~5 geos with media varying by geo.",
        ))


def _geo_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if re.search(r"geo|region|dma|state|country|market|city|province", str(c), re.I)
            and not pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() >= 2]


def _unused_columns(spec: MMMSpec, checks: dict, rep: AdvisorReport) -> None:
    unused = checks.get("unused_numeric_columns") or []
    grouped: dict[str, list[str]] = {}
    for col in unused:
        for kind, rx, _ in _NAME_HINTS:
            if rx.search(col):
                grouped.setdefault(kind, []).append(col)
                break
    for kind, rx, meaning in _NAME_HINTS:
        cols = grouped.get(kind)
        if cols:
            rep.opportunities.append(Opportunity(
                f"Unused {kind} columns",
                f"{cols} are in the data but not the spec.",
                f"Looks like {meaning}.",
            ))
    rest = [c for c in unused if not any(c in v for v in grouped.values())]
    if rest:
        rep.opportunities.append(Opportunity(
            "Other unused numeric columns",
            f"{rest[:10]}{'...' if len(rest) > 10 else ''}.",
            "Confirm each is not a driver. An omitted driver is credited to whatever correlates with it.",
        ))


def _brownfield(spec: MMMSpec, rep: AdvisorReport) -> None:
    if spec.mmm_type != MMMType.brownfield:
        return
    bf = spec.brownfield
    if bf is not None and not bf.known_results:
        rep.opportunities.append(Opportunity(
            "Record the previous model's results",
            "Brownfield project with no known_results in the spec.",
            "Write down the ROAS/contribution figures stakeholders already believe. New results will be "
            "judged against them, and large differences need an explanation ready.",
        ))


def _scorecard(rep: AdvisorReport) -> dict[str, str]:
    card = {}
    for area in AREAS:
        prios = [t.priority for t in rep.tips if t.area == area]
        card[area] = (
            "critical" if P_CRITICAL in prios
            else "needs work" if P_HIGH in prios
            else "minor" if prios
            else "good"
        )
    return card


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_ICON = {"critical": "🔴", "needs work": "🟠", "minor": "🟡", "good": "🟢"}


def render_tips(rep: AdvisorReport) -> str:
    ctx = rep.context
    lines = [
        "# Data quality: improvement plan",
        "",
        f"Audit verdict: **{ctx.get('audit_tier')}** · {ctx.get('periods')} periods · "
        f"{ctx.get('granularity')} · {ctx.get('mmm_type')}",
        "",
        "## Scorecard",
        "",
        "| Area | Status |",
        "|---|---|",
    ]
    lines += [f"| {AREA_TITLE[a]} | {_ICON[s]} {s} |" for a, s in rep.scorecard.items()]
    lines.append("")

    tips = rep.sorted_tips()
    if tips:
        lines += ["## Tips, most important first", ""]
        for i, t in enumerate(tips, 1):
            lines += [
                f"### {i}. {t.title} — *{t.priority_label}*",
                "",
                f"- **Evidence:** {t.evidence}",
                f"- **Why it matters:** {t.why}",
                f"- **Now (this model):** {t.now}",
                f"- **Going forward (data collection):** {t.future}",
                "",
            ]
    else:
        lines += ["No data-quality issues found by the audit.", ""]

    if rep.opportunities:
        lines += ["## Better use of this dataset", ""]
        for o in rep.opportunities:
            lines += [f"- **{o.title}.** {o.evidence} {o.action}"]
        lines.append("")

    futures = [t.future for t in tips if t.priority <= P_HIGH]
    if futures:
        lines += ["## Data collection checklist for the next refresh", ""]
        lines += [f"- [ ] {f}" for f in dict.fromkeys(futures)]
        lines.append("")
    return "\n".join(lines)
