"""Mapping raw campaign names to modelling channels.

A media export names things the way the person buying the media named them:
``BR_Prospecting_Q1_v2``, ``PERF | DPA | Catalog | UK``, ``2023_TV_Q4_Burst``.
The model needs a handful of channels. The mapping between the two is where a
surprising share of MMM error lives, because it is done once, by hand, in a
spreadsheet nobody keeps, and it is never checked again.

This module makes the mapping explicit, reviewable and *auditable*:

* deterministic rules, applied in priority order, each one recorded on the row
  it matched, so any channel assignment can be traced back to the rule that
  produced it;
* a coverage report that refuses to be quietly incomplete — unmapped names are
  listed with their spend, ranked by how much money is at stake;
* checks for the failure modes that matter: a rule that fires on nothing, two
  rules that fight over the same names, a channel that ends up holding almost
  all the spend, and brand/generic search collapsed into one bucket.

Nothing here guesses a mapping and proceeds. Unmapped spend above a threshold is
an error, not a rounding difference.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Pattern

import pandas as pd

from .utils.channel_classifier import classify_channel, normalize_column

__all__ = [
    "MappingRule",
    "MappingResult",
    "TaxonomyReport",
    "suggest_rules",
    "apply_rules",
    "apply_rules_with_provenance",
    "build_taxonomy",
    "render_taxonomy_report",
    "pivot_to_channels",
]


@dataclass
class MappingRule:
    """One rule: a pattern over the raw name, and the channel it produces.

    ``priority`` orders evaluation — lower runs first, and the first match wins.
    Ordering matters more than it looks: a ``brand`` rule must beat a generic
    ``search`` rule, or brand search disappears into a bucket where its
    (usually inflated, usually non-incremental) effect is impossible to see.
    """

    pattern: str
    channel: str
    priority: int = 100
    field: str = ""
    """Column the pattern applies to. Empty means the primary name column."""
    case_sensitive: bool = False
    note: str = ""

    def compiled(self) -> Pattern[str]:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(self.pattern, flags)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class MappingResult:
    """Per-rule outcome, so a mapping can be reviewed rather than trusted."""

    rule: MappingRule
    n_rows: int = 0
    n_names: int = 0
    spend: float = 0.0
    example_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.rule.pattern,
            "channel": self.rule.channel,
            "priority": self.rule.priority,
            "n_rows": self.n_rows,
            "n_names": self.n_names,
            "spend": round(self.spend, 2),
            "example_names": self.example_names,
        }


@dataclass
class TaxonomyReport:
    name_column: str
    spend_column: Optional[str]
    total_rows: int = 0
    total_spend: float = 0.0
    mapped_rows: int = 0
    mapped_spend: float = 0.0
    results: list[MappingResult] = field(default_factory=list)
    unmapped: list[dict[str, Any]] = field(default_factory=list)
    channel_spend: dict[str, float] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    @property
    def coverage_rows(self) -> float:
        return self.mapped_rows / self.total_rows if self.total_rows else 0.0

    @property
    def coverage_spend(self) -> float:
        return self.mapped_spend / self.total_spend if self.total_spend else 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_column": self.name_column,
            "spend_column": self.spend_column,
            "total_rows": self.total_rows,
            "total_spend": round(self.total_spend, 2),
            "mapped_rows": self.mapped_rows,
            "mapped_spend": round(self.mapped_spend, 2),
            "coverage_rows": round(self.coverage_rows, 4),
            "coverage_spend": round(self.coverage_spend, 4),
            "results": [r.to_dict() for r in self.results],
            "unmapped": self.unmapped,
            "channel_spend": {k: round(v, 2) for k, v in self.channel_spend.items()},
            "conflicts": self.conflicts,
            "errors": self.errors,
            "warnings": self.warnings,
            "questions": self.questions,
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
# Rule suggestion
# --------------------------------------------------------------------------- #
# Ordered most specific first. Brand/generic search and prospecting/retargeting
# are split deliberately: collapsing either pair hides the distinction the model
# most needs to make, because the two halves have opposite incrementality.
_SUGGESTION_RULES: tuple[tuple[str, str, int, str], ...] = (
    (r"\b(brand(ed)?|trademark|tm)\b.*\b(search|sem|ppc|paid_?search)\b|"
     r"\b(search|sem|ppc|paid_?search)\b.*\b(brand(ed)?|trademark|tm)\b",
     "search_brand", 10,
     "brand search is usually the least incremental line in the plan; keeping it separate is the "
     "point"),
    (r"\b(generic|non_?brand|nonbrand|unbranded|category)\b", "search_generic", 15, ""),
    (r"\b(pmax|performance_?max)\b", "pmax", 18,
     "PMax spans search, shopping, display and video; it cannot be attributed to a single channel"),
    (r"\b(shopping|pla|product_?listing)\b", "shopping", 20, ""),
    (r"\b(retarget\w*|remarket\w*|rtg|rmkt|dpa|dynamic_?product)\b", "social_retargeting", 25,
     "retargeting reaches people who already intended to buy; its measured ROAS is the most "
     "commonly overstated number in any plan"),
    (r"\b(prospect\w*|acquisition|cold|new_?customer|aquisition)\b", "social_prospecting", 30, ""),
    (r"\b(tv|linear_?tv|broadcast|bvod|ctv|connected_?tv|ott)\b", "tv", 35, ""),
    (r"\b(youtube|yt|trueview)\b", "video_youtube", 38, ""),
    (r"\b(video|preroll|pre_?roll|instream)\b", "video_other", 40, ""),
    (r"\b(meta|facebook|fb|instagram|ig)\b", "social_meta", 45, ""),
    (r"\b(tiktok|tt)\b", "social_tiktok", 46, ""),
    (r"\b(snap|snapchat|pinterest|reddit|twitter|x_ads)\b", "social_other", 47, ""),
    (r"\b(linkedin|li_ads)\b", "social_linkedin", 48, ""),
    (r"\b(display|gdn|banner|programmatic|dv360|prog)\b", "display", 50, ""),
    (r"\b(affiliate|partner|cj|awin|rakuten)\b", "affiliate", 55,
     "affiliate is frequently last-click credit for demand created elsewhere; check before "
     "treating it as incremental media"),
    (r"\b(email|crm|newsletter|edm)\b", "email", 60,
     "email is owned media: it gets carryover but never a ROAS, because there is no media cost"),
    (r"\b(ooh|oo_?h|outdoor|billboard|dooh|transit)\b", "ooh", 65, ""),
    (r"\b(radio|audio|spotify|podcast|dax)\b", "audio", 70, ""),
    (r"\b(print|press|magazine|newspaper)\b", "print", 75, ""),
    (r"\b(cinema)\b", "cinema", 78, ""),
    (r"\b(direct_?mail|dm|door_?drop|catalog(ue)?)\b", "direct_mail", 80, ""),
    (r"\b(search|sem|ppc|paid_?search|adwords|google_?ads)\b", "search_other", 90,
     "a search campaign that matched no brand/generic rule — confirm which it is before modelling"),
    (r"\b(social|paid_?social)\b", "social_other", 92, ""),
)


def _tokenise(name: str) -> str:
    """Normalise a raw campaign name so word-boundary patterns can match it."""
    text = str(name).lower()
    # Campaign names are built out of every separator anyone has ever typed.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


def suggest_rules(
    names: Iterable[str], *, include_unmatched_note: bool = True
) -> list[MappingRule]:
    """Propose a starting rule set from the names actually present.

    Only rules that match something are returned, so the output is a
    ready-to-review mapping for *this* account rather than a generic catalogue.
    The result is a draft: it is meant to be read, edited and argued with.
    """
    tokenised = [_tokenise(n) for n in names]
    rules: list[MappingRule] = []
    for pattern, channel, priority, note in _SUGGESTION_RULES:
        rx = re.compile(pattern, re.IGNORECASE)
        if any(rx.search(t) for t in tokenised):
            rules.append(MappingRule(pattern=pattern, channel=channel, priority=priority, note=note))

    if include_unmatched_note and rules:
        covered = set()
        for r in rules:
            rx = r.compiled()
            covered |= {t for t in tokenised if rx.search(t)}
        if len(covered) < len(set(tokenised)):
            # Deliberately not auto-filled: an unmatched name is a question.
            pass
    return sorted(rules, key=lambda r: r.priority)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def apply_rules(
    df: pd.DataFrame,
    rules: list[MappingRule],
    *,
    name_column: str,
    fallback: Optional[str] = None,
) -> pd.Series:
    """Return the channel for each row, plus which rule produced it.

    First match wins, in priority order. Rows that match nothing get
    ``fallback`` (or ``NA``) — never a silently invented channel.
    """
    channel, _ = apply_rules_with_provenance(
        df, rules, name_column=name_column, fallback=fallback
    )
    return channel


def apply_rules_with_provenance(
    df: pd.DataFrame,
    rules: list[MappingRule],
    *,
    name_column: str,
    fallback: Optional[str] = None,
) -> tuple[pd.Series, pd.Series]:
    """As ``apply_rules``, but also return the pattern that produced each row.

    Provenance is what makes a mapping auditable: given a channel number anyone
    disagrees with, this says exactly which rule put which campaigns into it.
    """
    tokenised = df[name_column].map(_tokenise)
    channel = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    via = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    for rule in sorted(rules, key=lambda r: r.priority):
        col = rule.field or name_column
        source = df[col].map(_tokenise) if col != name_column else tokenised
        rx = rule.compiled()
        # Compiled search rather than Series.str.contains: the suggestion
        # patterns carry capture groups, which str.contains warns about.
        hit = source.map(lambda t: bool(rx.search(t)))
        unclaimed = hit & channel.isna()
        channel = channel.mask(unclaimed, rule.channel)
        via = via.mask(unclaimed, rule.pattern)

    if fallback is not None:
        channel = channel.fillna(fallback)
    return channel, via


def _detect_conflicts(names: Iterable[str], rules: list[MappingRule]) -> list[dict[str, Any]]:
    """Find names that several rules claim, where only priority decides.

    A conflict is not automatically wrong — priority exists precisely to resolve
    them — but an unexamined conflict means a channel's spend depends on a
    number nobody chose deliberately.
    """
    ordered = sorted(rules, key=lambda r: r.priority)
    conflicts: list[dict[str, Any]] = []
    for name in set(names):
        token = _tokenise(name)
        matched = [r for r in ordered if r.compiled().search(token)]
        channels = list(dict.fromkeys(r.channel for r in matched))
        if len(channels) > 1:
            conflicts.append(
                {
                    "name": str(name),
                    "channels": channels,
                    "resolved_to": channels[0],
                    "losing_patterns": [r.pattern for r in matched[1:]],
                }
            )
    conflicts.sort(key=lambda c: c["name"])
    return conflicts


def build_taxonomy(
    df: pd.DataFrame,
    *,
    name_column: str,
    rules: Optional[list[MappingRule]] = None,
    spend_column: Optional[str] = None,
    min_spend_coverage: float = 0.99,
    concentration_warn: float = 0.6,
) -> TaxonomyReport:
    """Apply a mapping and report on it honestly.

    ``min_spend_coverage`` defaults to 99%: below that, the report carries an
    error rather than a warning, because a channel built from 90% of its spend
    produces a ROAS that is wrong by construction and looks entirely normal.
    """
    if name_column not in df.columns:
        raise KeyError(f"name_column '{name_column}' not in dataframe")
    if spend_column is not None and spend_column not in df.columns:
        raise KeyError(f"spend_column '{spend_column}' not in dataframe")

    names = df[name_column].dropna().astype(str)
    rules = rules if rules is not None else suggest_rules(names.unique())

    report = TaxonomyReport(name_column=name_column, spend_column=spend_column)
    report.total_rows = len(df)

    spend = (
        pd.to_numeric(df[spend_column], errors="coerce").fillna(0.0)
        if spend_column
        else pd.Series(0.0, index=df.index)
    )
    report.total_spend = float(spend.sum())

    if not rules:
        report.errors.append(
            f"No mapping rules matched any of the {names.nunique()} distinct names in "
            f"'{name_column}'. The naming convention is not one we recognise — the mapping has to "
            "be written by hand, with the person who named the campaigns."
        )
        report.unmapped = _unmapped_table(df, name_column, spend, pd.Series(pd.NA, index=df.index))
        return report

    assigned, via = apply_rules_with_provenance(df, rules, name_column=name_column)
    mapped_mask = assigned.notna()
    report.mapped_rows = int(mapped_mask.sum())
    report.mapped_spend = float(spend[mapped_mask].sum())

    for rule in sorted(rules, key=lambda r: r.priority):
        # Rows this rule actually won, not merely matched.
        own = via == rule.pattern
        res = MappingResult(rule=rule)
        res.n_rows = int(own.sum())
        matched_names = sorted(df.loc[own, name_column].astype(str).unique())
        res.n_names = len(matched_names)
        res.spend = float(spend[own].sum())
        res.example_names = matched_names[:5]
        report.results.append(res)

        if res.n_rows == 0:
            report.warnings.append(
                f"Rule `{rule.pattern}` → {rule.channel} matched nothing. It is either dead weight "
                "or it is being shadowed by a higher-priority rule."
            )

    report.channel_spend = {
        str(ch): float(spend[assigned == ch].sum()) for ch in sorted(assigned.dropna().unique())
    }
    report.unmapped = _unmapped_table(df, name_column, spend, assigned)
    report.conflicts = _detect_conflicts(names.unique(), rules)

    _assess(report, assigned, spend, min_spend_coverage, concentration_warn)
    return report


def _unmapped_table(
    df: pd.DataFrame, name_column: str, spend: pd.Series, assigned: pd.Series
) -> list[dict[str, Any]]:
    """Unmapped names ranked by spend — the order they should be worked through."""
    mask = assigned.isna()
    if not mask.any():
        return []
    frame = pd.DataFrame({"name": df.loc[mask, name_column].astype(str), "spend": spend[mask]})
    grouped = (
        frame.groupby("name", as_index=False)
        .agg(rows=("spend", "size"), spend=("spend", "sum"))
        .sort_values(["spend", "rows"], ascending=False)
    )
    return [
        {"name": r["name"], "rows": int(r["rows"]), "spend": round(float(r["spend"]), 2)}
        for _, r in grouped.iterrows()
    ]


def _assess(
    report: TaxonomyReport,
    assigned: pd.Series,
    spend: pd.Series,
    min_spend_coverage: float,
    concentration_warn: float,
) -> None:
    """Turn the mapping into findings and questions."""
    if report.spend_column and report.total_spend > 0:
        if report.coverage_spend < min_spend_coverage:
            missing = report.total_spend - report.mapped_spend
            report.errors.append(
                f"{1 - report.coverage_spend:.1%} of spend ({missing:,.0f}) is unmapped. Every "
                "unmapped currency unit is media the model cannot see but whose effect is still in "
                "the target, so it lands in the baseline or in whichever channel correlates with "
                "it."
            )
    elif report.coverage_rows < min_spend_coverage:
        report.errors.append(
            f"{1 - report.coverage_rows:.1%} of rows are unmapped and there is no spend column to "
            "weigh them by. Supply spend before accepting this mapping."
        )

    if report.unmapped:
        top = report.unmapped[:5]
        report.questions.append(
            "What channel do these unmapped names belong to? "
            + "; ".join(f"'{u['name']}' ({u['spend']:,.0f})" for u in top)
        )

    if report.channel_spend and report.total_spend > 0:
        biggest, biggest_spend = max(report.channel_spend.items(), key=lambda kv: kv[1])
        share = biggest_spend / report.total_spend
        if share > concentration_warn and len(report.channel_spend) > 1:
            report.warnings.append(
                f"'{biggest}' holds {share:.0%} of mapped spend. Either the mapping is too coarse, "
                "or this channel genuinely dominates and the others will be poorly identified."
            )
            report.questions.append(
                f"Can '{biggest}' be split into meaningfully different sub-channels (by objective, "
                "audience, or format)? A single bucket holding most of the budget cannot be "
                "optimised against."
            )

    channels = set(report.channel_spend)
    if "search_other" in channels:
        report.questions.append(
            "Some search campaigns matched neither brand nor generic. Which are they? Brand and "
            "generic search behave so differently that modelling them together produces a number "
            "that describes neither."
        )
    if "search_brand" in channels and "search_generic" not in channels:
        report.questions.append(
            "Brand search is present but generic search is not. Is generic search bought at all, "
            "and if so where is it in this export?"
        )
    if "pmax" in channels:
        report.questions.append(
            "PMax spend spans search, shopping, display and video inventory. Do we have the "
            "channel breakdown, or should PMax stay a single channel with the caveat that its "
            "coefficient mixes several media types?"
        )
    if "email" in channels:
        report.questions.append(
            "Email appears as a channel. Is there a media cost against it? If not it is organic: "
            "it gets carryover and saturation but must never be given a ROAS or entered into the "
            "budget optimiser."
        )

    if report.conflicts:
        report.warnings.append(
            f"{len(report.conflicts)} campaign name(s) match more than one rule; priority decided "
            "the outcome. Check the resolutions before accepting the mapping."
        )


def pivot_to_channels(
    df: pd.DataFrame,
    *,
    date_column: str,
    channel_column: str,
    value_columns: dict[str, str],
    geo_column: Optional[str] = None,
    freq: str = "W-MON",
) -> pd.DataFrame:
    """Aggregate a long export to one row per period (per geo) and one column per channel.

    ``value_columns`` maps a source column to an aggregation ("sum" or "mean").
    Sum is right for spend and impressions; mean is right for rates and prices,
    and using the wrong one is a silent, systematic error — summing a price
    across campaigns produces a number with no meaning.
    """
    from .discovery import parse_date_column

    work = df.copy()
    parsed, _, _ = parse_date_column(work[date_column])
    work["_period"] = parsed.dt.to_period(_freq_to_period(freq)).dt.start_time
    work = work[work["_period"].notna() & work[channel_column].notna()]

    keys = ["_period"] + ([geo_column] if geo_column else []) + [channel_column]
    agg = work.groupby(keys, as_index=False).agg(
        {col: how for col, how in value_columns.items()}
    )

    index = ["_period"] + ([geo_column] if geo_column else [])
    wide = agg.pivot_table(index=index, columns=channel_column, values=list(value_columns), aggfunc="first")
    wide.columns = [f"{channel}_{metric}" for metric, channel in wide.columns]
    wide = wide.sort_index().reset_index().rename(columns={"_period": date_column})

    # A channel absent from a period means no activity, not unknown activity —
    # this is the one place a zero-fill is a true statement about the world.
    spend_like = [c for c in wide.columns if c not in {date_column, geo_column}]
    wide[spend_like] = wide[spend_like].fillna(0.0)
    return wide


def _freq_to_period(freq: str) -> str:
    f = freq.upper()
    if f.startswith("W"):
        return "W"
    if f.startswith("M"):
        return "M"
    if f.startswith("Q"):
        return "Q"
    return "D"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_taxonomy_report(report: TaxonomyReport) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Channel Taxonomy Mapping")
    a("")
    verdict = "PASS" if report.ok else "FAIL"
    a(f"**{verdict}** — {report.coverage_rows:.1%} of rows mapped"
      + (f", {report.coverage_spend:.1%} of spend" if report.spend_column else ""))
    a("")
    a(f"* Name column: `{report.name_column}`")
    if report.spend_column:
        a(f"* Spend column: `{report.spend_column}` (total {report.total_spend:,.0f})")
    a(f"* Rows: {report.total_rows:,}")
    a("")

    if report.errors:
        a("## Errors — the mapping is not usable yet")
        a("")
        for e in report.errors:
            a(f"* {e}")
        a("")

    a("## Rules applied")
    a("")
    a("| Priority | Channel | Pattern | Names | Rows | Spend | Examples |")
    a("|---:|---|---|---:|---:|---:|---|")
    for r in report.results:
        examples = ", ".join(f"`{n}`" for n in r.example_names[:3])
        a(
            f"| {r.rule.priority} | {r.rule.channel} | `{r.rule.pattern}` | {r.n_names} | "
            f"{r.n_rows} | {r.spend:,.0f} | {examples} |"
        )
    a("")

    notes = [r.rule for r in report.results if r.rule.note]
    if notes:
        a("### Why some of these are separate channels")
        a("")
        for rule in notes:
            a(f"* **{rule.channel}** — {rule.note}")
        a("")

    if report.channel_spend:
        a("## Resulting channels")
        a("")
        a("| Channel | Spend | Share |")
        a("|---|---:|---:|")
        total = report.total_spend or 1.0
        for ch, sp in sorted(report.channel_spend.items(), key=lambda kv: -kv[1]):
            a(f"| {ch} | {sp:,.0f} | {sp / total:.1%} |")
        a("")

    if report.unmapped:
        a("## Unmapped names")
        a("")
        a("Ranked by spend — work down this list, not alphabetically.")
        a("")
        a("| Name | Rows | Spend |")
        a("|---|---:|---:|")
        for u in report.unmapped[:30]:
            a(f"| `{u['name']}` | {u['rows']} | {u['spend']:,.0f} |")
        if len(report.unmapped) > 30:
            a(f"| … {len(report.unmapped) - 30} more | | |")
        a("")

    if report.conflicts:
        a("## Rules in conflict")
        a("")
        a("| Name | Resolved to | Also matched |")
        a("|---|---|---|")
        for c in report.conflicts[:20]:
            a(f"| `{c['name']}` | {c['resolved_to']} | {', '.join(c['channels'][1:])} |")
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
