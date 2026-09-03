"""Raw-file discovery: turning a folder of exports into a defensible MMM dataset plan.

Real MMM projects do not start from a tidy CSV. They start from a shared drive
containing a Meta export, three Google Ads reports at different grains, a GA4
extract, a finance workbook, a promo calendar somebody maintained by hand, and
two files nobody can explain. This module reads that folder and answers the
questions that have to be settled before any modelling decision is meaningful:

* What is in each file, and what is its grain?
* Which column is the date, and what period does the data actually run on?
* Is the file already one row per period (wide), or one row per campaign/day (long)?
* Which columns are spend, exposure, target, price, control — and which are unknown?
* Do the files share a key we can join on, and over what date range do they overlap?
* What must a human answer before this can be trusted?

Nothing here silently decides anything. Every inference carries a confidence and
a reason, and everything the module could not settle becomes an explicit
question in ``DiscoveryReport.open_questions``. The agent asks those questions;
it does not guess past them.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .utils.channel_classifier import classify_channel, normalize_column

__all__ = [
    "parse_date_column",
    "ColumnProfile",
    "FileProfile",
    "JoinCandidate",
    "DiscoveryReport",
    "profile_dataframe",
    "profile_file",
    "discover",
    "render_discovery_report",
    "SEMANTIC_ROLES",
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
READABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls", ".json"}

#: Semantic role of a raw column, before it becomes a spec role. These are
#: deliberately coarser than ``spec.ChannelRole``: discovery says "this looks
#: like money spent on media", the intake conversation decides whether that is
#: ``paid_media`` or something else.
SEMANTIC_ROLES = (
    "date",
    "geo",
    "entity",          # campaign / adset / creative / product identifier
    "spend",
    "exposure",        # impressions, clicks, GRPs, reach, sends
    "target",          # revenue, orders, conversions, signups
    "price",
    "distribution",
    "promotion",
    "control",
    "currency",
    "identifier",      # ids, keys — useful for joins, never modelled
    "unknown",
)

# Ordered: the first pattern that matches a normalised column name wins, so more
# specific patterns must come first. `normalize_column` lowercases and collapses
# separators to underscores, so patterns are written against that form.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Date first — "order_date" must not be captured by the target patterns.
    ("date", r"(^|_)(date|week|month|day|period|dt|ds|time|timestamp|reporting_date)($|_)"),
    # Before geo: "store_count" is a distribution measure, while a bare "store"
    # column is usually a geo dimension.
    ("distribution", r"(^|_)(distribution|acv|tdp|store_count|stores|num_stores|store_cnt|"
                     r"availability|shelf|doors|weighted_distribution)($|_)"),
    ("geo", r"(^|_)(geo|region|market|country|state|province|dma|city|store|location|territory)($|_)"),
    ("currency", r"(^|_)(currency|currency_code|curr|ccy)($|_)"),
    # Spend before exposure: "cost_per_click" is a rate, not spend — excluded below.
    ("spend", r"(^|_)(spend|cost|investment|budget|media_cost|net_cost|gross_cost|amount_spent)($|_)"),
    ("exposure", r"(^|_)(impression|impressions|imps|clicks|reach|grp|trp|views|sends|opens|"
                 r"sessions|visits|deliveries|circulation|plays)($|_)"),
    ("target", r"(^|_)(revenue|sales|orders|conversions|transactions|purchases|bookings|signups|"
                r"subscriptions|installs|leads|registrations|units_sold|gmv|net_revenue)($|_)"),
    ("price", r"(^|_)(price|asp|avg_price|average_price|unit_price|discount|markdown|price_index)($|_)"),
    ("promotion", r"(^|_)(promo|promotion|deal|offer|coupon|campaign_flag|feature|display)($|_)"),
    ("entity", r"(^|_)(campaign|campaign_name|adset|ad_set|adgroup|ad_group|creative|ad_name|"
               r"placement|publisher|product|sku|line_item)($|_)"),
    ("identifier", r"(^|_)(id|key|uuid|guid|account_id|customer_id|hash)($|_)"),
    ("control", r"(^|_)(holiday|weather|temperature|temp|rainfall|competitor|cpi|inflation|gdp|"
                r"unemployment|covid|lockdown|index|macro)($|_)"),
)

# Rate-like names that superficially match "cost" or "revenue" but are ratios.
# Modelling a ratio as spend is a silent, serious error, so they are demoted to
# `control` and always raise a question.
_RATE_PATTERN = re.compile(
    r"(^|_)(cpc|cpm|cpa|cpi|cpl|roas|roi|ctr|cvr|cpv|cpp|rate|per_click|per_impression|"
    r"per_order|per_acquisition|_pct|percent|share)($|_)"
)

_CURRENCY_SYMBOLS = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR",
    "₽": "RUB", "R$": "BRL", "kr": "SEK", "zł": "PLN", "₩": "KRW",
}

_TRUE_STRINGS = {"true", "yes", "y", "t", "1"}
_FALSE_STRINGS = {"false", "no", "n", "f", "0"}

# Column names that are dates in every marketing export, checked before the
# generic regex so a file with both `date` and `week_start` picks sensibly.
_PREFERRED_DATE_NAMES = (
    "date", "week", "week_start", "week_starting", "week_commencing", "period",
    "month", "day", "reporting_date", "ds", "dt", "event_date", "order_date",
)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
@dataclass
class ColumnProfile:
    """What we can say about one raw column without asking anyone."""

    name: str
    dtype: str
    n_rows: int
    n_missing: int
    n_unique: int
    role: str = "unknown"
    role_confidence: float = 0.0
    role_reason: str = ""
    is_numeric: bool = False
    is_datelike: bool = False
    is_constant: bool = False
    is_boolean_like: bool = False
    numeric_after_cleaning: bool = False
    """True when the column is stored as text but becomes numeric once currency
    symbols and thousands separators are stripped. A very common export defect."""
    detected_currency: Optional[str] = None
    date_format: Optional[str] = None
    ambiguous_date_formats: list[str] = field(default_factory=list)
    zero_fraction: float = 0.0
    negative_fraction: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    sample_values: list[str] = field(default_factory=list)
    channel_guess: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def missing_fraction(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["missing_fraction"] = round(self.missing_fraction, 4)
        return d


@dataclass
class FileProfile:
    """What we can say about one raw file."""

    path: str
    name: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile] = field(default_factory=list)
    date_column: Optional[str] = None
    date_confidence: float = 0.0
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_format: Optional[str] = None
    ambiguous_date_formats: list[str] = field(default_factory=list)
    inferred_grain: Optional[str] = None
    grain_confidence: float = 0.0
    shape_kind: str = "unknown"          # "wide_timeseries" | "long_transactional" | "lookup" | "unknown"
    grain_keys: list[str] = field(default_factory=list)
    duplicate_key_rows: int = 0
    detected_currencies: list[str] = field(default_factory=list)
    likely_source: Optional[str] = None  # meta_ads, google_ads, ga4, finance, ...
    read_error: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def columns_with_role(self, *roles: str) -> list[ColumnProfile]:
        wanted = set(roles)
        return [c for c in self.columns if c.role in wanted]

    def column(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "columns"}
        d["columns"] = [c.to_dict() for c in self.columns]
        return d


@dataclass
class JoinCandidate:
    """A proposed join between two files, with the evidence for it."""

    left: str
    right: str
    on: list[str]
    overlap_start: Optional[str] = None
    overlap_end: Optional[str] = None
    overlap_periods: int = 0
    left_only_periods: int = 0
    right_only_periods: int = 0
    grain_match: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class DiscoveryReport:
    """The output of a discovery pass over a folder of raw files."""

    root: str
    files: list[FileProfile] = field(default_factory=list)
    joins: list[JoinCandidate] = field(default_factory=list)
    open_questions: list[dict[str, str]] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    recommended_grain: Optional[str] = None
    recommended_date_range: Optional[tuple[str, str]] = None

    def ask(self, topic: str, question: str, why: str, blocking: bool = False) -> None:
        """Record a question a human must answer. This is the primary output."""
        self.open_questions.append(
            {"topic": topic, "question": question, "why": why, "blocking": "yes" if blocking else "no"}
        )
        if blocking:
            self.blocking.append(f"[{topic}] {question}")

    @property
    def blocking_questions(self) -> list[dict[str, str]]:
        return [q for q in self.open_questions if q["blocking"] == "yes"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": [f.to_dict() for f in self.files],
            "joins": [j.to_dict() for j in self.joins],
            "open_questions": self.open_questions,
            "blocking": self.blocking,
            "warnings": self.warnings,
            "skipped": self.skipped,
            "recommended_grain": self.recommended_grain,
            "recommended_date_range": list(self.recommended_date_range)
            if self.recommended_date_range
            else None,
        }


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def _strip_currency(series: pd.Series) -> tuple[pd.Series, Optional[str]]:
    """Try to turn a text column into numbers, reporting any currency found.

    Marketing exports routinely ship "£1,234.50" and "(2,000)" for negatives.
    Reading those as text and later coercing with ``errors="coerce"`` turns real
    spend into zeros, so we detect it here and say so loudly.
    """
    text = series.dropna().astype(str)
    if text.empty:
        return pd.Series(dtype="float64"), None

    currency = None
    joined = "".join(text.head(200).tolist())
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in joined:
            currency = code
            break

    cleaned = (
        text.str.replace(r"[^\d\.\-\(\)eE]", "", regex=True)
        # Accounting negatives: (1,234) means -1234.
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric, currency


# Formats are tried in this order and applied to the *whole* column. Parsing
# element-by-element (pandas' format="mixed") is what turns "05/06/2022" into
# May 6th and "19/06/2022" into June 19th in the same column, silently
# scrambling the calendar.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d.%m.%Y", "%Y%m%d", "%d %b %Y", "%b %d, %Y", "%d-%b-%Y",
    "%Y-%m", "%Y%m", "%b %Y", "%Y",
)

#: Pairs that are indistinguishable when every day-of-month is <= 12.
_AMBIGUOUS_PAIRS = (("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"))


def parse_date_column(series: pd.Series) -> tuple[pd.Series, Optional[str], list[str]]:
    """Parse a column with ONE format applied consistently to every row.

    Returns the parsed series, the format that won, and any formats that fit the
    data equally well. A non-empty third element means the column is genuinely
    ambiguous — ``03/04/2023`` is the 3rd of April or the 4th of March depending
    on who exported it, and no amount of inspection settles it. That is a
    question for a human, not a default.
    """
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(dtype="datetime64[ns]"), None, []
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce"), "datetime64", []

    text = non_null.astype(str).str.strip()
    scores: dict[str, float] = {}
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        frac = float(parsed.notna().mean())
        if frac >= 0.99:
            scores[fmt] = frac

    if not scores:
        # Cheap pre-filter: most columns handed to this function are campaign
        # names and product codes, and running dateutil over them is slow and
        # noisy for a guaranteed miss.
        if not text.head(50).str.contains(r"\d{4}|\d{1,2}[/\-.]\d{1,2}", regex=True).any():
            return pd.Series(dtype="datetime64[ns]"), None, []
        # Last resort: let pandas infer, accepting that it may vary per element.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
            if float(parsed.notna().mean()) < 0.8:
                return pd.Series(dtype="datetime64[ns]"), None, []
            return pd.to_datetime(series, errors="coerce", dayfirst=False), None, []

    best = next(f for f in _DATE_FORMATS if f in scores)
    ambiguous: list[str] = []
    for a, b in _AMBIGUOUS_PAIRS:
        if a in scores and b in scores:
            # Both fit every row, so every day-of-month is <= 12 and the order
            # cannot be recovered from the data.
            ambiguous = [a, b]
            break

    return pd.to_datetime(series.astype(str).str.strip(), format=best, errors="coerce"), best, ambiguous


def _looks_datelike(series: pd.Series) -> tuple[bool, float]:
    """Fraction of non-null values that parse as dates, and whether that is enough."""
    non_null = series.dropna()
    if non_null.empty:
        return False, 0.0
    if pd.api.types.is_datetime64_any_dtype(series):
        return True, 1.0

    sample = non_null.head(500)
    # A bare number column (1, 2, 3...) is not a date even though pandas will
    # happily read it as nanoseconds since epoch. Only accept the integer forms
    # marketing exports actually use for dates: YYYYMMDD and YYYY.
    if pd.api.types.is_numeric_dtype(sample):
        as_int = pd.to_numeric(sample, errors="coerce").dropna()
        if as_int.empty or (as_int % 1 != 0).any():
            return False, 0.0
        as_str = as_int.astype("int64").astype(str)
        if not as_str.str.fullmatch(r"(19|20)\d{6}|(19|20)\d{2}").all():
            return False, 0.0
        sample = as_str

    parsed, _, _ = parse_date_column(sample)
    frac = float(parsed.notna().mean()) if len(parsed) else 0.0
    return frac >= 0.8, frac


def _infer_grain(dates: pd.Series) -> tuple[Optional[str], float]:
    """Infer daily/weekly/monthly from the modal gap between distinct dates."""
    distinct = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(distinct) < 3:
        return None, 0.0
    gaps = distinct.diff().dropna().dt.days
    if gaps.empty:
        return None, 0.0
    modal = float(gaps.mode().iloc[0])
    share = float((gaps == modal).mean())
    if modal <= 1:
        return "daily", share
    if 6 <= modal <= 8:
        return "weekly", share
    if 28 <= modal <= 31:
        return "monthly", share
    if 89 <= modal <= 92:
        return "quarterly", share
    return None, share


def _classify_role(name: str, prof: ColumnProfile) -> tuple[str, float, str]:
    """Assign a semantic role from the column name, then sanity-check against data."""
    norm = normalize_column(name)

    if _RATE_PATTERN.search(norm):
        return (
            "control",
            0.4,
            "name looks like a rate or ratio (CPC/CPM/ROAS/CTR); a ratio is never spend "
            "and modelling it as media is a category error",
        )

    for role, pattern in _ROLE_PATTERNS:
        if re.search(pattern, norm):
            confidence = 0.8
            reason = f"column name matches the {role} vocabulary"

            # Data-based corrections. The name proposes; the values dispose.
            if role in {"spend", "exposure", "target", "price"} and not (
                prof.is_numeric or prof.numeric_after_cleaning
            ):
                return (
                    "unknown",
                    0.2,
                    f"name suggests {role} but the column is not numeric even after "
                    "stripping currency symbols",
                )
            if role == "date" and not prof.is_datelike:
                return "unknown", 0.2, "name suggests a date but the values do not parse as dates"
            if role in {"geo", "entity"} and prof.n_unique == prof.n_rows and prof.n_rows > 20:
                confidence = 0.5
                reason += "; every value is unique, so this may be a row id rather than a grouping key"
            return role, confidence, reason

    # Nothing matched by name — fall back to shape.
    if prof.is_datelike:
        return "date", 0.5, "values parse as dates although the name does not say so"
    if prof.is_boolean_like:
        return "control", 0.4, "binary column; probably an event or promo flag"
    if prof.is_numeric and prof.n_unique <= 2:
        return "control", 0.35, "numeric with at most two distinct values; probably a flag"
    if not prof.is_numeric and prof.n_unique <= max(50, int(0.05 * max(prof.n_rows, 1))):
        return "entity", 0.3, "low-cardinality text; probably a grouping key"
    return "unknown", 0.0, "no name or shape evidence"


def _guess_source(name: str, columns: Iterable[str]) -> Optional[str]:
    """Guess which platform an export came from, to anticipate its known quirks."""
    haystack = " ".join([normalize_column(name)] + [normalize_column(c) for c in columns])
    signatures: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("meta_ads", ("amount_spent", "adset", "ad_set", "facebook", "meta", "reach_frequency")),
        ("google_ads", ("google_ads", "adwords", "ad_group", "adgroup", "search_impr_share")),
        ("ga4", ("ga4", "google_analytics", "session_source", "sessions", "engaged_sessions")),
        ("tiktok", ("tiktok",)),
        ("linkedin", ("linkedin",)),
        ("dv360", ("dv360", "display_video", "trueview")),
        ("tv_plan", ("grp", "trp", "spot", "daypart", "tvr")),
        ("finance", ("gl_", "general_ledger", "invoice", "net_revenue", "cogs", "actuals", "budget_holder")),
        ("crm", ("crm", "lead_id", "opportunity", "salesforce", "hubspot")),
        ("promo_calendar", ("promo", "promotion", "markdown", "offer_start")),
    )
    for source, tokens in signatures:
        if any(t in haystack for t in tokens):
            return source
    return None


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def profile_dataframe(df: pd.DataFrame, name: str = "dataframe", path: str = "") -> FileProfile:
    """Profile an in-memory frame. ``profile_file`` wraps this after reading."""
    n_rows = len(df)
    prof = FileProfile(path=path or name, name=name, n_rows=n_rows, n_columns=df.shape[1])

    for col in df.columns:
        series = df[col]
        cp = ColumnProfile(
            name=str(col),
            dtype=str(series.dtype),
            n_rows=n_rows,
            n_missing=int(series.isna().sum()),
            n_unique=int(series.nunique(dropna=True)),
        )
        cp.is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        cp.is_datelike, _ = _looks_datelike(series)
        if cp.is_datelike:
            _, cp.date_format, cp.ambiguous_date_formats = parse_date_column(series)
            if cp.ambiguous_date_formats:
                cp.notes.append(
                    "every day-of-month is 12 or less, so day-first and month-first parse "
                    "identically — the true order cannot be recovered from the file"
                )
        cp.is_constant = cp.n_unique <= 1 and n_rows > 1

        non_null = series.dropna()
        cp.sample_values = [str(v) for v in non_null.head(5).tolist()]

        if not cp.is_numeric and not cp.is_datelike and not non_null.empty:
            lowered = set(non_null.astype(str).str.strip().str.lower().unique()[:10])
            if lowered and lowered <= (_TRUE_STRINGS | _FALSE_STRINGS):
                cp.is_boolean_like = True
            else:
                numeric, currency = _strip_currency(series)
                # Require nearly everything to convert; a column with a few
                # numbers in free text is not a numeric column.
                if not numeric.empty and float(numeric.notna().mean()) >= 0.95:
                    cp.numeric_after_cleaning = True
                    cp.detected_currency = currency
                    cp.notes.append(
                        "stored as text but numeric once currency symbols and separators are "
                        "stripped — read it explicitly, never with errors='coerce'"
                    )
                    non_null = numeric.dropna()
        elif cp.is_numeric:
            cp.is_boolean_like = bool(set(non_null.unique()[:5]) <= {0, 1, 0.0, 1.0}) and cp.n_unique <= 2

        if (cp.is_numeric or cp.numeric_after_cleaning) and not non_null.empty:
            values = pd.to_numeric(non_null, errors="coerce").dropna()
            if not values.empty:
                cp.min_value = float(values.min())
                cp.max_value = float(values.max())
                cp.zero_fraction = float((values == 0).mean())
                cp.negative_fraction = float((values < 0).mean())

        cp.role, cp.role_confidence, cp.role_reason = _classify_role(str(col), cp)

        if cp.role in {"spend", "exposure"}:
            guess = classify_channel(str(col))
            if guess != "unknown":
                cp.channel_guess = guess

        if cp.role == "spend" and cp.negative_fraction > 0:
            cp.notes.append(
                f"{cp.negative_fraction:.1%} of values are negative — refunds, credits or "
                "make-goods; decide whether to net them off or zero them"
            )
        if cp.role == "spend" and cp.zero_fraction > 0.9:
            cp.notes.append(
                f"{cp.zero_fraction:.1%} of values are zero — this channel is nearly always dark "
                "and will be hard to identify"
            )
        if cp.is_constant:
            cp.notes.append("constant — carries no information and cannot be modelled")

        prof.columns.append(cp)

    prof.detected_currencies = sorted(
        {c.detected_currency for c in prof.columns if c.detected_currency}
    )
    _infer_file_dates(df, prof)
    _infer_file_shape(df, prof)
    prof.likely_source = _guess_source(name, [str(c) for c in df.columns])
    return prof


def _infer_file_dates(df: pd.DataFrame, prof: FileProfile) -> None:
    """Pick the date column, then read the range and grain from it."""
    candidates = [c for c in prof.columns if c.is_datelike]
    if not candidates:
        return

    def rank(cp: ColumnProfile) -> tuple[int, float]:
        norm = normalize_column(cp.name)
        # Prefer an explicitly named date column over an incidental one
        # (`order_date` beats `created_at` beats a stray parseable string).
        preferred = 0
        for i, token in enumerate(_PREFERRED_DATE_NAMES):
            if norm == token or norm.startswith(token + "_") or norm.endswith("_" + token):
                preferred = len(_PREFERRED_DATE_NAMES) - i
                break
        return preferred, -cp.missing_fraction

    best = max(candidates, key=rank)
    parsed_all, fmt, ambiguous = parse_date_column(df[best.name])
    parsed = parsed_all.dropna()
    if parsed.empty:
        return

    prof.date_column = best.name
    prof.date_format = fmt
    prof.ambiguous_date_formats = ambiguous
    prof.date_confidence = 0.9 if rank(best)[0] > 0 else 0.5
    if ambiguous:
        prof.date_confidence = min(prof.date_confidence, 0.4)
        prof.notes.append(
            f"'{best.name}' is date-format ambiguous ({' or '.join(ambiguous)}); parsed as "
            f"{fmt} but this must be confirmed"
        )
    prof.date_min = str(parsed.min().date())
    prof.date_max = str(parsed.max().date())
    prof.inferred_grain, prof.grain_confidence = _infer_grain(parsed)

    if len(candidates) > 1:
        others = [c.name for c in candidates if c.name != best.name]
        prof.notes.append(
            f"multiple date-like columns ({', '.join([best.name] + others)}); "
            f"using '{best.name}' — confirm which one marks *delivery*, not booking or billing"
        )


def _infer_file_shape(df: pd.DataFrame, prof: FileProfile) -> None:
    """Decide whether the file is one row per period, or many rows per period.

    This is the fork that decides whether the file needs aggregating and
    pivoting before it can join, and it is the question people most often get
    wrong when they hand over "the data".
    """
    if prof.date_column is None:
        prof.shape_kind = "lookup"
        prof.notes.append("no date column — treated as a lookup/dimension table, not a time series")
        return

    dates, _, _ = parse_date_column(df[prof.date_column])
    n_dates = int(dates.nunique(dropna=True))
    if n_dates == 0:
        return

    geo_cols = [c.name for c in prof.columns_with_role("geo") if not c.is_constant]
    entity_cols = [c.name for c in prof.columns_with_role("entity") if not c.is_constant]

    rows_per_date = prof.n_rows / n_dates
    if rows_per_date <= 1.05:
        prof.shape_kind = "wide_timeseries"
        prof.grain_keys = [prof.date_column]
    elif geo_cols and not entity_cols:
        # One row per (date, geo) is still a wide panel.
        keys = [prof.date_column] + geo_cols[:1]
        dupes = int(df.duplicated(subset=keys).sum())
        prof.shape_kind = "wide_timeseries" if dupes == 0 else "long_transactional"
        prof.grain_keys = keys
        prof.duplicate_key_rows = dupes
    else:
        prof.shape_kind = "long_transactional"
        prof.grain_keys = [prof.date_column] + geo_cols[:1] + entity_cols[:2]
        prof.notes.append(
            f"~{rows_per_date:.0f} rows per date — long format; it must be aggregated to the "
            "modelling period and pivoted to one column per channel before joining"
        )

    if prof.grain_keys and prof.shape_kind == "wide_timeseries":
        dupes = int(df.duplicated(subset=prof.grain_keys).sum())
        prof.duplicate_key_rows = dupes
        if dupes:
            prof.notes.append(
                f"{dupes} duplicate rows on {prof.grain_keys} — joining without deduplicating "
                "will multiply spend and target"
            )


def profile_file(path: str | Path, sample_rows: Optional[int] = None) -> FileProfile:
    """Read and profile one file. Read failures are captured, never raised."""
    p = Path(path)
    name = p.name
    try:
        df = _read_any(p, sample_rows)
    except Exception as exc:  # pragma: no cover - exercised via discover()
        return FileProfile(
            path=str(p), name=name, n_rows=0, n_columns=0, read_error=f"{type(exc).__name__}: {exc}"
        )
    return profile_dataframe(df, name=name, path=str(p))


def _read_any(p: Path, sample_rows: Optional[int] = None) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(p, nrows=sample_rows, sep=None, engine="python")
    if ext == ".tsv":
        return pd.read_csv(p, nrows=sample_rows, sep="\t")
    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        return df.head(sample_rows) if sample_rows else df
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p, nrows=sample_rows)
    if ext == ".json":
        df = pd.read_json(p)
        return df.head(sample_rows) if sample_rows else df
    raise ValueError(f"Unsupported extension: {ext}")


# --------------------------------------------------------------------------- #
# Joins
# --------------------------------------------------------------------------- #
def _period_index(prof: FileProfile, df_dates: pd.Series, grain: str) -> pd.DatetimeIndex:
    parsed, _, _ = parse_date_column(df_dates)
    parsed = parsed.dropna()
    if parsed.empty:
        return pd.DatetimeIndex([])
    if grain == "weekly":
        floored = parsed.dt.to_period("W").dt.start_time
    elif grain == "monthly":
        floored = parsed.dt.to_period("M").dt.start_time
    else:
        floored = parsed.dt.normalize()
    return pd.DatetimeIndex(sorted(floored.unique()))


def propose_joins(
    profiles: list[FileProfile], frames: dict[str, pd.DataFrame], grain: str
) -> list[JoinCandidate]:
    """Propose pairwise joins between time-series files and quantify the overlap.

    Overlap is the thing that actually constrains an MMM: a media file covering
    three years joined to a sales file covering eight months yields an eight-month
    model, and people are consistently surprised by this at the wrong moment.
    """
    timeseries = [p for p in profiles if p.date_column and p.shape_kind != "lookup"]
    candidates: list[JoinCandidate] = []

    for i, left in enumerate(timeseries):
        for right in timeseries[i + 1 :]:
            left_df, right_df = frames.get(left.name), frames.get(right.name)
            if left_df is None or right_df is None:
                continue

            left_idx = _period_index(left, left_df[left.date_column], grain)
            right_idx = _period_index(right, right_df[right.date_column], grain)
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            shared = left_idx.intersection(right_idx)
            on = ["date"]
            notes: list[str] = []

            left_geo = [c.name for c in left.columns_with_role("geo") if not c.is_constant]
            right_geo = [c.name for c in right.columns_with_role("geo") if not c.is_constant]
            grain_match = True
            if left_geo and right_geo:
                on.append("geo")
                lvals = set(left_df[left_geo[0]].dropna().astype(str).str.strip().str.lower())
                rvals = set(right_df[right_geo[0]].dropna().astype(str).str.strip().str.lower())
                if lvals and rvals:
                    coverage = len(lvals & rvals) / len(lvals | rvals)
                    if coverage < 0.9:
                        notes.append(
                            f"geo values only overlap {coverage:.0%} "
                            f"('{left_geo[0]}' vs '{right_geo[0]}') — the naming needs a crosswalk "
                            "before this join is safe"
                        )
                        grain_match = False
            elif left_geo or right_geo:
                notes.append(
                    "one file is a geo panel and the other is national — the national file must be "
                    "allocated to geos (by population or by sales share) or the panel collapsed"
                )
                grain_match = False

            if left.inferred_grain and right.inferred_grain and left.inferred_grain != right.inferred_grain:
                notes.append(
                    f"grains differ ({left.inferred_grain} vs {right.inferred_grain}); both must be "
                    f"aggregated to {grain} first — never upsample the coarser one"
                )

            overlap_n = len(shared)
            union_n = len(left_idx.union(right_idx))
            confidence = round(overlap_n / union_n, 3) if union_n else 0.0
            candidates.append(
                JoinCandidate(
                    left=left.name,
                    right=right.name,
                    on=on,
                    overlap_start=str(shared.min().date()) if overlap_n else None,
                    overlap_end=str(shared.max().date()) if overlap_n else None,
                    overlap_periods=overlap_n,
                    left_only_periods=len(left_idx.difference(right_idx)),
                    right_only_periods=len(right_idx.difference(left_idx)),
                    grain_match=grain_match,
                    confidence=confidence,
                    notes=notes,
                )
            )

    candidates.sort(key=lambda c: (-c.overlap_periods, c.left, c.right))
    return candidates


# --------------------------------------------------------------------------- #
# Question generation — the point of the module
# --------------------------------------------------------------------------- #
def _minimum_periods(grain: str) -> int:
    """Periods below which carryover and seasonality cannot both be identified."""
    return {"daily": 365 * 2, "weekly": 104, "monthly": 36}.get(grain, 104)


def _ask_about_files(report: DiscoveryReport) -> None:
    for prof in report.files:
        tag = prof.name

        if prof.read_error:
            report.ask(
                tag,
                f"'{tag}' could not be read ({prof.read_error}). What format is it, and can you "
                "re-export it as CSV or Parquet?",
                "A file we cannot read is a file whose contents we are guessing about.",
                blocking=True,
            )
            continue

        if prof.date_column is None:
            report.ask(
                tag,
                f"'{tag}' has no recognisable date column. Is it a lookup table (a channel "
                "mapping, a geo crosswalk), or does it have a date under an unexpected name?",
                "Everything that enters the model must be placed in time. A lookup table is fine; "
                "an undated time series is not.",
            )
        else:
            if prof.grain_confidence < 0.8 and prof.inferred_grain:
                report.ask(
                    tag,
                    f"'{tag}' looks {prof.inferred_grain} but only {prof.grain_confidence:.0%} of "
                    "the gaps between dates are consistent. Are there missing periods, or is the "
                    "reporting irregular?",
                    "Adstock convolves over consecutive rows. A missing period silently shortens "
                    "carryover for everything after it.",
                )
            if prof.inferred_grain is None:
                report.ask(
                    tag,
                    f"The date spacing in '{tag}' does not match daily, weekly or monthly. What "
                    "period does this file actually report on?",
                    "The modelling grain has to be a real reporting grain, not one we assumed.",
                    blocking=True,
                )
            if prof.date_column and prof.inferred_grain == "weekly":
                report.ask(
                    tag,
                    f"Which day does the week start on in '{tag}'?",
                    "A Monday-start media week joined to a Sunday-start sales week shifts spend one "
                    "day against the target and shows up as a spurious lag.",
                )
            report.ask(
                tag,
                f"Does the date in '{tag}' ('{prof.date_column}') mark when the media was "
                "*delivered*, or when it was booked, billed or reported?",
                "Adstock measures the decay of a delivered impression. Billing dates can sit weeks "
                "away from delivery and will corrupt every carryover estimate.",
            )

        if prof.shape_kind == "long_transactional":
            report.ask(
                tag,
                f"'{tag}' is long format ({prof.n_rows} rows across {prof.grain_keys}). Which "
                "column identifies the channel we should pivot to, and are there rows that must be "
                "excluded (test campaigns, internal traffic, cancelled orders)?",
                "Aggregating the wrong rows into a channel is invisible afterwards: the model fits "
                "and the ROAS is simply wrong.",
                blocking=True,
            )

        if prof.duplicate_key_rows:
            report.ask(
                tag,
                f"'{tag}' has {prof.duplicate_key_rows} duplicate rows on {prof.grain_keys}. Are "
                "these genuine repeats to be summed, restatements where the latest wins, or an "
                "export defect?",
                "Each answer produces a different number and there is no way to tell them apart "
                "from the file.",
                blocking=True,
            )

        if len(prof.detected_currencies) > 1:
            report.ask(
                tag,
                f"'{tag}' contains more than one currency ({', '.join(prof.detected_currencies)}). "
                "What rate should we convert at, and as of when?",
                "Converting at today's rate rewrites history; converting at the transaction rate "
                "mixes FX movement into media response.",
                blocking=True,
            )

        for cp in prof.columns:
            if cp.numeric_after_cleaning:
                report.ask(
                    tag,
                    f"'{cp.name}' in '{tag}' is stored as text (e.g. {cp.sample_values[:2]}). "
                    "Confirm it is a plain number and not a formatted or annotated field.",
                    "A careless to_numeric(errors='coerce') turns these into zeros, which reads as "
                    "'the channel was dark' rather than 'we failed to parse it'.",
                )
            if cp.role == "control" and _RATE_PATTERN.search(normalize_column(cp.name)):
                report.ask(
                    tag,
                    f"'{cp.name}' in '{tag}' looks like a rate or ratio. Do we have the numerator "
                    "and denominator separately (spend and impressions, conversions and clicks)?",
                    "A ratio cannot carry adstock or saturation and must never be used as a channel "
                    "input. The components can.",
                )
            if cp.role == "spend" and cp.negative_fraction > 0.01:
                report.ask(
                    tag,
                    f"'{cp.name}' in '{tag}' is negative in {cp.negative_fraction:.1%} of rows. Are "
                    "these refunds, credits or make-goods, and should they net off the period they "
                    "appear in or the period they relate to?",
                    "Negative spend in a saturation curve is undefined behaviour in every framework.",
                )
            uncertain = cp.role == "unknown" or cp.role_confidence < 0.5
            if uncertain and not cp.is_constant and cp.missing_fraction < 0.5:
                guess = (
                    "we could not classify it"
                    if cp.role == "unknown"
                    else f"we guessed '{cp.role}' at {cp.role_confidence:.0%} confidence"
                )
                report.ask(
                    tag,
                    f"What is '{cp.name}' in '{tag}'? (sample: {cp.sample_values[:3]}; {guess})",
                    "An unclassified or weakly classified column is either a missing control or a "
                    "leak. Both matter, and we cannot tell which without you.",
                )


def _ask_about_structure(report: DiscoveryReport) -> None:
    all_roles: dict[str, list[str]] = {r: [] for r in SEMANTIC_ROLES}
    for prof in report.files:
        for cp in prof.columns:
            all_roles[cp.role].append(f"{prof.name}:{cp.name}")

    if not all_roles["target"]:
        report.ask(
            "target",
            "No column looks like a target (revenue, orders, conversions). Which file and column "
            "holds the outcome the business wants to move?",
            "Without a target there is nothing to attribute.",
            blocking=True,
        )
    elif len(all_roles["target"]) > 1:
        report.ask(
            "target",
            f"Several columns could be the target ({', '.join(all_roles['target'][:6])}). Which one "
            "is the decision made on, and is it gross or net of returns, tax and discounts?",
            "Modelling gross revenue and reporting net ROAS overstates every channel by the return "
            "rate.",
            blocking=True,
        )

    if not all_roles["spend"]:
        report.ask(
            "media",
            "No spend columns were found. Is media cost held somewhere else, or are we modelling "
            "exposure only?",
            "Without spend there is no ROAS and no budget optimisation — only relative effect sizes.",
            blocking=True,
        )

    if all_roles["exposure"] and all_roles["spend"]:
        report.ask(
            "media",
            "Both spend and exposure columns exist. For each channel, which should drive the "
            "transformation?",
            "Where buying is efficiency-driven (auction channels), exposure is the better input and "
            "spend stays only for ROAS. Where price per unit is stable, spend is simpler.",
        )

    if not all_roles["price"]:
        report.ask(
            "confounders",
            "No price or discount column was found. Does price move over the modelling window?",
            "If price moves and is not controlled, media takes credit for promotions. This is the "
            "single most common source of inflated ROAS in retail and CPG.",
        )

    if not all_roles["distribution"]:
        report.ask(
            "confounders",
            "Is there a distribution, store-count or availability measure?",
            "Growth from opening stores looks exactly like growth from advertising when it is not "
            "controlled.",
        )

    report.ask(
        "experiments",
        "Has any channel ever been tested — a geo holdout, a lift study, a switch-off, a matched "
        "market test? Even an informal one, and even one with a disappointing result.",
        "An MMM without an experiment is an argument from correlation. One usable test anchors the "
        "whole decomposition and changes which model we should build.",
        blocking=True,
    )
    report.ask(
        "channels",
        "For each channel: is it always on, and has its spend level changed materially in the "
        "window?",
        "A channel that never varies cannot be measured observationally, whatever the model reports. "
        "Naming these up front prevents presenting a prior as a finding.",
    )
    report.ask(
        "channels",
        "Which of these are paid media, which are organic, and which are business levers rather "
        "than media (price, promotion, distribution, product launches)?",
        "Roles decide structure. An organic channel given a ROAS, or a price index given adstock, is "
        "a category error no diagnostic will catch.",
        blocking=True,
    )
    report.ask(
        "history",
        "What happened in this window that is not in the data — a rebrand, a site migration, a "
        "tracking change, a stockout, a competitor entering, a pandemic?",
        "Structural breaks explain more variance than most channels and will otherwise be absorbed "
        "into whichever channel happened to move at the time.",
    )
    report.ask(
        "decision",
        "What decision will this model inform, by when, and what would change if the answer came "
        "back differently?",
        "This decides the required precision, the channel granularity and whether an MMM is even the "
        "right instrument.",
        blocking=True,
    )


def _summarise_coverage(report: DiscoveryReport) -> None:
    """Pick the modelling grain and window, and warn when they are too small."""
    grains = [p.inferred_grain for p in report.files if p.inferred_grain]
    if grains:
        order = ["daily", "weekly", "monthly", "quarterly"]
        rank = {g: n for n, g in enumerate(order)}

        # The modelling grain is the one most sources already report on — not the
        # coarsest. Letting a single monthly TV plan drag a weekly model down to
        # 24 observations is a much worse trade than deciding what to do about
        # that one file.
        modal = pd.Series(grains).mode().iloc[0]
        report.recommended_grain = "weekly" if modal == "daily" else modal

        if modal == "daily":
            report.warnings.append(
                "Most sources are daily. Weekly is still the default modelling grain: day-of-week "
                "noise dominates daily media response and most media is planned and bought weekly."
            )

        coarser = [
            p for p in report.files
            if p.inferred_grain
            and rank.get(p.inferred_grain, 0) > rank.get(report.recommended_grain, 1)
        ]
        for prof in coarser:
            report.ask(
                "grain",
                f"'{prof.name}' is {prof.inferred_grain} while the rest of the data supports "
                f"{report.recommended_grain}. Can it be re-exported at {report.recommended_grain}, "
                "should we model at the coarser grain, or do we have a schedule to allocate it "
                "down (broadcast calendar, spot log, flight dates)?",
                f"Splitting {prof.inferred_grain} totals evenly across {report.recommended_grain} "
                "periods invents variation the model will read as real, and modelling everything at "
                f"{prof.inferred_grain} costs most of the observations.",
                blocking=True,
            )
        if coarser:
            report.warnings.append(
                f"{len(coarser)} source(s) are coarser than the proposed {report.recommended_grain} "
                "grain: " + ", ".join(f"{p.name} ({p.inferred_grain})" for p in coarser)
            )

    starts = [p.date_min for p in report.files if p.date_min]
    ends = [p.date_max for p in report.files if p.date_max]
    if starts and ends:
        overlap_start, overlap_end = max(starts), min(ends)
        if overlap_start <= overlap_end:
            report.recommended_date_range = (overlap_start, overlap_end)
            span_days = (pd.Timestamp(overlap_end) - pd.Timestamp(overlap_start)).days
            grain = report.recommended_grain or "weekly"
            per_period = {"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 91}.get(grain, 7)
            periods = int(span_days // per_period) + 1
            needed = _minimum_periods(grain)
            if periods < needed:
                report.warnings.append(
                    f"The files only overlap from {overlap_start} to {overlap_end} — about "
                    f"{periods} {grain} periods against a practical minimum of {needed}. Two full "
                    "years is the usual floor because seasonality and carryover are otherwise "
                    "confounded."
                )
                report.ask(
                    "coverage",
                    f"The usable window is only {periods} {grain} periods. Can history be extended, "
                    "or should we model fewer channels, or move to a geo panel to buy back "
                    "observations?",
                    "Below roughly two years, the model cannot separate an annual seasonal cycle "
                    "from a channel that happens to be seasonal.",
                    blocking=True,
                )
        else:
            report.warnings.append(
                f"The files do not overlap in time at all (latest start {overlap_start} is after "
                f"earliest end {overlap_end})."
            )
            report.ask(
                "coverage",
                "The supplied files share no common date range. Are some of them extracts from "
                "different periods, and is there a version that covers the same window?",
                "There is no dataset to build until the sources overlap.",
                blocking=True,
            )


def discover(
    root: str | Path,
    *,
    patterns: Optional[Iterable[str]] = None,
    sample_rows: Optional[int] = None,
    max_files: int = 50,
) -> DiscoveryReport:
    """Profile every readable file under ``root`` and produce a discovery report.

    This is deliberately read-only. It changes nothing, decides nothing, and its
    most valuable output is ``open_questions`` — the list of things a human has
    to settle before the data engineering can be called correct.
    """
    root_path = Path(root)
    report = DiscoveryReport(root=str(root_path))
    if not root_path.exists():
        raise FileNotFoundError(f"Discovery root not found: {root}")

    if root_path.is_file():
        paths = [root_path]
    else:
        globs = list(patterns) if patterns else ["**/*"]
        seen: set[Path] = set()
        paths = []
        root_resolved = root_path.resolve()
        for g in globs:
            for p in sorted(root_path.glob(g)):
                if not p.is_file() or p in seen:
                    continue
                # Stay inside the folder we were pointed at. A symlink out of the
                # data drop is almost always an accident, and silently profiling
                # files from elsewhere on the machine is never what was wanted.
                try:
                    resolved = p.resolve()
                    resolved.relative_to(root_resolved)
                except (ValueError, OSError):
                    report.skipped.append(
                        {"path": str(p), "reason": "symlink or path resolving outside the root"}
                    )
                    continue
                seen.add(p)
                paths.append(p)

    frames: dict[str, pd.DataFrame] = {}
    for p in paths:
        if len(report.files) >= max_files:
            report.warnings.append(
                f"Stopped after {max_files} files. Narrow the search with `patterns` if more need "
                "profiling."
            )
            break
        if p.suffix.lower() not in READABLE_EXTENSIONS:
            report.skipped.append({"path": str(p), "reason": f"unsupported extension '{p.suffix}'"})
            continue
        if p.name.startswith("~$") or p.name.startswith("."):
            report.skipped.append({"path": str(p), "reason": "temporary or hidden file"})
            continue

        prof = profile_file(p, sample_rows=sample_rows)
        report.files.append(prof)
        if prof.read_error is None:
            try:
                frames[prof.name] = _read_any(p, sample_rows)
            except Exception:  # pragma: no cover - already profiled successfully
                pass

    if not report.files:
        report.warnings.append(f"No readable data files found under {root_path}.")
        return report

    _summarise_coverage(report)
    report.joins = propose_joins(report.files, frames, report.recommended_grain or "weekly")
    _ask_about_files(report)
    _ask_about_structure(report)

    for join in report.joins:
        if join.overlap_periods == 0:
            report.ask(
                "joins",
                f"'{join.left}' and '{join.right}' share no periods. Are they meant to be joined at "
                "all?",
                "Two files that never overlap cannot both inform the same model.",
            )
        elif not join.grain_match:
            report.ask(
                "joins",
                f"'{join.left}' and '{join.right}' do not align on their keys "
                f"({'; '.join(join.notes) or 'differing grain'}). How should they be reconciled?",
                "A join across mismatched keys drops rows quietly and biases whatever survives.",
                blocking=True,
            )

    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_range(prof: FileProfile) -> str:
    if prof.date_min and prof.date_max:
        return f"{prof.date_min} → {prof.date_max}"
    return "—"


def render_discovery_report(report: DiscoveryReport) -> str:
    """Render the discovery report as Markdown for a human to read and answer."""
    lines: list[str] = []
    a = lines.append

    a("# Data Discovery Report")
    a("")
    a(f"Source: `{report.root}`")
    a(f"Files profiled: {len(report.files)}  ")
    a(f"Open questions: {len(report.open_questions)} ({len(report.blocking_questions)} blocking)")
    a("")
    a("This report describes what the files appear to contain. It does not decide anything.")
    a("Every blocking question below must be answered before the dataset can be trusted.")
    a("")

    if report.recommended_grain or report.recommended_date_range:
        a("## Proposed modelling frame")
        a("")
        if report.recommended_grain:
            a(f"* **Grain:** {report.recommended_grain}")
        if report.recommended_date_range:
            start, end = report.recommended_date_range
            a(f"* **Usable window:** {start} → {end} (where all sources overlap)")
        a("")

    if report.blocking_questions:
        a("## Blocking — answer these first")
        a("")
        for q in report.blocking_questions:
            a(f"**[{q['topic']}]** {q['question']}")
            a("")
            a(f"> *Why it matters:* {q['why']}")
            a("")

    a("## Files")
    a("")
    a("| File | Rows | Cols | Date column | Range | Grain | Shape | Likely source |")
    a("|---|---:|---:|---|---|---|---|---|")
    for p in report.files:
        if p.read_error:
            a(f"| `{p.name}` | — | — | — | — | — | **unreadable** | — |")
            continue
        a(
            f"| `{p.name}` | {p.n_rows} | {p.n_columns} | {p.date_column or '—'} | "
            f"{_fmt_range(p)} | {p.inferred_grain or '?'} | {p.shape_kind} | "
            f"{p.likely_source or '—'} |"
        )
    a("")

    for p in report.files:
        if p.read_error:
            a(f"### `{p.name}` — could not be read")
            a("")
            a(f"`{p.read_error}`")
            a("")
            continue

        a(f"### `{p.name}`")
        a("")
        if p.notes:
            for n in p.notes:
                a(f"* {n}")
            a("")
        a("| Column | Role | Conf. | Type | Missing | Unique | Notes |")
        a("|---|---|---:|---|---:|---:|---|")
        for c in sorted(p.columns, key=lambda c: (SEMANTIC_ROLES.index(c.role), c.name)):
            notes = "; ".join(c.notes) if c.notes else ""
            if c.channel_guess:
                notes = f"channel guess: {c.channel_guess}" + (f"; {notes}" if notes else "")
            a(
                f"| `{c.name}` | {c.role} | {c.role_confidence:.1f} | {c.dtype} | "
                f"{c.missing_fraction:.0%} | {c.n_unique} | {notes} |"
            )
        a("")

    if report.joins:
        a("## Proposed joins")
        a("")
        a("| Left | Right | On | Overlap | Periods | Left only | Right only | Aligned |")
        a("|---|---|---|---|---:|---:|---:|---|")
        for j in report.joins:
            window = f"{j.overlap_start} → {j.overlap_end}" if j.overlap_periods else "none"
            a(
                f"| `{j.left}` | `{j.right}` | {', '.join(j.on)} | {window} | "
                f"{j.overlap_periods} | {j.left_only_periods} | {j.right_only_periods} | "
                f"{'yes' if j.grain_match else '**no**'} |"
            )
        a("")
        for j in report.joins:
            for n in j.notes:
                a(f"* `{j.left}` ↔ `{j.right}`: {n}")
        a("")

    if report.warnings:
        a("## Warnings")
        a("")
        for w in report.warnings:
            a(f"* {w}")
        a("")

    non_blocking = [q for q in report.open_questions if q["blocking"] == "no"]
    if non_blocking:
        a("## Questions to work through")
        a("")
        by_topic: dict[str, list[dict[str, str]]] = {}
        for q in non_blocking:
            by_topic.setdefault(q["topic"], []).append(q)
        for topic, qs in by_topic.items():
            a(f"### {topic}")
            a("")
            for q in qs:
                a(f"* {q['question']}")
                a(f"  * *Why:* {q['why']}")
            a("")

    if report.skipped:
        a("## Skipped")
        a("")
        for s in report.skipped:
            a(f"* `{s['path']}` — {s['reason']}")
        a("")

    return "\n".join(lines)
