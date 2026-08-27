"""Data engineering for MMM: make the dataset mean what the spec says it means.

Every function here is explicit about *what it assumes* and records what it did,
because the dangerous transformations in MMM are the silent ones. A blanket
``fillna(0)`` turns "we did not receive this file" into "we spent nothing",
which is a claim about the world; resampling with the wrong aggregation turns a
price index into nonsense. Nothing in this module guesses on your behalf without
saying so in the returned report.

Typical order of operations:

1. :func:`standardise_dates` — parse, sort, and check the declared frequency.
2. :func:`aggregate_to_period` — collapse daily rows to the modelling period.
3. :func:`reindex_complete` — make the panel rectangular and the calendar dense.
4. :func:`impute` — fill gaps, per column, with a stated rule.
5. :func:`deflate` / :func:`convert_currency` — put money on one comparable scale.
6. :func:`add_calendar_features` / :func:`add_event_flags` — build controls.
7. :func:`flag_outliers` — mark, do not delete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from agent_mmm.spec import ChannelRole, DataGranularity, MMMSpec

FREQ_ALIAS = {
    DataGranularity.daily: "D",
    DataGranularity.weekly: "W-MON",
    DataGranularity.monthly: "MS",
}

ImputeStrategy = Literal["zero", "ffill", "interpolate", "median", "group_median", "drop", "leave"]


@dataclass
class PrepReport:
    """A record of every change made to the data."""

    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def act(self, msg: str) -> None:
        self.actions.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_markdown(self) -> str:
        lines = ["# Data Preparation Report", ""]
        lines += [f"Rows: {self.rows_before} → {self.rows_after}", ""]
        if self.actions:
            lines += ["## Changes made", ""] + [f"- {a}" for a in self.actions] + [""]
        if self.warnings:
            lines += ["## Judgement calls to confirm", ""] + [f"- {w}" for w in self.warnings] + [""]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def standardise_dates(
    df: pd.DataFrame, date_column: str, granularity: DataGranularity, report: PrepReport | None = None
) -> pd.DataFrame:
    """Parse dates, sort, and compare the observed cadence to the declared one."""
    report = report or PrepReport()
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column])
    out = out.sort_values(date_column).reset_index(drop=True)

    gaps = pd.Series(sorted(out[date_column].unique())).diff().dt.days.dropna()
    if len(gaps):
        modal = float(gaps.mode().iloc[0])
        expected = {DataGranularity.daily: 1, DataGranularity.weekly: 7, DataGranularity.monthly: 30}[
            granularity
        ]
        if granularity != DataGranularity.monthly and abs(modal - expected) > 0.5:
            report.warn(
                f"Declared {granularity.value} data but rows are typically {modal:.0f} days apart. "
                "Half-lives, l_max and every carryover prior are expressed in periods, so this "
                "mismatch rescales the whole model."
            )
    report.act(f"Parsed and sorted '{date_column}'.")
    return out


def infer_aggregation(column: str, spec: MMMSpec) -> str:
    """How a column should be combined when collapsing periods.

    Flows add up (spend, impressions, sales). States and rates do not: averaging
    a price index is right, summing it is meaningless. Getting this wrong is a
    quiet way to destroy a control variable.
    """
    ch = spec.channel_by_name(column)
    if ch is not None:
        if ch.role == ChannelRole.non_media_treatment:
            return "mean"
        if ch.frequency_column == column:
            return "mean"
        return "sum"
    ctrl = next((c for c in spec.controls if c.column == column), None)
    if ctrl is not None:
        from agent_mmm.spec import ControlCategory

        if ctrl.category in (
            ControlCategory.pricing,
            ControlCategory.macro,
            ControlCategory.distribution,
        ):
            return "mean"
        return "sum"
    if column == spec.target_column:
        return "sum"
    if spec.geo.population_column and column == spec.geo.population_column:
        return "mean"
    return "sum"


def aggregate_to_period(
    df: pd.DataFrame,
    spec: MMMSpec,
    target_granularity: DataGranularity | None = None,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Collapse a finer series to the modelling period, one rule per column.

    Weekly is the usual choice for MMM: daily data is dominated by day-of-week
    noise and by the fact that most media is planned weekly, while monthly data
    gives too few observations to identify carryover at all.
    """
    report = report or PrepReport()
    target = target_granularity or spec.granularity
    freq = FREQ_ALIAS[target]
    date_col = spec.date_column

    group_keys: list[Any] = [pd.Grouper(key=date_col, freq=freq)]
    if spec.geo.is_panel and spec.geo.geo_column:
        group_keys.insert(0, spec.geo.geo_column)

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    agg = {c: infer_aggregation(c, spec) for c in numeric}
    out = df.groupby(group_keys, dropna=False).agg(agg).reset_index()

    summed = sorted(c for c, how in agg.items() if how == "sum")
    meaned = sorted(c for c, how in agg.items() if how == "mean")
    report.act(
        f"Aggregated to {target.value} ({freq}). Summed: {summed}. Averaged: {meaned}."
    )
    if meaned:
        report.warn(
            f"Averaged when collapsing periods: {meaned}. Confirm each is a rate or a state "
            "(price, distribution, frequency) rather than a flow — summing a flow and "
            "averaging a state are both correct, and swapping them is not detectable later."
        )
    return out


def reindex_complete(
    df: pd.DataFrame, spec: MMMSpec, report: PrepReport | None = None
) -> pd.DataFrame:
    """Insert every missing period (and geo-period cell) as an explicit row.

    Adstock convolves over consecutive positions, so a missing week is not a gap
    the model works around — it silently shortens the carryover for everything
    that follows. Panel models additionally *require* a rectangular grid.
    Inserted cells carry NaN, not zero; :func:`impute` decides what they mean.
    """
    report = report or PrepReport()
    date_col = spec.date_column
    freq = FREQ_ALIAS[spec.granularity]
    full_dates = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)

    if spec.geo.is_panel and spec.geo.geo_column:
        geos = sorted(df[spec.geo.geo_column].dropna().unique())
        grid = pd.MultiIndex.from_product([geos, full_dates], names=[spec.geo.geo_column, date_col])
        out = (
            df.set_index([spec.geo.geo_column, date_col])
            .reindex(grid)
            .reset_index()
        )
    else:
        out = df.set_index(date_col).reindex(full_dates).rename_axis(date_col).reset_index()

    added = len(out) - len(df)
    if added > 0:
        report.act(f"Inserted {added} missing period rows as NaN (not zero).")
        report.warn(
            f"{added} periods were absent from the source. Decide what each missing value means "
            "per column before imputing: absent spend is usually 0, an absent target is usually "
            "an unusable period."
        )
    return out


def impute(
    df: pd.DataFrame,
    strategies: dict[str, ImputeStrategy],
    spec: MMMSpec | None = None,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Fill missing values with an explicitly chosen rule per column.

    There is no default. Choosing a strategy is a statement about why the data is
    missing, and the right statement differs by column:

    ``zero``
        "No activity happened." Correct for media spend and exposure.
    ``ffill``
        "The last known state persisted." Correct for price, distribution,
        store count — anything that changes only when someone changes it.
    ``interpolate``
        "The series moved smoothly through the gap." Reasonable for slow macro
        series, dangerous for anything spiky.
    ``median`` / ``group_median``
        "This period was typical." A last resort; it shrinks variance and
        therefore biases the coefficient towards zero.
    ``drop``
        "This period is unusable." The honest choice for a missing target,
        provided you drop few enough rows to keep the series dense.
    ``leave``
        Keep the NaN and let the modelling framework decide.
    """
    report = report or PrepReport()
    out = df.copy()
    to_drop = pd.Series(False, index=out.index)

    for col, how in strategies.items():
        if col not in out.columns:
            report.warn(f"Imputation requested for '{col}', which is not in the data.")
            continue
        n_missing = int(out[col].isna().sum())
        if n_missing == 0:
            continue

        if how == "zero":
            out[col] = out[col].fillna(0)
        elif how == "ffill":
            if spec and spec.geo.is_panel and spec.geo.geo_column:
                out[col] = out.groupby(spec.geo.geo_column)[col].ffill().bfill()
            else:
                out[col] = out[col].ffill().bfill()
        elif how == "interpolate":
            out[col] = out[col].interpolate(limit_direction="both")
        elif how == "median":
            out[col] = out[col].fillna(out[col].median())
        elif how == "group_median":
            if spec and spec.geo.is_panel and spec.geo.geo_column:
                out[col] = out[col].fillna(out.groupby(spec.geo.geo_column)[col].transform("median"))
            else:
                out[col] = out[col].fillna(out[col].median())
        elif how == "drop":
            to_drop |= out[col].isna()
            continue
        elif how == "leave":
            continue

        report.act(f"Imputed {n_missing} missing values in '{col}' using '{how}'.")
        if how in ("median", "group_median"):
            report.warn(
                f"'{col}': {n_missing} values replaced with a median. This removes variation the "
                "model needs and biases that variable's coefficient towards zero. Prefer a real "
                "backfill from the source if one exists."
            )

    if to_drop.any():
        n = int(to_drop.sum())
        out = out.loc[~to_drop].reset_index(drop=True)
        report.act(f"Dropped {n} rows with unusable values.")
        report.warn(
            f"Dropping {n} rows breaks the date sequence unless they are at the edges. "
            "Check that the remaining series is still contiguous before fitting."
        )
    return out


def suggest_imputation(spec: MMMSpec, df: pd.DataFrame) -> dict[str, ImputeStrategy]:
    """Propose a strategy per column from the spec's declared roles.

    A suggestion, not a decision: review it before passing it to :func:`impute`.
    """
    from agent_mmm.spec import ControlCategory

    out: dict[str, ImputeStrategy] = {}
    for ch in spec.active_channels():
        for col in (ch.column, ch.spend_column, ch.exposure_column, ch.reach_column):
            if col and col in df.columns:
                out[col] = "zero" if ch.role != ChannelRole.non_media_treatment else "ffill"
        if ch.frequency_column and ch.frequency_column in df.columns:
            # Frequency is undefined when reach is zero; zero is the right fill.
            out[ch.frequency_column] = "zero"
    for ctrl in spec.controls:
        if ctrl.column not in df.columns:
            continue
        if ctrl.category in (ControlCategory.pricing, ControlCategory.distribution):
            out[ctrl.column] = "ffill"
        elif ctrl.category == ControlCategory.macro:
            out[ctrl.column] = "interpolate"
        elif ctrl.category == ControlCategory.calendar:
            out[ctrl.column] = "zero"
        else:
            out[ctrl.column] = "ffill"
    if spec.target_column in df.columns:
        out[spec.target_column] = "drop"
    if spec.geo.population_column and spec.geo.population_column in df.columns:
        out[spec.geo.population_column] = "ffill"
    return out


# --------------------------------------------------------------------------- #
def deflate(
    df: pd.DataFrame,
    money_columns: Iterable[str],
    price_index: pd.Series,
    base_period: Any = None,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Convert nominal money to real terms using a price index.

    Over three years of history, inflation alone can make late spend look 15%
    larger than early spend at identical delivery. The model reads that as more
    media and attributes the difference to effectiveness. Deflate both spend and
    a monetary target, or neither — deflating one side manufactures a trend.
    """
    report = report or PrepReport()
    out = df.copy()
    idx = price_index.reindex(out.index) if len(price_index) == len(out) else price_index
    base = float(idx.loc[base_period]) if base_period is not None else float(idx.iloc[-1])
    factor = base / idx.astype(float)
    for col in money_columns:
        if col in out.columns:
            out[col] = out[col] * factor
    report.act(f"Deflated {list(money_columns)} to base period {base_period or 'last'}.")
    return out


def convert_currency(
    df: pd.DataFrame,
    money_columns: Iterable[str],
    fx_rate: pd.Series | float,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Convert money columns with a per-period or constant FX rate.

    Use the rate that was in force when the money was spent, not today's: a
    constant rate applied retrospectively removes real variation in local
    purchasing power, while a per-period rate applied to a target measured in a
    different currency introduces variation that no channel caused.
    """
    report = report or PrepReport()
    out = df.copy()
    for col in money_columns:
        if col in out.columns:
            out[col] = out[col] * (fx_rate if np.isscalar(fx_rate) else fx_rate.values)
    report.act(f"Converted {list(money_columns)} with {'a constant' if np.isscalar(fx_rate) else 'a per-period'} FX rate.")
    return out


def add_calendar_features(
    df: pd.DataFrame,
    date_column: str,
    country: str | None = None,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Add calendar controls: quarter flags, week of year, and holiday proximity.

    Fourier seasonality handles smooth annual shape well and sharp calendar
    events badly. Christmas, Black Friday and Ramadan are step changes; give
    them their own indicators so the model does not credit whichever campaign
    happened to run alongside them.
    """
    report = report or PrepReport()
    out = df.copy()
    d = pd.to_datetime(out[date_column])
    out["year"] = d.dt.year
    out["quarter"] = d.dt.quarter
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    for q in (1, 2, 3, 4):
        out[f"is_q{q}"] = (out["quarter"] == q).astype(int)
    added = ["year", "quarter", "week_of_year", "month"] + [f"is_q{q}" for q in (1, 2, 3, 4)]

    if country:
        try:
            import holidays as holidays_pkg

            years = sorted(out["year"].unique().tolist())
            cal = holidays_pkg.country_holidays(country, years=years)
            holiday_dates = pd.to_datetime(sorted(cal.keys()))
            # A weekly row "contains" a holiday if the holiday falls in its span.
            span = (d.diff().dt.days.mode().iloc[0] if len(d) > 1 else 7) or 7
            flags = np.zeros(len(out), dtype=int)
            for i, start in enumerate(d):
                end = start + pd.Timedelta(days=int(span) - 1)
                flags[i] = int(((holiday_dates >= start) & (holiday_dates <= end)).any())
            out["is_holiday_period"] = flags
            added.append("is_holiday_period")
            report.act(f"Added holiday flags for {country} ({len(holiday_dates)} dates).")
        except Exception as e:
            report.warn(f"Could not build holiday flags for '{country}': {e}")

    report.act(f"Added calendar features: {added}.")
    report.warn(
        "Calendar dummies and Fourier terms both explain seasonality. Using many of both "
        "leaves little variance for media; keep Fourier for the smooth annual shape and "
        "dummies only for sharp, named events."
    )
    return out


def add_event_flags(
    df: pd.DataFrame,
    date_column: str,
    events: dict[str, tuple[str, str]],
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Add a 0/1 indicator per named event window ``{name: (start, end)}``.

    Use these for the things that genuinely broke the series: a relaunch, a
    system migration, a stockout, a competitor's exit. An unmodelled regime
    change does not stay unmodelled — it gets attributed to whatever moved at
    the same time.
    """
    report = report or PrepReport()
    out = df.copy()
    d = pd.to_datetime(out[date_column])
    for name, (start, end) in events.items():
        out[name] = ((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))).astype(int)
        report.act(f"Added event flag '{name}' for {start}..{end} ({int(out[name].sum())} periods).")
    return out


def flag_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    z_threshold: float = 4.0,
    report: PrepReport | None = None,
) -> pd.DataFrame:
    """Mark extreme values with an indicator column — never delete them.

    A spike is a data point with an explanation attached. Deleting it throws
    away the explanation and the observation; flagging it lets the model
    estimate the spike's effect separately, and lets a reviewer ask what
    happened that week.
    """
    report = report or PrepReport()
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        s = out[col].astype(float)
        if s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        flag = (z.abs() > z_threshold).astype(int)
        if flag.sum():
            out[f"outlier_{col}"] = flag
            report.act(f"Flagged {int(flag.sum())} outliers in '{col}' (|z| > {z_threshold}).")
            report.warn(
                f"'{col}' has {int(flag.sum())} extreme periods. Explain each one before deciding "
                "whether to keep the flag as a control, switch to a StudentT likelihood, or fix "
                "the underlying data."
            )
    return out


# --------------------------------------------------------------------------- #
def prepare_dataset(
    spec: MMMSpec,
    df: pd.DataFrame | None = None,
    impute_strategies: dict[str, ImputeStrategy] | None = None,
    aggregate: bool = False,
    add_calendar: bool = False,
) -> tuple[pd.DataFrame, PrepReport]:
    """Run the standard preparation pipeline and return the data plus its report.

    Deliberately conservative: it standardises dates, optionally aggregates,
    makes the calendar dense, and imputes with rules you (or
    :func:`suggest_imputation`) chose. It does not deflate, convert currency or
    drop anything you did not ask it to.
    """
    from agent_mmm.utils.io import load_data

    if df is None:
        df = load_data(spec.data_path)
    report = PrepReport(rows_before=len(df))

    out = standardise_dates(df, spec.date_column, spec.granularity, report)
    if aggregate:
        out = aggregate_to_period(out, spec, report=report)
    out = reindex_complete(out, spec, report=report)

    strategies = impute_strategies or suggest_imputation(spec, out)
    report.details["impute_strategies"] = dict(strategies)
    out = impute(out, strategies, spec=spec, report=report)

    if add_calendar:
        out = add_calendar_features(out, spec.date_column, spec.seasonality.holiday_country, report)

    report.rows_after = len(out)
    return out, report
