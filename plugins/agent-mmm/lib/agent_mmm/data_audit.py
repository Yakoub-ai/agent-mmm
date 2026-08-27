"""Data audit: find the problems that make an MMM wrong before it makes them.

Most bad MMMs are not bad models — they are good models fitted to data that does
not mean what the modeller thinks. The checks here are grouped by what they can
tell you, and each one names the *modelling consequence* rather than just the
statistic, because "VIF is 14" is not actionable and "SEM and Social always move
together, so the model cannot tell their effects apart" is.

Structure of the audit:

* **Contract**   — do the columns the spec promises exist, with the right types?
* **Shape**      — enough periods, regular dates, rectangular panel, no duplicates.
* **Integrity**  — impossible values, missing data, sudden regime changes.
* **Identifiability** — collinearity, flat channels, parameter budget.
* **Signal**     — does the target actually move, and does anything explain it?
* **Semantics**  — do spend and exposure imply sane costs? Is price present?

Findings carry a severity (``error`` blocks modelling, ``warning`` shapes it)
and a tier is derived from them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from agent_mmm.spec import ChannelRole, ControlCategory, DataGranularity, MMMSpec
from agent_mmm.utils.breaks import detect_structural_break
from agent_mmm.utils.io import load_data, parse_dates, validate_columns
from agent_mmm.utils.seasonality import seasonal_strength
from agent_mmm.utils.vif import compute_vif
from agent_mmm.workspace import ensure_workspace

# --- thresholds ------------------------------------------------------------ #
MIN_PERIODS_HARD = 52
MIN_PERIODS_RECOMMENDED = 104
SPARSE_ZERO_THRESHOLD = 0.30
FLAT_CV_THRESHOLD = 0.15
VIF_WARN = 5.0
VIF_CRITICAL = 10.0
PAIR_CORR_WARN = 0.80
SKEWNESS_WARN = 2.0
TARGET_CV_MIN = 0.05
OUTLIER_Z = 4.0
# A lagged correlation only says something if it is large enough to be signal;
# below this, 'best lag' is just where the noise happened to peak.
LAG_CORR_MIN = 0.15
SEASONALITY_LOW = 0.05
SEASONALITY_HIGH = 0.50
OBS_PER_PARAM_MIN = 10.0
CPM_DRIFT_RATIO = 5.0

# Backwards-compatible aliases used by older callers/tests.
MIN_ROWS_REQUIRED = MIN_PERIODS_HARD
MIN_ROWS_RECOMMENDED = MIN_PERIODS_RECOMMENDED
NEG_CORR_WARN = -0.1

_PERIOD_DAYS = {
    DataGranularity.daily: 1,
    DataGranularity.weekly: 7,
    DataGranularity.monthly: 30,
}


def run_audit(spec: MMMSpec, base: str | Path = ".", df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the full data audit and write ``audit.json`` + ``audit_report.md``."""
    if df is None:
        df = load_data(spec.data_path)
    df = parse_dates(df, spec.date_column)

    findings: dict[str, Any] = {
        "audited_at": datetime.now().isoformat(),
        "data_path": str(spec.data_path),
        "framework": spec.framework.value,
        "checks": {},
        "warnings": [],
        "errors": [],
        "summary": {},
    }
    warn = findings["warnings"].append
    err = findings["errors"].append

    date_col, target_col = spec.date_column, spec.target_column
    channel_cols = [c for c in spec.channel_columns() if c in df.columns]
    control_cols = [c for c in spec.control_columns() if c in df.columns]

    _check_contract(spec, df, findings, warn, err)
    n_periods = _check_shape(spec, df, findings, warn, err)
    _check_integrity(spec, df, channel_cols, findings, warn, err)
    _check_identifiability(spec, df, channel_cols, control_cols, n_periods, findings, warn, err)
    _check_signal(spec, df, channel_cols, n_periods, findings, warn, err)
    _check_semantics(spec, df, findings, warn, err)

    findings["summary"] = {
        "total_warnings": len(findings["warnings"]),
        "total_errors": len(findings["errors"]),
        "rows": int(len(df)),
        "periods": int(n_periods),
        "channels": len(channel_cols),
        "controls": len(control_cols),
        "experiments": len(spec.experiments),
        "data_quality_tier": (
            "FAIL" if findings["errors"]
            else "WARN" if len(findings["warnings"]) > 2
            else "PASS"
        ),
    }

    ws = ensure_workspace(base)
    audit_dir = Path(ws) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_dir / "audit.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    with open(audit_dir / "audit_report.md", "w") as f:
        f.write(render_audit_report(findings, spec))
    return findings


# --------------------------------------------------------------------------- #
# 1. Contract
# --------------------------------------------------------------------------- #
def _check_contract(spec, df, findings, warn, err) -> None:
    required = spec.required_columns()
    missing = validate_columns(df, required)
    findings["checks"]["missing_columns"] = missing
    for m in missing:
        err(f"Column '{m}' is named in the spec but absent from the data.")

    non_numeric = []
    for col in spec.channel_columns() + spec.control_columns() + [spec.target_column]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    findings["checks"]["non_numeric_columns"] = non_numeric
    for col in non_numeric:
        err(
            f"'{col}' is not numeric. Currency symbols, thousands separators and 'NULL' "
            "strings are the usual cause; clean the extract rather than coercing here, "
            "because silent coercion turns unparseable rows into zeros."
        )

    unused = [
        c for c in df.columns
        if c not in required and pd.api.types.is_numeric_dtype(df[c])
    ]
    findings["checks"]["unused_numeric_columns"] = unused
    if unused:
        warn(
            f"{len(unused)} numeric columns in the file are not in the spec: "
            f"{unused[:8]}{'...' if len(unused) > 8 else ''}. Confirm none of them is a "
            "driver you meant to include — an omitted driver is attributed to whatever "
            "correlates with it, usually media."
        )


# --------------------------------------------------------------------------- #
# 2. Shape
# --------------------------------------------------------------------------- #
def _check_shape(spec, df, findings, warn, err) -> int:
    date_col = spec.date_column
    if date_col not in df.columns:
        findings["checks"]["dimensions"] = {"rows": len(df)}
        return len(df)

    n_periods = int(df[date_col].nunique())
    findings["checks"]["dimensions"] = {
        "rows": int(len(df)),
        "periods": n_periods,
        "columns": int(len(df.columns)),
        "date_min": df[date_col].min().strftime("%Y-%m-%d"),
        "date_max": df[date_col].max().strftime("%Y-%m-%d"),
    }

    if n_periods < MIN_PERIODS_HARD:
        err(
            f"{n_periods} periods of data. Below {MIN_PERIODS_HARD} you cannot separate "
            "seasonality from media, because you have not seen a full year once. "
            "Collect more history, model at a finer granularity, or use a geo panel to "
            "buy back sample size."
        )
    elif n_periods < MIN_PERIODS_RECOMMENDED:
        warn(
            f"{n_periods} periods. One seasonal cycle lets you fit seasonality but not "
            f"validate it; {MIN_PERIODS_RECOMMENDED}+ is where out-of-sample testing "
            "becomes meaningful."
        )

    # Date regularity, against the declared granularity.
    expected_days = _PERIOD_DAYS[spec.granularity]
    dates = pd.Series(sorted(df[date_col].unique()))
    gaps = dates.diff().dt.days.dropna()
    inferred = float(gaps.mode().iloc[0]) if len(gaps) else None
    big_gaps = gaps[gaps > expected_days * 1.5]
    findings["checks"]["date_regularity"] = {
        "declared_granularity": spec.granularity.value,
        "expected_gap_days": expected_days,
        "modal_gap_days": inferred,
        "n_gaps": int(len(big_gaps)),
        "max_gap_days": int(gaps.max()) if len(gaps) else None,
    }
    if inferred is not None and spec.granularity != DataGranularity.monthly:
        if abs(inferred - expected_days) > 0.5:
            err(
                f"Spec says {spec.granularity.value} data but the typical gap between rows is "
                f"{inferred:.0f} days. Every carryover prior, l_max and half-life in the model "
                "is expressed in periods, so a wrong granularity silently rescales all of them."
            )
    if len(big_gaps) > 0:
        warn(
            f"{len(big_gaps)} gaps longer than one period (largest {int(gaps.max())} days). "
            "Adstock convolution assumes evenly spaced periods; missing weeks make carryover "
            "jump across a hole. Reindex to a complete date range and decide explicitly what "
            "the missing periods mean."
        )

    # Duplicates and panel rectangularity.
    keys = [date_col] + ([spec.geo.geo_column] if spec.geo.is_panel and spec.geo.geo_column else [])
    dupes = int(df.duplicated(subset=keys).sum())
    findings["checks"]["duplicate_rows"] = {"keys": keys, "n_duplicates": dupes}
    if dupes:
        err(
            f"{dupes} duplicate rows on {keys}. Duplicated periods double-count spend and "
            "target; deduplicate (or aggregate) before anything else."
        )

    if spec.geo.is_panel and spec.geo.geo_column and spec.geo.geo_column in df.columns:
        counts = df.groupby(spec.geo.geo_column)[date_col].nunique()
        rectangular = bool(counts.nunique() == 1 and int(counts.iloc[0]) == n_periods)
        findings["checks"]["panel"] = {
            "n_geos": int(counts.size),
            "periods_per_geo": {str(k): int(v) for k, v in counts.items()},
            "rectangular": rectangular,
        }
        if not rectangular:
            err(
                "Panel is not rectangular: geos have different numbers of periods. "
                "Both pymc-marketing and Meridian require every (geo, period) cell to exist. "
                "Fill the missing cells explicitly — with zero spend and a real target, or by "
                "dropping geos with partial history."
            )
        if counts.size < 5:
            warn(
                f"Only {counts.size} geos. Hierarchical pooling needs enough units to estimate "
                "the between-geo variance; with a handful of geos the pooled model is barely "
                "different from a national one."
            )
    return n_periods


# --------------------------------------------------------------------------- #
# 3. Integrity
# --------------------------------------------------------------------------- #
def _check_integrity(spec, df, channel_cols, findings, warn, err) -> None:
    target_col = spec.target_column
    completeness: dict[str, dict] = {}
    for col in [target_col] + channel_cols + spec.control_columns():
        if col not in df.columns:
            continue
        s = df[col]
        pct_missing = float(s.isna().mean())
        entry = {"pct_missing": round(pct_missing * 100, 2)}
        if col in channel_cols:
            entry["pct_zero"] = round(float((s.fillna(0) == 0).mean()) * 100, 2)
        completeness[col] = entry

        if pct_missing > 0:
            severity = err if col == target_col else warn
            severity(
                f"'{col}' is {pct_missing*100:.1f}% missing. For spend, a missing value almost "
                "always means zero spend and should be filled with 0; for a target or a control "
                "it means the period is unusable and filling it with 0 invents data. Decide "
                "per column — never with a blanket fillna."
            )
        if col in channel_cols and entry.get("pct_zero", 0) > SPARSE_ZERO_THRESHOLD * 100:
            warn(
                f"'{col}' has no spend in {entry['pct_zero']:.0f}% of periods. A bursty channel "
                "is informative (on/off contrast identifies the effect) but its saturation "
                "curve is only pinned down over the range it actually ran."
            )
    findings["checks"]["completeness"] = completeness

    negatives = {}
    for col in channel_cols:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            if n_neg:
                negatives[col] = n_neg
    findings["checks"]["negative_spend"] = negatives
    for col, n in negatives.items():
        err(
            f"'{col}' has {n} negative values. Negative spend is usually a credit or rebate "
            "posted against the wrong period. Media transformations assume non-negative input; "
            "reallocate the credit to its original period instead of clipping."
        )

    if target_col in df.columns:
        t = df[target_col].dropna().astype(float)
        skew = float(stats.skew(t)) if len(t) > 2 else 0.0
        z = (t - t.mean()) / t.std() if t.std() > 0 else t * 0
        outlier_idx = list(np.where(np.abs(z) > OUTLIER_Z)[0])
        outlier_dates = (
            [str(d)[:10] for d in df.loc[df.index[outlier_idx], spec.date_column]]
            if spec.date_column in df.columns else []
        )
        findings["checks"]["target_distribution"] = {
            "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "cv": round(float(t.std() / t.mean()), 4) if t.mean() else None,
            "skewness": round(skew, 3),
            "n_zero": int((t == 0).sum()),
            "n_negative": int((t < 0).sum()),
            "outlier_dates": outlier_dates[:20],
        }
        if abs(skew) > SKEWNESS_WARN:
            warn(
                f"Target skewness {skew:.2f}. A Normal likelihood will chase the tail; use "
                "StudentT (heavier tails, robust to spikes) or a log link (multiplicative "
                "structure) rather than deleting the outlying periods."
            )
        if outlier_dates:
            warn(
                f"{len(outlier_dates)} target periods beyond {OUTLIER_Z} sd: "
                f"{outlier_dates[:5]}. Identify each one before modelling — a promotion, a "
                "stockout, a data error and a genuine spike each need a different treatment, "
                "and only the data error should be removed."
            )
        if int((t < 0).sum()):
            err("Target has negative values; a multiplicative or log-link model cannot fit them.")

    if target_col in df.columns and len(df) >= 20:
        br = detect_structural_break(df[target_col])
        findings["checks"]["structural_break"] = br
        if br.get("break_detected"):
            warn(
                f"Structural break detected in the target (p={br.get('p_value', float('nan')):.3f}). "
                "Something changed that is not in the model — a relaunch, a pricing change, a "
                "measurement change, a pandemic. Add an indicator for the regime or split the "
                "series; leaving it unmodelled pushes the shift onto whichever channel moved "
                "at the same time."
            )


# --------------------------------------------------------------------------- #
# 4. Identifiability
# --------------------------------------------------------------------------- #
def _check_identifiability(spec, df, channel_cols, control_cols, n_periods, findings, warn, err) -> None:
    # Flat / constant columns.
    flat = {}
    for col in channel_cols + control_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(float)
        if len(s) == 0:
            continue
        mean = float(s.mean())
        cv = float(s.std() / mean) if mean else 0.0
        if s.nunique() <= 1:
            flat[col] = {"cv": 0.0, "constant": True}
        elif cv < FLAT_CV_THRESHOLD:
            flat[col] = {"cv": round(cv, 4), "constant": False}
    findings["checks"]["flat_variables"] = flat
    for col, info in flat.items():
        if info["constant"]:
            err(
                f"'{col}' is constant. A constant column is perfectly collinear with the "
                "intercept and carries no information at all; drop it."
            )
        else:
            warn(
                f"'{col}' barely varies (CV={info['cv']:.2f}). Its coefficient is determined "
                "almost entirely by the prior. Report it as an assumption, not an estimate — "
                "or run an experiment to create the variation the history lacks."
            )

    # Pairwise correlation among everything the model must tell apart.
    regressors = [c for c in channel_cols + control_cols if c in df.columns]
    if len(regressors) >= 2:
        corr = df[regressors].corr()
        pairs = []
        for i, a in enumerate(regressors):
            for b in regressors[i + 1:]:
                r = float(corr.loc[a, b])
                if abs(r) >= PAIR_CORR_WARN:
                    pairs.append({"a": a, "b": b, "r": round(r, 3)})
        findings["checks"]["high_correlation_pairs"] = pairs
        for p in pairs:
            warn(
                f"{p['a']} and {p['b']} are correlated at r={p['r']}. The model cannot tell "
                "their effects apart from this data — whatever split it reports is the prior's "
                "opinion. Merge them, drop one, or break the correlation with an experiment or "
                "a deliberate flighting change."
            )

    if len(channel_cols) >= 2:
        vif = compute_vif(df, channel_cols)
        findings["checks"]["vif"] = {
            k: (round(v, 2) if not np.isnan(v) else None) for k, v in vif.items()
        }
        for ch, v in vif.items():
            if np.isnan(v):
                continue
            if v > VIF_CRITICAL:
                err(
                    f"VIF {v:.1f} on '{ch}'. Its variation is almost fully explained by the "
                    "other channels, so its coefficient is unstable: small data changes will "
                    "swing the ROAS wildly. Group the channels that move together."
                )
            elif v > VIF_WARN:
                warn(f"VIF {v:.1f} on '{ch}' — moderate collinearity; widen its prior and "
                     "expect a broad credible interval.")

    # Parameter budget. Every parameter has to be paid for with data.
    n_media = len(channel_cols)
    n_params = (
        3 * n_media                                # adstock alpha, saturation lam, beta
        + len(control_cols)
        + 2 * spec.seasonality.yearly_fourier_modes
        + 2                                        # intercept + sigma
    )
    if spec.geo.is_panel and spec.geo.geo_column and spec.geo.geo_column in df.columns:
        n_obs = int(len(df))
    else:
        n_obs = int(n_periods)
    ratio = n_obs / n_params if n_params else float("inf")
    findings["checks"]["parameter_budget"] = {
        "n_observations": n_obs,
        "n_parameters_approx": n_params,
        "obs_per_parameter": round(ratio, 2),
    }
    if ratio < OBS_PER_PARAM_MIN:
        warn(
            f"Roughly {ratio:.1f} observations per parameter ({n_obs} obs, ~{n_params} params). "
            f"Below {OBS_PER_PARAM_MIN:.0f} the model leans on priors for most of its answers. "
            "Cut Fourier modes, merge channels, or add geo dimensions before adding complexity."
        )


# --------------------------------------------------------------------------- #
# 5. Signal
# --------------------------------------------------------------------------- #
def _check_signal(spec, df, channel_cols, n_periods, findings, warn, err) -> None:
    target_col = spec.target_column
    if target_col not in df.columns:
        return
    t = df[target_col].dropna().astype(float)
    cv = float(t.std() / t.mean()) if t.mean() else 0.0
    if cv < TARGET_CV_MIN:
        err(
            f"The target barely moves (CV={cv:.3f}). With no variation to explain there is "
            "nothing for media to be credited with; the model will return priors."
        )

    # Contemporaneous and lagged correlation: an upper-funnel channel that leads
    # the target by weeks looks uncorrelated at lag 0.
    corrs: dict[str, dict] = {}
    for ch in channel_cols:
        if ch not in df.columns:
            continue
        best_lag, best_r = 0, 0.0
        for lag in range(0, min(9, max(n_periods // 8, 1))):
            shifted = df[ch].shift(lag)
            mask = shifted.notna() & df[target_col].notna()
            if mask.sum() < 10:
                continue
            r = float(np.corrcoef(shifted[mask], df[target_col][mask])[0, 1])
            if abs(r) > abs(best_r):
                best_lag, best_r = lag, r
        corrs[ch] = {
            "r_lag0": round(float(df[[target_col, ch]].corr().iloc[0, 1]), 3),
            "best_lag": best_lag,
            "r_best": round(best_r, 3),
        }
    findings["checks"]["target_channel_correlations"] = corrs
    for ch, c in corrs.items():
        if c["r_lag0"] < NEG_CORR_WARN and c["r_best"] < 0:
            warn(
                f"'{ch}' correlates negatively with the target (r={c['r_lag0']} at lag 0, "
                f"best r={c['r_best']} at lag {c['best_lag']}). Three usual causes: the spend "
                "is dated by invoice rather than delivery; the channel is switched on when "
                "sales fall (reverse causality); or it is a defensive channel in a declining "
                "segment. None is fixed by the model."
            )
        elif (
            c["best_lag"] > 0
            and abs(c["r_best"]) >= LAG_CORR_MIN
            and abs(c["r_best"]) > abs(c["r_lag0"]) * 1.5
        ):
            warn(
                f"'{ch}' correlates best with the target {c['best_lag']} periods later "
                f"(r={c['r_best']} vs {c['r_lag0']} at lag 0). Make sure l_max is at least "
                f"{c['best_lag'] + 2} so the adstock can reach that far."
            )

    if n_periods >= MIN_PERIODS_RECOMMENDED:
        ss = seasonal_strength(df[target_col], period=spec.periods_per_year())
        findings["checks"]["seasonality"] = {
            "seasonal_strength": round(ss, 3) if not np.isnan(ss) else None
        }
        if not np.isnan(ss):
            if ss < SEASONALITY_LOW:
                warn(
                    f"Seasonal strength {ss:.3f} — essentially none. Fourier terms will fit "
                    "noise; reduce yearly_fourier_modes to 2-4."
                )
            elif ss > SEASONALITY_HIGH:
                warn(
                    f"Seasonal strength {ss:.3f} — very strong. Seasonality and media compete "
                    "for the same variance when campaigns are flighted seasonally. Use explicit "
                    "holiday/event flags alongside Fourier terms so the model attributes the "
                    "sharp peaks to the calendar rather than to whatever ran during them."
                )
    else:
        findings["checks"]["seasonality"] = {
            "seasonal_strength": None,
            "note": f"Needs >= {MIN_PERIODS_RECOMMENDED} periods for a stable decomposition.",
        }


# --------------------------------------------------------------------------- #
# 6. Semantics
# --------------------------------------------------------------------------- #
def _check_semantics(spec, df, findings, warn, err) -> None:
    # Cost per exposure unit: the cheapest way to catch a broken spend or
    # impressions feed. A CPM that swings 10x is a join problem, not a market.
    cpu: dict[str, dict] = {}
    for ch in spec.active_channels():
        if not (ch.spend_column and ch.exposure_column):
            continue
        if ch.spend_column not in df.columns or ch.exposure_column not in df.columns:
            continue
        spend, exposure = df[ch.spend_column], df[ch.exposure_column]
        mask = (exposure > 0) & spend.notna()
        if mask.sum() < 5:
            continue
        unit_cost = (spend[mask] / exposure[mask]).astype(float)
        lo, hi = float(unit_cost.quantile(0.05)), float(unit_cost.quantile(0.95))
        entry = {
            "median": round(float(unit_cost.median()), 6),
            "p05": round(lo, 6),
            "p95": round(hi, 6),
            "ratio_p95_p05": round(hi / lo, 2) if lo > 0 else None,
            "periods_spend_without_exposure": int(((spend > 0) & (exposure == 0)).sum()),
            "periods_exposure_without_spend": int(((spend == 0) & (exposure > 0)).sum()),
        }
        cpu[ch.column] = entry
        if entry["ratio_p95_p05"] and entry["ratio_p95_p05"] > CPM_DRIFT_RATIO:
            warn(
                f"'{ch.column}': cost per exposure unit varies {entry['ratio_p95_p05']:.0f}x "
                "between the 5th and 95th percentile. Real auction prices do not move that far; "
                "this is usually spend and delivery joined on mismatched dates or a partially "
                "populated feed."
            )
        if entry["periods_spend_without_exposure"]:
            warn(
                f"'{ch.column}': {entry['periods_spend_without_exposure']} periods have spend "
                "but zero exposure. The model will see cost with no delivery and will learn "
                "that the channel does nothing."
            )
        if entry["periods_exposure_without_spend"]:
            warn(
                f"'{ch.column}': {entry['periods_exposure_without_spend']} periods have exposure "
                "with no spend — free delivery, a make-good, or a broken join. If it is genuinely "
                "free, model it as organic; ROAS is undefined at zero cost."
            )
    findings["checks"]["cost_per_exposure_unit"] = cpu

    # Business drivers that are almost always confounders.
    categories = {c.category for c in spec.controls}
    roles = {c.role for c in spec.active_channels()}
    has_price = (
        ControlCategory.pricing in categories
        or any(c.channel_type == "price" for c in spec.active_channels())
    )
    has_distribution = (
        ControlCategory.distribution in categories
        or any(c.channel_type == "distribution" for c in spec.active_channels())
    )
    findings["checks"]["confounder_coverage"] = {
        "price": has_price,
        "distribution": has_distribution,
        "non_media_treatments": ChannelRole.non_media_treatment in roles,
        "control_categories": sorted(c.value for c in categories),
    }
    if not has_price:
        warn(
            "No price or promotion variable. Price moves sales directly and is usually planned "
            "alongside media, so omitting it transfers promotion's effect to whatever channel "
            "was running. This is the single most common cause of overstated retail ROAS."
        )
    if not has_distribution and spec.industry.lower() in {"retail", "cpg", "fmcg", "grocery"}:
        warn(
            "No distribution/availability variable in a retail or CPG model. Store count, ACV "
            "or stock availability moves sales on its own and correlates with media plans."
        )

    if not spec.experiments:
        warn(
            "No experiments recorded. Without at least one incrementality test the model has "
            "no external anchor: several very different attributions fit this data equally "
            "well, and the one it reports is the one the priors preferred."
        )
    else:
        usable = [
            e for e in spec.experiments
            if e.lift_absolute is not None and e.lift_se is not None and e.spend_during_test
        ]
        findings["checks"]["experiments"] = {
            "n": len(spec.experiments),
            "n_usable_as_constraint": len(usable),
            "channels_covered": sorted({e.channel for e in spec.experiments}),
        }
        uncovered = sorted(
            {c.column for c in spec.channels_with_role(ChannelRole.paid_media)}
            - {e.channel for e in spec.experiments}
        )
        if uncovered:
            warn(
                f"Channels with no experiment: {uncovered}. Their effects rest on observational "
                "identification alone; present them with wider ranges than the tested channels."
            )


# --------------------------------------------------------------------------- #
def render_audit_report(findings: dict, spec: MMMSpec) -> str:
    tier = findings["summary"]["data_quality_tier"]
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(tier, "?")
    checks = findings["checks"]
    s = findings["summary"]

    lines = [
        "# MMM Data Audit",
        "",
        f"**Data**: `{findings['data_path']}`  ",
        f"**Target framework**: {findings['framework']}  ",
        f"**Audited**: {findings['audited_at'][:19]}  ",
        f"**Verdict**: {icon} {tier}",
        "",
        "| | |",
        "|---|---|",
        f"| Rows / periods | {s['rows']} / {s['periods']} |",
        f"| Channels | {s['channels']} |",
        f"| Controls | {s['controls']} |",
        f"| Experiments | {s['experiments']} |",
        f"| Blocking issues | {s['total_errors']} |",
        f"| Warnings | {s['total_warnings']} |",
        "",
    ]

    if findings["errors"]:
        lines += ["## Blocking issues", "", "Fix these before fitting anything.", ""]
        lines += [f"{i}. {e}" for i, e in enumerate(findings["errors"], 1)]
        lines.append("")
    if findings["warnings"]:
        lines += ["## Warnings", "", "These shape how the model must be specified and how its "
                  "results should be read.", ""]
        lines += [f"{i}. {w}" for i, w in enumerate(findings["warnings"], 1)]
        lines.append("")

    budget = checks.get("parameter_budget")
    if budget:
        lines += [
            "## Parameter budget",
            "",
            f"{budget['n_observations']} observations for roughly "
            f"{budget['n_parameters_approx']} parameters "
            f"(**{budget['obs_per_parameter']}** obs per parameter).",
            "",
        ]

    vif = checks.get("vif") or {}
    if vif:
        lines += ["## Collinearity (VIF)", "", "| Channel | VIF |", "|---|---|"]
        for ch, v in sorted(vif.items(), key=lambda kv: kv[1] or 0, reverse=True):
            flag = " ❌" if v and v > VIF_CRITICAL else " ⚠️" if v and v > VIF_WARN else ""
            lines.append(f"| {ch} | {v}{flag} |")
        lines.append("")

    pairs = checks.get("high_correlation_pairs") or []
    if pairs:
        lines += ["| Variable A | Variable B | r |", "|---|---|---|"]
        lines += [f"| {p['a']} | {p['b']} | {p['r']} |" for p in pairs]
        lines.append("")

    corrs = checks.get("target_channel_correlations") or {}
    if corrs:
        lines += [
            "## Channel-target relationship",
            "",
            "| Channel | r (lag 0) | Best lag | r at best lag |",
            "|---|---|---|---|",
        ]
        for ch, c in sorted(corrs.items(), key=lambda kv: kv[1]["r_lag0"]):
            lines.append(f"| {ch} | {c['r_lag0']} | {c['best_lag']} | {c['r_best']} |")
        lines.append("")

    cpu = checks.get("cost_per_exposure_unit") or {}
    if cpu:
        lines += [
            "## Cost per exposure unit",
            "",
            "| Channel | Median | p05 | p95 | p95/p05 |",
            "|---|---|---|---|---|",
        ]
        for ch, e in cpu.items():
            lines.append(
                f"| {ch} | {e['median']} | {e['p05']} | {e['p95']} | {e['ratio_p95_p05']} |"
            )
        lines.append("")

    cover = checks.get("confounder_coverage") or {}
    if cover:
        lines += [
            "## Confounder coverage",
            "",
            "| Driver | Present |",
            "|---|---|",
            f"| Price / promotion | {'yes' if cover.get('price') else 'NO'} |",
            f"| Distribution / availability | {'yes' if cover.get('distribution') else 'NO'} |",
            f"| Non-media treatments declared | {'yes' if cover.get('non_media_treatments') else 'no'} |",
            "",
        ]

    lines += ["---", "*Generated by agent-mmm data audit engine*", ""]
    return "\n".join(lines)


_render_report = render_audit_report
