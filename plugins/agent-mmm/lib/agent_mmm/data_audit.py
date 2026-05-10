"""Automated data quality audit for MMM datasets."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from agent_mmm.spec import MMMSpec
from agent_mmm.utils.io import load_data, parse_dates, validate_columns
from agent_mmm.utils.vif import compute_vif
from agent_mmm.utils.seasonality import seasonal_strength
from agent_mmm.utils.breaks import detect_structural_break
from agent_mmm.workspace import ensure_workspace


# Thresholds (configurable via kwargs in run_audit)
MIN_ROWS_REQUIRED = 52
MIN_ROWS_RECOMMENDED = 104
SPARSE_ZERO_THRESHOLD = 0.30   # >30% zero-spend weeks
VIF_WARN = 5.0
VIF_CRITICAL = 10.0
SKEWNESS_WARN = 2.0
NEG_CORR_WARN = -0.1
SEASONALITY_LOW = 0.05
SEASONALITY_HIGH = 0.50


def run_audit(spec: MMMSpec, base: str | Path = ".") -> dict[str, Any]:
    """Run the full data audit. Returns a dict of findings."""
    df = load_data(spec.data_path)
    df = parse_dates(df, spec.date_column)

    ws = ensure_workspace(base)
    channel_cols = spec.channel_columns()
    control_cols = spec.control_columns()
    target_col = spec.target_column

    findings: dict[str, Any] = {
        "audited_at": datetime.now().isoformat(),
        "data_path": str(spec.data_path),
        "checks": {},
        "warnings": [],
        "errors": [],
        "summary": {},
    }

    def _warn(msg: str) -> None:
        findings["warnings"].append(msg)

    def _error(msg: str) -> None:
        findings["errors"].append(msg)

    # --- 1. Missing required columns ---
    all_required = [spec.date_column, target_col] + channel_cols
    missing = validate_columns(df, all_required)
    findings["checks"]["missing_columns"] = missing
    for m in missing:
        _error(f"Required column missing: {m}")

    present_channels = [c for c in channel_cols if c in df.columns]

    # --- 2. Dimensions ---
    n_rows = len(df)
    date_min = df[spec.date_column].min().strftime("%Y-%m-%d") if spec.date_column in df.columns else None
    date_max = df[spec.date_column].max().strftime("%Y-%m-%d") if spec.date_column in df.columns else None
    findings["checks"]["dimensions"] = {
        "rows": n_rows, "columns": len(df.columns),
        "date_min": date_min, "date_max": date_max,
    }
    if n_rows < MIN_ROWS_REQUIRED:
        _error(f"Insufficient data: {n_rows} rows (need >= {MIN_ROWS_REQUIRED} weeks)")
    elif n_rows < MIN_ROWS_RECOMMENDED:
        _warn(f"Marginal data: {n_rows} rows (recommend >= {MIN_ROWS_RECOMMENDED} for reliable CV)")

    # --- 3. Date continuity ---
    if spec.date_column in df.columns:
        dates = df[spec.date_column].sort_values()
        gaps = dates.diff().dt.days.dropna()
        expected_gap = 7
        large_gaps = gaps[gaps > expected_gap * 2]
        findings["checks"]["date_continuity"] = {
            "expected_gap_days": expected_gap,
            "max_gap_days": int(gaps.max()) if len(gaps) else None,
            "n_large_gaps": int(len(large_gaps)),
        }
        if len(large_gaps) > 0:
            _warn(f"Date gaps detected: {len(large_gaps)} gaps > 14 days — check for missing weeks")

    # --- 4. Completeness ---
    completeness = {}
    for col in [target_col] + present_channels:
        if col not in df.columns:
            continue
        pct_missing = float(df[col].isna().mean())
        pct_zero = float((df[col] == 0).mean()) if col in present_channels else None
        completeness[col] = {
            "pct_missing": round(pct_missing * 100, 2),
            "pct_zero": round(pct_zero * 100, 2) if pct_zero is not None else None,
        }
        if pct_missing > 0.01:
            _warn(f"{col}: {pct_missing*100:.1f}% missing values")
        if col in present_channels and pct_zero is not None and pct_zero > SPARSE_ZERO_THRESHOLD:
            _warn(f"{col}: {pct_zero*100:.1f}% zero-spend weeks (sparse channel — widen priors)")
    findings["checks"]["completeness"] = completeness

    # --- 5. Target distribution ---
    if target_col in df.columns:
        t = df[target_col].dropna()
        skew = float(stats.skew(t))
        findings["checks"]["target_distribution"] = {
            "min": float(t.min()), "max": float(t.max()),
            "mean": float(t.mean()), "std": float(t.std()),
            "skewness": round(skew, 3),
        }
        if abs(skew) > SKEWNESS_WARN:
            _warn(f"Target skewness={skew:.2f} (>2): consider log transform or StudentT(nu=3) likelihood")

    # --- 6. VIF ---
    if len(present_channels) >= 2:
        vif_results = compute_vif(df, present_channels)
        findings["checks"]["vif"] = {k: round(v, 2) if not np.isnan(v) else None for k, v in vif_results.items()}
        for ch, v in vif_results.items():
            if np.isnan(v):
                continue
            if v > VIF_CRITICAL:
                _error(f"VIF critical: {ch} VIF={v:.1f} (>10) — high multicollinearity, consider grouping channels")
            elif v > VIF_WARN:
                _warn(f"VIF elevated: {ch} VIF={v:.1f} (>5) — moderate multicollinearity")

    # --- 7. Target-channel correlations ---
    if target_col in df.columns:
        corr_results = {}
        for ch in present_channels:
            if ch in df.columns:
                r = float(df[[target_col, ch]].corr().iloc[0, 1])
                corr_results[ch] = round(r, 3)
                if r < NEG_CORR_WARN:
                    _warn(f"Negative correlation: {ch} ρ={r:.3f} with target — verify data, check lag effects")
        findings["checks"]["target_channel_correlations"] = corr_results

    # --- 8. Structural break ---
    if target_col in df.columns:
        break_result = detect_structural_break(df[target_col])
        findings["checks"]["structural_break"] = break_result
        if break_result.get("break_detected"):
            _warn(f"Structural break detected in target (p={break_result.get('p_value', '?'):.3f}) — add a pre/post dummy control")

    # --- 9. Seasonality ---
    if target_col in df.columns and len(df) >= 104:
        ss = seasonal_strength(df[target_col], period=52)
        findings["checks"]["seasonality"] = {"seasonal_strength": round(ss, 3) if not np.isnan(ss) else None}
        if not np.isnan(ss):
            if ss < SEASONALITY_LOW:
                _warn(f"Very low seasonality (strength={ss:.3f}) — Fourier modes may not help; consider fewer modes")
            elif ss > SEASONALITY_HIGH:
                _warn(f"Very strong seasonality (strength={ss:.3f}) — consider yearly_fourier_modes >= 12 or explicit holiday controls")
    elif target_col in df.columns:
        findings["checks"]["seasonality"] = {"seasonal_strength": None, "note": "Need >= 104 rows for STL"}

    # --- 10. Multi-geo check ---
    if spec.geo.is_panel and spec.geo.geo_column and spec.geo.geo_column in df.columns:
        geo_counts = df.groupby(spec.geo.geo_column).size().to_dict()
        findings["checks"]["multi_geo"] = {"rows_per_geo": {str(k): int(v) for k, v in geo_counts.items()}}
        if len(set(geo_counts.values())) > 1:
            _warn("Unequal row counts across geos — check for missing date-geo combinations")

    # --- 11. Channel spend trends ---
    if spec.date_column in df.columns and len(df) >= 4:
        midpoint = df[spec.date_column].median()
        trend_results = {}
        for ch in present_channels:
            if ch not in df.columns:
                continue
            first_half = df.loc[df[spec.date_column] < midpoint, ch].mean()
            second_half = df.loc[df[spec.date_column] >= midpoint, ch].mean()
            if first_half > 0:
                change_pct = (second_half - first_half) / first_half * 100
                trend_results[ch] = {"first_half_mean": round(float(first_half), 2),
                                     "second_half_mean": round(float(second_half), 2),
                                     "change_pct": round(float(change_pct), 1)}
                if change_pct < -50:
                    _warn(f"{ch}: spend dropped {abs(change_pct):.0f}% in second half — possible channel wind-down")
                elif change_pct > 200:
                    _warn(f"{ch}: spend increased {change_pct:.0f}% in second half — possible structural change")
        findings["checks"]["channel_spend_trends"] = trend_results

    # Build summary
    findings["summary"] = {
        "total_warnings": len(findings["warnings"]),
        "total_errors": len(findings["errors"]),
        "rows": n_rows,
        "channels": len(present_channels),
        "data_quality_tier": (
            "PASS" if not findings["errors"] and len(findings["warnings"]) <= 2
            else "WARN" if not findings["errors"]
            else "FAIL"
        ),
    }

    # Save artifacts
    audit_dir = Path(ws) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_dir / "audit.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    report_md = _render_report(findings, spec)
    with open(audit_dir / "audit_report.md", "w") as f:
        f.write(report_md)

    return findings


def _render_report(findings: dict, spec: MMMSpec) -> str:
    """Render the audit findings as a human-readable Markdown report."""
    tier = findings["summary"]["data_quality_tier"]
    tier_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(tier, "?")
    lines = [
        f"# MMM Data Audit Report",
        f"",
        f"**Data**: `{findings['data_path']}`  ",
        f"**Audited**: {findings['audited_at'][:19]}  ",
        f"**Quality Tier**: {tier_emoji} {tier}  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rows | {findings['summary']['rows']} |",
        f"| Channels | {findings['summary']['channels']} |",
        f"| Errors | {findings['summary']['total_errors']} |",
        f"| Warnings | {findings['summary']['total_warnings']} |",
        f"",
    ]

    if findings["errors"]:
        lines += ["## ❌ Errors (must fix before modeling)", ""]
        for e in findings["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    if findings["warnings"]:
        lines += ["## ⚠️ Warnings", ""]
        for w in findings["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    # Dimension block
    dims = findings["checks"].get("dimensions", {})
    if dims:
        lines += [
            "## Dataset Dimensions",
            f"",
            f"- **Rows**: {dims.get('rows')}",
            f"- **Date range**: {dims.get('date_min')} → {dims.get('date_max')}",
            f"",
        ]

    # VIF block
    vif = findings["checks"].get("vif", {})
    if vif:
        lines += ["## Channel Collinearity (VIF)", "", "| Channel | VIF |", "|---------|-----|"]
        for ch, v in sorted(vif.items(), key=lambda x: x[1] or 0, reverse=True):
            flag = " ❌" if v and v > 10 else " ⚠️" if v and v > 5 else ""
            lines.append(f"| {ch} | {v}{flag} |")
        lines.append("")

    # Correlations block
    corr = findings["checks"].get("target_channel_correlations", {})
    if corr:
        lines += ["## Target Correlations", "", "| Channel | Pearson ρ |", "|---------|-----------|"]
        for ch, r in sorted(corr.items(), key=lambda x: x[1]):
            flag = " ⚠️" if r < NEG_CORR_WARN else ""
            lines.append(f"| {ch} | {r:.3f}{flag} |")
        lines.append("")

    # Seasonality
    seas = findings["checks"].get("seasonality", {})
    if seas and seas.get("seasonal_strength") is not None:
        ss = seas["seasonal_strength"]
        interp = "None/very weak" if ss < 0.05 else "Weak" if ss < 0.15 else "Moderate" if ss < 0.35 else "Strong" if ss < 0.50 else "Very strong"
        lines += [f"## Seasonality", f"", f"Seasonal strength: **{ss:.3f}** ({interp})", ""]

    lines += ["---", "", "*Generated by agent-mmm data audit engine*", ""]
    return "\n".join(lines)
