"""Diagnostics engine: convergence, fit metrics, overfit detection, prior-pull, plausibility."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agent_mmm.workspace import ensure_workspace

logger = logging.getLogger(__name__)

# Thresholds (from mmm-diagnostics/SKILL.md)
RHAT_THRESHOLD = 1.05
ESS_THRESHOLD = 400
MAX_DIVERGENCES = 0
OVERFIT_GAP_THRESHOLD = 0.20
PRIOR_PULL_SIGMA_FACTOR = 1.5   # posterior mean within 1.5σ of prior → possible pull
CONTRIBUTION_MIN_PCT = 0.10
CONTRIBUTION_MAX_PCT = 0.90
SINGLE_CHANNEL_MAX_PCT = 0.70


def run_diagnostics(
    run_id: str,
    idata_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    cv_metrics: dict | None = None,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Run all diagnostic checks for a completed run.

    Parameters
    ----------
    run_id: identifier for the run
    idata_path: path to idata.nc (ArviZ InferenceData)
    metrics_path: path to metrics.json from fit_runner
    cv_metrics: optional dict with 'r2_cv' from a CV run
    base: workspace base directory

    Returns dict of findings. Also writes:
    - ./mmm-workspace/runs/<run_id>/diagnostics.json
    - ./mmm-workspace/runs/<run_id>/diagnostics_report.md
    """
    findings: dict[str, Any] = {
        "run_id": run_id,
        "checks": {},
        "warnings": [],
        "errors": [],
        "summary": {},
    }

    def _warn(msg): findings["warnings"].append(msg)
    def _error(msg): findings["errors"].append(msg)

    # Load metrics
    in_sample_r2 = None
    if metrics_path is not None:
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            in_sample_r2 = m.get("r2_insample")
        except Exception as e:
            _warn(f"Could not load metrics.json: {e}")

    # Load InferenceData
    idata = None
    if idata_path is not None:
        try:
            import arviz as az
            idata = az.from_netcdf(str(idata_path))
        except Exception as e:
            _warn(f"Could not load idata.nc: {e}")

    # --- 1. Convergence ---
    convergence = _check_convergence(idata)
    findings["checks"]["convergence"] = convergence
    if convergence.get("max_rhat") and convergence["max_rhat"] > RHAT_THRESHOLD:
        _error(f"High rhat: max={convergence['max_rhat']:.3f} (threshold {RHAT_THRESHOLD}) — model not converged")
    if convergence.get("min_ess_bulk") and convergence["min_ess_bulk"] < ESS_THRESHOLD:
        _warn(f"Low ESS: min={convergence['min_ess_bulk']:.0f} (threshold {ESS_THRESHOLD}) — increase draws")
    if convergence.get("n_divergences") and convergence["n_divergences"] > MAX_DIVERGENCES:
        _error(f"Divergences: {convergence['n_divergences']} — widen priors or increase target_accept")

    # --- 2. In-sample fit ---
    fit_check = {"r2_insample": in_sample_r2}
    if in_sample_r2 is not None:
        if in_sample_r2 < 0.5:
            _warn(f"Low in-sample R²={in_sample_r2:.3f} — model may be underfitting")
        elif in_sample_r2 > 0.95:
            _warn(f"Very high in-sample R²={in_sample_r2:.3f} — check for overfitting")
    findings["checks"]["fit"] = fit_check

    # --- 3. Overfit detection ---
    overfit = _check_overfit(in_sample_r2, cv_metrics)
    findings["checks"]["overfit"] = overfit
    if overfit.get("gap") is not None and overfit["gap"] > OVERFIT_GAP_THRESHOLD:
        _error(f"Overfit detected: gap={overfit['gap']:.3f} > {OVERFIT_GAP_THRESHOLD} — reduce model complexity or widen priors")
    elif overfit.get("gap") is not None:
        pass  # OK

    # --- 4. Prior-pull ---
    prior_pull = _check_prior_pull(idata)
    findings["checks"]["prior_pull"] = prior_pull
    for param, info in prior_pull.items():
        if info.get("possible_pull"):
            _warn(f"Prior pull on {param}: posterior mean {info['posterior_mean']:.3f} is close to prior boundary — prior may be too tight")

    # --- 5. Attribution plausibility ---
    plausibility = _check_attribution_plausibility(idata)
    findings["checks"]["attribution_plausibility"] = plausibility
    if plausibility.get("total_media_pct") is not None:
        pct = plausibility["total_media_pct"]
        if pct < CONTRIBUTION_MIN_PCT * 100:
            _warn(f"Low total media contribution: {pct:.1f}% — marketing channels may have negligible effect")
        elif pct > CONTRIBUTION_MAX_PCT * 100:
            _warn(f"Very high total media contribution: {pct:.1f}% — baseline too small, check controls")
    if plausibility.get("dominant_channel"):
        ch, pct = plausibility["dominant_channel"]
        _error(f"Dominant channel: {ch} explains {pct:.1f}% of media effect — likely collinearity or data issue")

    # Summary
    tier = "PASS" if not findings["errors"] and len(findings["warnings"]) <= 2 else (
        "WARN" if not findings["errors"] else "FAIL"
    )
    findings["summary"] = {
        "tier": tier,
        "n_errors": len(findings["errors"]),
        "n_warnings": len(findings["warnings"]),
        "rhat_ok": convergence.get("max_rhat", 99) < RHAT_THRESHOLD,
        "ess_ok": (convergence.get("min_ess_bulk") or 0) > ESS_THRESHOLD,
        "divergences_ok": (convergence.get("n_divergences") or 0) == 0,
        "overfit_ok": (overfit.get("gap") or 0) <= OVERFIT_GAP_THRESHOLD,
    }

    # Save
    ws = ensure_workspace(base)
    run_dir = ws / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    report = _render_diagnostics_report(findings)
    with open(run_dir / "diagnostics_report.md", "w") as f:
        f.write(report)

    return findings


def _check_convergence(idata) -> dict:
    """Compute rhat, ESS_bulk, and divergence count from InferenceData."""
    if idata is None:
        return {"available": False}
    try:
        import arviz as az
        summary = az.summary(idata, var_names=None)
        max_rhat = float(summary["r_hat"].max()) if "r_hat" in summary.columns else None
        min_ess = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else None
        n_div = 0
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            n_div = int(idata.sample_stats["diverging"].values.sum())
        return {
            "available": True,
            "max_rhat": round(max_rhat, 4) if max_rhat else None,
            "min_ess_bulk": round(min_ess, 1) if min_ess else None,
            "n_divergences": n_div,
            "rhat_ok": max_rhat is not None and max_rhat < RHAT_THRESHOLD,
            "ess_ok": min_ess is not None and min_ess > ESS_THRESHOLD,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _check_overfit(in_sample_r2: float | None, cv_metrics: dict | None) -> dict:
    """Compute overfit gap = in_sample_r2 - cv_r2."""
    if in_sample_r2 is None:
        return {"available": False, "gap": None}
    if cv_metrics is None or cv_metrics.get("r2_cv") is None:
        return {"available": False, "in_sample_r2": in_sample_r2, "gap": None,
                "note": "Run CV to compute overfit gap"}
    cv_r2 = float(cv_metrics["r2_cv"])
    gap = round(in_sample_r2 - cv_r2, 4)
    return {
        "available": True,
        "in_sample_r2": round(in_sample_r2, 4),
        "cv_r2": round(cv_r2, 4),
        "gap": gap,
        "overfit": gap > OVERFIT_GAP_THRESHOLD,
    }


def _check_prior_pull(idata) -> dict[str, dict]:
    """Detect parameters whose posterior mean is suspiciously close to prior boundary.

    Heuristic: for adstock_alpha (Beta), if posterior mean < 0.05 or > 0.95 → possible pull.
    For saturation_lam (Gamma), if posterior mean < 0.3 → possible pull to near-zero.
    """
    if idata is None:
        return {}
    results = {}
    try:
        posterior = idata.posterior
        for var in posterior.data_vars:
            vals = posterior[var].values.flatten()
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            possible_pull = False
            if "alpha" in var.lower() and 0 < mean_val < 1:
                # Beta: pull if mean very near bounds
                possible_pull = mean_val < 0.03 or mean_val > 0.97
            elif "lam" in var.lower() and mean_val > 0:
                # Gamma: pull if mean near zero
                possible_pull = mean_val < 0.2
            results[var] = {
                "posterior_mean": round(mean_val, 4),
                "posterior_std": round(std_val, 4),
                "possible_pull": possible_pull,
            }
    except Exception as e:
        results["_error"] = {"error": str(e), "possible_pull": False}
    return results


def _check_attribution_plausibility(idata) -> dict:
    """Check if channel contributions are plausible."""
    if idata is None:
        return {"available": False}
    try:
        posterior = idata.posterior
        # Look for contribution variables (added by add_original_scale_contribution_variable)
        contrib_vars = [v for v in posterior.data_vars if "contribution" in v.lower() and "channel" in v.lower()]
        if not contrib_vars:
            return {"available": False, "note": "No contribution variables found in posterior — run attribution extraction first"}

        channel_contribs = {}
        for var in contrib_vars:
            arr = posterior[var].values
            channel_contribs[var] = float(np.mean(arr))

        total = sum(channel_contribs.values())
        if total == 0:
            return {"available": True, "total_media_pct": 0}

        # Dominant channel check
        dominant = None
        for ch, val in channel_contribs.items():
            pct = val / total * 100
            if pct > SINGLE_CHANNEL_MAX_PCT * 100:
                dominant = (ch, pct)

        return {
            "available": True,
            "channel_contributions": {k: round(v / total * 100, 1) for k, v in channel_contribs.items()},
            "total_media_pct": round(total, 1),
            "dominant_channel": dominant,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _render_diagnostics_report(findings: dict) -> str:
    tier = findings["summary"].get("tier", "UNKNOWN")
    tier_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(tier, "?")
    lines = [
        f"# MMM Diagnostics Report",
        f"",
        f"**Run ID**: `{findings['run_id']}`  ",
        f"**Overall**: {tier_emoji} {tier}  ",
        f"",
        "---",
        "",
        "## Convergence",
        "",
    ]
    conv = findings["checks"].get("convergence", {})
    if conv.get("available"):
        rhat_icon = "✅" if conv.get("rhat_ok") else "❌"
        ess_icon = "✅" if conv.get("ess_ok") else "⚠️"
        div_icon = "✅" if conv.get("n_divergences", 1) == 0 else "❌"
        lines += [
            f"| Check | Value | Status |",
            f"|-------|-------|--------|",
            f"| Max rhat | {conv.get('max_rhat', 'N/A')} | {rhat_icon} (< {RHAT_THRESHOLD}) |",
            f"| Min ESS_bulk | {conv.get('min_ess_bulk', 'N/A')} | {ess_icon} (> {ESS_THRESHOLD}) |",
            f"| Divergences | {conv.get('n_divergences', 'N/A')} | {div_icon} (= 0) |",
            "",
        ]
    else:
        lines += [f"*Convergence data not available ({conv.get('error', 'no idata')})*", ""]

    lines += ["## Overfit Detection", ""]
    ov = findings["checks"].get("overfit", {})
    if ov.get("available"):
        gap = ov["gap"]
        icon = "✅" if gap <= OVERFIT_GAP_THRESHOLD else "❌"
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| In-sample R² | {ov['in_sample_r2']} |",
            f"| CV R² | {ov['cv_r2']} |",
            f"| Gap (overfit) | **{gap:.3f}** {icon} (threshold {OVERFIT_GAP_THRESHOLD}) |",
            "",
        ]
    else:
        lines += [f"*{ov.get('note', 'CV not run yet')}*", ""]

    if findings["errors"]:
        lines += ["## ❌ Errors", ""]
        for e in findings["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    if findings["warnings"]:
        lines += ["## ⚠️ Warnings", ""]
        for w in findings["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines += ["---", "*Generated by agent-mmm diagnostics engine*", ""]
    return "\n".join(lines)
