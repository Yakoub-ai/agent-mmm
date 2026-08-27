"""Diagnostics: does this model deserve to be believed?

The checks are ordered the way a reviewer should read them, because an answer to
a later question is meaningless if an earlier one failed:

1. **Convergence** — did the sampler actually explore the posterior? A model
   with r-hat 1.3 has no results to interpret, however good the R² looks.
2. **Fit** — does it reproduce the data it was trained on?
3. **Generalisation** — does it reproduce data it has not seen?
4. **Baseline health** — is the decomposition sane before we look at channels?
   A negative or runaway baseline means every channel number below it is wrong.
5. **Learning** — did the data move the priors, or is the posterior just the
   prior wearing a hat?
6. **Plausibility** — do the channel effects survive contact with what we know
   about the business?

Written for ArviZ >= 1.2 and pymc-marketing >= 1.0, where inference data is an
``xarray.DataTree`` and ``az.waic`` no longer exists (use ``az.loo``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agent_mmm.workspace import ensure_workspace

logger = logging.getLogger(__name__)

# Convergence thresholds. r-hat 1.01 is the modern rank-normalised threshold
# (Vehtari et al. 2021) and is what PyMC itself warns at; 1.05 is the older,
# looser bar and is treated here as an outright failure rather than a target.
RHAT_TARGET = 1.01
RHAT_FAIL = 1.05
RHAT_THRESHOLD = RHAT_FAIL  # backwards-compatible alias
ESS_PER_CHAIN_TARGET = 100
ESS_THRESHOLD = 400
MAX_DIVERGENCES = 0
BFMI_THRESHOLD = 0.2

OVERFIT_GAP_THRESHOLD = 0.20
CONTRACTION_MIN = 0.20
"""Posterior contraction below this means the data said almost nothing about a
parameter: 1 - sd(posterior)/sd(prior)."""

# Baseline plausibility. Wide on purpose — the aim is to catch decompositions
# that are structurally broken, not to enforce a house view.
BASELINE_MIN_SHARE = 0.30
BASELINE_MAX_SHARE = 0.95
MEDIA_SHARE_WARN_HIGH = 0.60
SINGLE_CHANNEL_MAX_SHARE = 0.70


# --------------------------------------------------------------------------- #
# idata access helpers (DataTree in ArviZ 1.x, InferenceData in 0.x)
# --------------------------------------------------------------------------- #
def _group(idata: Any, name: str):
    """Return a group from either a DataTree or a legacy InferenceData."""
    if idata is None:
        return None
    try:
        if hasattr(idata, "children"):
            return idata[name].to_dataset() if name in idata.children else None
    except Exception:
        pass
    try:
        return getattr(idata, name, None)
    except Exception:
        return None


def _has_group(idata: Any, name: str) -> bool:
    return _group(idata, name) is not None


def load_idata(path: str | Path) -> Any:
    """Load inference data saved by ``MMM.save`` or ``DataTree.to_netcdf``."""
    import arviz as az

    return az.from_netcdf(str(path))


def _sample_dims(da) -> list[str]:
    return [d for d in ("chain", "draw", "sample") if d in da.dims]


# --------------------------------------------------------------------------- #
def run_diagnostics(
    run_id: str,
    idata_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    cv_metrics: dict | None = None,
    idata: Any = None,
    spend_totals: dict[str, float] | None = None,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Run every diagnostic check for one run.

    Writes ``diagnostics.json`` and ``diagnostics_report.md`` under
    ``mmm-workspace/runs/<run_id>/`` and returns the findings.
    """
    findings: dict[str, Any] = {
        "run_id": run_id,
        "checks": {},
        "warnings": [],
        "errors": [],
        "summary": {},
    }

    def _warn(msg: str) -> None:
        findings["warnings"].append(msg)

    def _error(msg: str) -> None:
        findings["errors"].append(msg)

    in_sample_r2 = None
    if metrics_path is not None:
        try:
            with open(metrics_path) as f:
                in_sample_r2 = json.load(f).get("r2_insample")
        except Exception as e:
            _warn(f"Could not load metrics.json: {e}")

    if idata is None and idata_path is not None:
        try:
            idata = load_idata(idata_path)
        except Exception as e:
            _warn(f"Could not load inference data: {e}")

    # --- 1. Convergence ---------------------------------------------------- #
    convergence = check_convergence(idata)
    findings["checks"]["convergence"] = convergence
    max_rhat = convergence.get("max_rhat")
    if max_rhat is not None:
        if max_rhat > RHAT_FAIL:
            _error(
                f"r-hat {max_rhat:.3f} exceeds {RHAT_FAIL} on {convergence.get('worst_rhat_param')}. "
                "The chains disagree about where the posterior is; nothing downstream is "
                "interpretable. Raise tune, then look for an identifiability problem."
            )
        elif max_rhat > RHAT_TARGET:
            _warn(
                f"r-hat {max_rhat:.3f} above the {RHAT_TARGET} target on "
                f"{convergence.get('worst_rhat_param')}. Usable for exploration, not for a "
                "decision that moves budget."
            )
    if convergence.get("min_ess_bulk") is not None and convergence["min_ess_bulk"] < ESS_THRESHOLD:
        _warn(
            f"Lowest bulk ESS is {convergence['min_ess_bulk']:.0f} on "
            f"{convergence.get('worst_ess_param')} (< {ESS_THRESHOLD}). Credible intervals on "
            "that parameter are noisy — increase draws before quoting them."
        )
    if (convergence.get("n_divergences") or 0) > MAX_DIVERGENCES:
        _error(
            f"{convergence['n_divergences']} divergent transitions. The sampler could not "
            "traverse part of the posterior, so the samples are biased, not merely noisy. "
            "Raise target_accept to 0.95-0.99; if divergences persist, the geometry is the "
            "problem (usually a funnel from a near-unidentified channel)."
        )
    if convergence.get("bfmi") is not None and convergence["bfmi"] < BFMI_THRESHOLD:
        _warn(
            f"BFMI {convergence['bfmi']:.2f} < {BFMI_THRESHOLD}: the sampler is exploring the "
            "energy distribution poorly, which usually points at a heavy-tailed or badly "
            "scaled parameter."
        )

    # --- 2. Fit ------------------------------------------------------------- #
    fit_check = {"r2_insample": in_sample_r2}
    if in_sample_r2 is not None:
        if in_sample_r2 < 0.5:
            _warn(
                f"In-sample R² {in_sample_r2:.3f}. Something systematic is missing — a control, "
                "a structural break, or the wrong target."
            )
        elif in_sample_r2 > 0.98:
            _warn(
                f"In-sample R² {in_sample_r2:.3f} is suspiciously high. Check for a control that "
                "is a proxy for the target (last year's sales, a lagged target, a variable "
                "derived from the target). A near-perfect MMM is almost always leaking."
            )
    findings["checks"]["fit"] = fit_check

    # --- 3. Generalisation --------------------------------------------------- #
    overfit = check_overfit(in_sample_r2, cv_metrics)
    findings["checks"]["overfit"] = overfit
    if overfit.get("gap") is not None and overfit["gap"] > OVERFIT_GAP_THRESHOLD:
        _error(
            f"Train/validation R² gap {overfit['gap']:.3f} > {OVERFIT_GAP_THRESHOLD}. The model "
            "has memorised the history. Tighten priors, cut Fourier modes, or merge collinear "
            "channels — do not ship the ROAS numbers."
        )
    elif overfit.get("gap") is None:
        _warn(
            "No cross-validation run, so generalisation is unknown. An MMM that has never been "
            "asked to predict unseen weeks has not been validated."
        )

    # --- 4. Baseline --------------------------------------------------------- #
    # One decomposition, shared by the baseline and plausibility checks, so the
    # two can never report contradictory shares of the same target.
    decomposition = decompose(idata)
    findings["checks"]["decomposition"] = {
        k: v for k, v in decomposition.items() if k != "series"
    }
    baseline = check_baseline(idata, decomposition=decomposition)
    findings["checks"]["baseline"] = baseline
    if baseline.get("available"):
        share = baseline.get("baseline_share")
        if baseline.get("negative_periods", 0) > 0:
            _error(
                f"The baseline is negative in {baseline['negative_periods']} periods "
                f"({baseline['negative_pct']:.1f}% of the series). A negative baseline means the "
                "model claims the business would have sold less than nothing without marketing. "
                "It is a symptom, not a result: media is absorbing a trend or a control that is "
                "missing from the model."
            )
        if share is not None and share < BASELINE_MIN_SHARE:
            _warn(
                f"Baseline is only {share:.1%} of the target. Media rarely drives more than "
                "half of an established brand's sales; a small baseline usually means an "
                "omitted driver (distribution, price, seasonality) is being credited to media."
            )
        if share is not None and share > BASELINE_MAX_SHARE:
            _warn(
                f"Baseline absorbs {share:.1%} of the target, leaving almost nothing for media. "
                "Check whether a time-varying intercept or a high Fourier order is soaking up "
                "the variation media should be explaining."
            )
        if baseline.get("trend_drift_pct") is not None and abs(baseline["trend_drift_pct"]) > 50:
            _warn(
                f"Baseline drifts {baseline['trend_drift_pct']:+.0f}% from start to end of the "
                "series. That much unexplained drift is a missing structural variable; it also "
                "makes any forward-looking budget recommendation fragile."
            )

    # --- 5. Learning --------------------------------------------------------- #
    contraction = check_prior_contraction(idata)
    findings["checks"]["prior_contraction"] = contraction
    weak = [k for k, v in contraction.items() if isinstance(v, dict) and v.get("prior_dominated")]
    if weak:
        _warn(
            "Prior-dominated parameters (posterior barely narrower than prior): "
            + ", ".join(weak[:8])
            + ". The data did not identify these; their posteriors restate your assumptions. "
            "Say so in the report rather than presenting them as findings."
        )

    # --- 6. Plausibility ------------------------------------------------------ #
    plausibility = check_attribution_plausibility(
        idata, spend_totals, decomposition=decomposition
    )
    findings["checks"]["attribution_plausibility"] = plausibility
    if plausibility.get("available"):
        media_share = plausibility.get("media_share")
        if media_share is not None and media_share > MEDIA_SHARE_WARN_HIGH:
            _warn(
                f"Media explains {media_share:.1%} of the target. Above ~60% is implausible for "
                "an established brand and usually means the baseline is being starved."
            )
        if plausibility.get("dominant_channel"):
            ch, pct = plausibility["dominant_channel"]
            _error(
                f"{ch} takes {pct:.1f}% of all media contribution. One channel absorbing the "
                "media effect is the signature of collinearity: check the VIF and the spend "
                "correlation matrix before reporting this."
            )
        for ch, r in (plausibility.get("negative_channels") or {}).items():
            _warn(
                f"{ch} has a negative mean contribution ({r:.3f}). Either the spend data is "
                "misaligned in time, or the channel is collinear with something that has the "
                "opposite sign. Media effects should not be negative."
            )

    tier = (
        "FAIL" if findings["errors"]
        else "WARN" if len(findings["warnings"]) > 2
        else "PASS"
    )
    findings["summary"] = {
        "tier": tier,
        "n_errors": len(findings["errors"]),
        "n_warnings": len(findings["warnings"]),
        "rhat_ok": max_rhat is not None and max_rhat <= RHAT_FAIL,
        "ess_ok": (convergence.get("min_ess_bulk") or 0) >= ESS_THRESHOLD,
        "divergences_ok": (convergence.get("n_divergences") or 0) == 0,
        "overfit_ok": (overfit.get("gap") or 0) <= OVERFIT_GAP_THRESHOLD,
        "baseline_ok": bool(baseline.get("available")) and baseline.get("negative_periods", 0) == 0,
    }

    ws = ensure_workspace(base)
    run_dir = Path(ws) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    with open(run_dir / "diagnostics_report.md", "w") as f:
        f.write(render_diagnostics_report(findings))

    return findings


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _target_scale(idata: Any) -> float:
    """Scale factor that maps normalised model output back to target units.

    pymc-marketing max-abs scales the target before fitting, so every
    contribution in the posterior is a fraction of that scale. Comparing a
    normalised contribution against real spend gives a ROAS off by orders of
    magnitude, which is exactly the kind of error that survives review because
    the number still *looks* like a number.
    """
    const = _group(idata, "constant_data")
    if const is None or "target_scale" not in const:
        return 1.0
    try:
        return float(np.asarray(const["target_scale"]).ravel()[0])
    except Exception:
        return 1.0


def decompose(idata: Any, original_scale: bool = True) -> dict[str, Any]:
    """Split the fitted target into baseline, seasonality, controls and media.

    Returns totals over the whole series in a single consistent denominator, so
    the shares reported by different checks cannot contradict each other. This
    is the one place the decomposition is computed; every share downstream is
    derived from it.
    """
    posterior = _group(idata, "posterior")
    if posterior is None:
        return {"available": False}

    buckets = {
        "intercept": ("intercept_contribution",),
        "trend": ("trend_contribution", "linear_trend"),
        "seasonality": ("fourier_contribution", "yearly_seasonality_contribution",
                        "monthly_seasonality_contribution", "weekly_seasonality_contribution"),
        "controls": ("control_contribution",),
        "media": ("channel_contribution",),
    }
    scale = _target_scale(idata) if original_scale else 1.0

    n_dates = int(posterior.sizes.get("date", 0)) or None

    totals: dict[str, float] = {}
    series: dict[str, np.ndarray] = {}

    for bucket, keys in buckets.items():
        names = [v for v in posterior.data_vars if str(v) in keys]
        if not names:
            continue
        acc = None
        for name in names:
            da = posterior[name]
            m = da.mean(dim=_sample_dims(da))
            extra = [d for d in m.dims if d != "date"]
            m = m.sum(dim=extra) if extra else m
            values = np.atleast_1d(np.asarray(m)).ravel().astype(float)
            # A time-constant component (a plain intercept) has no date dim, but
            # it applies in every period. Broadcasting it here is what makes the
            # shares below add up to the fitted target instead of understating
            # the baseline by a factor of n_periods.
            if n_dates and values.size == 1 and n_dates > 1:
                values = np.repeat(values, n_dates)
            acc = values if acc is None else acc + values
        acc = acc * scale
        series[bucket] = acc
        totals[bucket] = float(acc.sum())

    grand = sum(totals.values())
    shares = {k: (v / grand if grand else None) for k, v in totals.items()}
    return {
        "available": bool(totals),
        "original_scale": original_scale,
        "target_scale": scale,
        "totals": {k: round(v, 4) for k, v in totals.items()},
        "shares": {k: (round(v, 4) if v is not None else None) for k, v in shares.items()},
        "series": series,
        "grand_total": round(grand, 4),
    }


def check_convergence(idata: Any) -> dict:
    """r-hat, ESS, divergences and BFMI, with the worst offender named."""
    if idata is None:
        return {"available": False}
    try:
        import arviz as az

        summary = az.summary(idata)
        out: dict[str, Any] = {"available": True}

        if "r_hat" in summary.columns:
            rhat = summary["r_hat"].dropna()
            if len(rhat):
                out["max_rhat"] = round(float(rhat.max()), 4)
                out["worst_rhat_param"] = str(rhat.idxmax())
        for col, key in (("ess_bulk", "min_ess_bulk"), ("ess_tail", "min_ess_tail")):
            if col in summary.columns:
                ess = summary[col].dropna()
                if len(ess):
                    out[key] = round(float(ess.min()), 1)
                    if col == "ess_bulk":
                        out["worst_ess_param"] = str(ess.idxmin())

        stats = _group(idata, "sample_stats")
        if stats is not None:
            if "diverging" in stats:
                out["n_divergences"] = int(np.asarray(stats["diverging"]).sum())
            if "energy" in stats:
                try:
                    out["bfmi"] = round(float(np.nanmin(np.atleast_1d(az.bfmi(idata)))), 3)
                except Exception:
                    pass
            if "chain" in stats.sizes:
                out["n_chains"] = int(stats.sizes["chain"])
                if out.get("min_ess_bulk") is not None:
                    out["ess_per_chain"] = round(out["min_ess_bulk"] / out["n_chains"], 1)

        out["rhat_ok"] = out.get("max_rhat", 99) <= RHAT_FAIL
        out["ess_ok"] = (out.get("min_ess_bulk") or 0) >= ESS_THRESHOLD
        return out
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_overfit(in_sample_r2: float | None, cv_metrics: dict | None) -> dict:
    """Train minus validation R². The gap, not the level, is what matters."""
    if in_sample_r2 is None:
        return {"available": False, "gap": None}
    cv_r2 = (cv_metrics or {}).get("r2_cv")
    if cv_r2 is None:
        return {
            "available": False,
            "in_sample_r2": round(in_sample_r2, 4),
            "gap": None,
            "note": "Run time-slice cross-validation to measure generalisation.",
        }
    gap = round(in_sample_r2 - float(cv_r2), 4)
    return {
        "available": True,
        "in_sample_r2": round(in_sample_r2, 4),
        "cv_r2": round(float(cv_r2), 4),
        "gap": gap,
        "overfit": gap > OVERFIT_GAP_THRESHOLD,
    }


def check_baseline(idata: Any, decomposition: dict | None = None) -> dict:
    """Is the non-media part of the decomposition sane?

    The baseline is what the business would have done without media. It is the
    first thing to check and the last thing most reports mention. Three failure
    modes matter: it goes negative (impossible), it is tiny (media is being
    credited with the whole business), or it drifts hugely (a structural driver
    is missing and the trend term is papering over it).
    """
    dec = decomposition if decomposition is not None else decompose(idata)
    if not dec.get("available"):
        return {"available": False, "note": dec.get("note", "No decomposable posterior.")}
    if "intercept" not in dec["series"]:
        return {"available": False, "note": "No intercept_contribution in the posterior."}

    try:
        # Baseline = everything that is not media. Seasonality and controls belong
        # here: they are business-as-usual variation, not marketing's doing.
        components = [k for k in dec["series"] if k != "media"]
        values = np.zeros_like(next(iter(dec["series"].values())), dtype=float)
        for k in components:
            values = values + dec["series"][k]

        n_negative = int((values < 0).sum())
        n = max(len(values), 1)
        drift = None
        if len(values) >= 8:
            window = max(len(values) // 8, 1)
            head = float(np.mean(values[:window]))
            tail = float(np.mean(values[-window:]))
            if abs(head) > 1e-9:
                drift = round((tail - head) / abs(head) * 100, 1)

        grand = dec["grand_total"]
        return {
            "available": True,
            "components": components,
            "baseline_total": round(float(values.sum()), 4),
            "baseline_share": round(float(values.sum()) / grand, 4) if grand else None,
            "component_shares": {k: dec["shares"].get(k) for k in components},
            "negative_periods": n_negative,
            "negative_pct": round(n_negative / n * 100, 1),
            "min_baseline": round(float(values.min()), 4),
            "trend_drift_pct": drift,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_prior_contraction(idata: Any) -> dict[str, dict]:
    """How much did the data narrow each prior?

    Contraction = 1 - sd(posterior)/sd(prior). Near 1 means the data determined
    the parameter; near 0 means the posterior is the prior restated. This is the
    honest version of "prior pull" — it needs the prior group, so run
    ``sample_prior_predictive`` before fitting.
    """
    posterior = _group(idata, "posterior")
    prior = _group(idata, "prior")
    if posterior is None:
        return {}
    if prior is None:
        return {"_note": {"error": "No prior group; call sample_prior_predictive before fitting."}}

    results: dict[str, dict] = {}
    for var in posterior.data_vars:
        name = str(var)
        if "contribution" in name or name.endswith("_sigma"):
            continue
        if var not in prior.data_vars:
            continue
        try:
            post = posterior[var]
            pre = prior[var]
            post_sd = float(post.std(dim=_sample_dims(post)).mean())
            prior_sd = float(pre.std(dim=_sample_dims(pre)).mean())
            if prior_sd <= 0:
                continue
            contraction = 1.0 - post_sd / prior_sd
            results[name] = {
                "prior_sd": round(prior_sd, 5),
                "posterior_sd": round(post_sd, 5),
                "contraction": round(contraction, 4),
                "prior_dominated": contraction < CONTRACTION_MIN,
            }
        except Exception:
            continue
    return results


def check_attribution_plausibility(
    idata: Any,
    spend_totals: dict[str, float] | None = None,
    decomposition: dict | None = None,
) -> dict:
    """Channel shares, negative effects, and (if spend is known) implied ROAS.

    ``spend_totals`` must be in the target's own currency/units; contributions
    are converted out of the model's normalised space first, so the two sides of
    the ratio agree.
    """
    posterior = _group(idata, "posterior")
    if posterior is None or "channel_contribution" not in posterior:
        return {"available": False, "note": "No channel_contribution in the posterior."}
    try:
        dec = decomposition if decomposition is not None else decompose(idata)
        scale = dec.get("target_scale", 1.0) if dec.get("available") else _target_scale(idata)

        cc = posterior["channel_contribution"]
        mean = cc.mean(dim=_sample_dims(cc))
        reduce_dims = [d for d in mean.dims if d != "channel"]
        per_channel = mean.sum(dim=reduce_dims) if reduce_dims else mean
        channels = [str(c) for c in np.atleast_1d(per_channel["channel"].values)]
        values = np.atleast_1d(np.asarray(per_channel)).astype(float) * scale

        media_total = float(values.sum())
        shares = {
            ch: (round(float(v) / media_total * 100, 1) if media_total else None)
            for ch, v in zip(channels, values)
        }
        media_share = dec["shares"].get("media") if dec.get("available") else None

        # Concentration is only evidence of a problem when there are enough
        # channels for it to be surprising. With two channels one of them is
        # bound to exceed 70%, and flagging that is noise, not a finding.
        dominant = None
        if len(channels) >= 3:
            for ch, pct in shares.items():
                if pct is not None and pct > SINGLE_CHANNEL_MAX_SHARE * 100:
                    dominant = (ch, pct)

        negative = {ch: round(float(v), 4) for ch, v in zip(channels, values) if v < 0}

        roas = None
        if spend_totals:
            roas = {}
            for ch, v in zip(channels, values):
                spend = spend_totals.get(ch)
                if spend:
                    roas[ch] = round(float(v) / spend, 3)

        return {
            "available": True,
            "channel_contribution_total": round(media_total, 4),
            "channel_shares_pct": shares,
            "media_share": media_share,
            "dominant_channel": dominant,
            "negative_channels": negative,
            "implied_roas": roas,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def compare_models(idatas: dict[str, Any]) -> Any:
    """LOO comparison across candidate models.

    ``az.waic`` was removed in ArviZ 1.x; LOO-PSIS is strictly better anyway
    because it reports Pareto-k, which tells you when the approximation itself
    is untrustworthy. For MMM, LOO on a time series is optimistic — it leaves
    out single points from an autocorrelated series — so treat it as a tiebreak
    between similar models, never as a substitute for time-slice CV.
    """
    import arviz as az

    return az.compare(idatas)


# --------------------------------------------------------------------------- #
def render_diagnostics_report(findings: dict) -> str:
    tier = findings["summary"].get("tier", "UNKNOWN")
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(tier, "?")
    checks = findings["checks"]
    lines = [
        "# MMM Diagnostics Report",
        "",
        f"**Run**: `{findings['run_id']}`  ",
        f"**Verdict**: {icon} {tier}",
        "",
        "---",
        "",
        "## 1. Convergence",
        "",
    ]
    conv = checks.get("convergence", {})
    if conv.get("available"):
        lines += [
            "| Check | Value | Target |",
            "|---|---|---|",
            f"| Max r-hat | {conv.get('max_rhat', 'n/a')} ({conv.get('worst_rhat_param', '-')}) | <= {RHAT_TARGET} |",
            f"| Min ESS bulk | {conv.get('min_ess_bulk', 'n/a')} ({conv.get('worst_ess_param', '-')}) | >= {ESS_THRESHOLD} |",
            f"| Min ESS tail | {conv.get('min_ess_tail', 'n/a')} | >= {ESS_THRESHOLD} |",
            f"| Divergences | {conv.get('n_divergences', 'n/a')} | 0 |",
            f"| BFMI | {conv.get('bfmi', 'n/a')} | >= {BFMI_THRESHOLD} |",
            "",
        ]
    else:
        lines += [f"*Not available ({conv.get('error', 'no inference data')}).*", ""]

    lines += ["## 2. Fit and generalisation", ""]
    ov = checks.get("overfit", {})
    fit = checks.get("fit", {})
    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| In-sample R2 | {fit.get('r2_insample', 'n/a')} |",
        f"| CV R2 | {ov.get('cv_r2', 'not run')} |",
        f"| Gap | {ov.get('gap', 'n/a')} (threshold {OVERFIT_GAP_THRESHOLD}) |",
        "",
    ]

    base = checks.get("baseline", {})
    lines += ["## 3. Baseline health", ""]
    if base.get("available"):
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Baseline share of target | {base.get('baseline_share')} |",
            f"| Periods with negative baseline | {base.get('negative_periods')} ({base.get('negative_pct')}%) |",
            f"| Baseline drift start-to-end | {base.get('trend_drift_pct')}% |",
            f"| Components counted | {', '.join(base.get('components', []))} |",
            "",
        ]
    else:
        lines += [f"*Not available ({base.get('note') or base.get('error', 'no posterior')}).*", ""]

    contraction = checks.get("prior_contraction", {})
    lines += ["## 4. Did the data teach us anything?", ""]
    real = {k: v for k, v in contraction.items() if isinstance(v, dict) and "contraction" in v}
    if real:
        lines += ["| Parameter | Prior sd | Posterior sd | Contraction |", "|---|---|---|---|"]
        for k, v in sorted(real.items(), key=lambda kv: kv[1]["contraction"]):
            flag = " (prior-dominated)" if v["prior_dominated"] else ""
            lines.append(f"| {k} | {v['prior_sd']} | {v['posterior_sd']} | {v['contraction']:.2f}{flag} |")
        lines.append("")
    else:
        lines += ["*No prior group — run `sample_prior_predictive` before fitting to enable this.*", ""]

    pl = checks.get("attribution_plausibility", {})
    lines += ["## 5. Attribution plausibility", ""]
    if pl.get("available"):
        lines += [
            f"Media share of modelled target: **{pl.get('media_share')}**",
            "",
            "| Channel | Share of media effect |",
            "|---|---|",
        ]
        for ch, pct in (pl.get("channel_shares_pct") or {}).items():
            lines.append(f"| {ch} | {pct}% |")
        if pl.get("implied_roas"):
            lines += ["", "| Channel | Implied ROAS |", "|---|---|"]
            for ch, r in pl["implied_roas"].items():
                lines.append(f"| {ch} | {r} |")
        lines.append("")
    else:
        lines += [f"*Not available ({pl.get('note') or pl.get('error', '')}).*", ""]

    if findings["errors"]:
        lines += ["## Blocking issues", ""] + [f"- {e}" for e in findings["errors"]] + [""]
    if findings["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in findings["warnings"]] + [""]

    lines += ["---", "*Generated by agent-mmm diagnostics engine*", ""]
    return "\n".join(lines)


# Backwards-compatible private aliases.
_check_convergence = check_convergence
_check_overfit = check_overfit
_check_prior_pull = check_prior_contraction
_check_attribution_plausibility = check_attribution_plausibility
_render_diagnostics_report = render_diagnostics_report
