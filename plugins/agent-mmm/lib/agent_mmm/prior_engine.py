"""Prior recommendation engine.

Turns a spec plus the data's own characteristics into priors that are defensible
line by line. Three ideas drive the design:

1. **Priors are stated in interpretable units first.** Every carryover prior
   starts life as a half-life in weeks and every media-effect prior as an ROI
   range. The distribution parameters are derived, never hand-picked.
2. **Confidence tracks identifiability.** A channel with flat, always-on spend
   or a mostly-zero history cannot be estimated precisely from observational
   data. The engine widens those priors rather than letting a tight prior
   masquerade as a finding.
3. **Experiments outrank catalogs.** Any channel with a measured lift result
   gets flagged for calibration; the catalog prior is only a fallback for
   channels nobody has measured.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent_mmm.spec import ChannelMeta, ChannelRole, MMMSpec
from agent_mmm.utils.channel_classifier import (
    alpha_to_halflife,
    classify_channel,
    get_channel_type,
    halflife_to_alpha,
    suggested_l_max,
)
from agent_mmm.utils.io import load_data, parse_dates
from agent_mmm.utils.moment_match import (
    beta_moment_match,
    gamma_moment_match,
    lognormal_from_range,
    lognormal_quantile,
)
from agent_mmm.workspace import ensure_workspace

_CATALOG_PATH = Path(__file__).parent.parent.parent / "references" / "prior_catalog.yaml"

# Spend variation below this coefficient of variation means the channel is
# effectively a constant and cannot be separated from the intercept.
FLAT_SPEND_CV = 0.15
SPARSE_ZERO_FRACTION = 0.30


def _load_prior_catalog() -> dict:
    if not _CATALOG_PATH.exists():
        return {}
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f) or {}


def _get_channel_prior(channel_type: str, catalog: dict) -> dict:
    priors = catalog.get("channel_priors", {})
    fallback = catalog.get("defaults") or {
        "halflife_weeks": 1.5, "alpha_mu": 0.63, "alpha_sigma": 0.15,
        "lam_mu": 2.5, "lam_sigma": 0.8,
        "roi_prior_low": 0.3, "roi_prior_high": 6.0,
        "rationale": "default fallback",
    }
    return dict(priors.get(channel_type) or fallback)


def compute_spend_shares(df: pd.DataFrame, channel_cols: list[str]) -> dict[str, float]:
    """Each channel's share of total spend."""
    totals = {c: float(df[c].sum()) for c in channel_cols if c in df.columns}
    grand_total = sum(totals.values())
    if grand_total <= 0:
        n = max(len(channel_cols), 1)
        return {c: 1.0 / n for c in channel_cols}
    return {c: v / grand_total for c, v in totals.items()}


def _rescale_halflife(halflife_weeks: float, periods_per_year: int) -> float:
    """Catalog half-lives are in weeks; convert to the spec's period length."""
    return halflife_weeks * (periods_per_year / 52.0)


def _channel_stats(series: pd.Series) -> dict[str, float]:
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return {"pct_zero": 1.0, "cv": 0.0, "mean": 0.0, "n_active": 0}
    mean = float(vals.mean())
    return {
        "pct_zero": float((vals == 0).mean()),
        "cv": float(vals.std() / mean) if mean > 0 else 0.0,
        "mean": mean,
        "n_active": int((vals > 0).sum()),
    }


def recommend_priors(
    spec: MMMSpec,
    audit_findings: dict | None = None,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Generate prior recommendations for every channel in the spec.

    Returns a dict with:
      * ``model_config`` — JSON-serialisable pymc-marketing model_config
      * ``roi_priors``   — LogNormal ROI priors per paid channel (Meridian-ready)
      * ``structure``    — per-channel adstock/saturation/l_max recommendations
      * ``per_channel_audit`` — the reasoning behind each prior
      * ``calibration_targets`` — channels that have an experiment and should be
        calibrated rather than left to the catalog
    """
    catalog = _load_prior_catalog()
    sparse_mult = catalog.get("sparse_channel_sigma_multiplier", 1.5)
    short_mult = catalog.get("short_history_sigma_multiplier", 1.3)
    always_on_mult = catalog.get("always_on_sigma_multiplier", 1.4)
    min_obs = catalog.get("min_obs_for_tight_priors", 104)

    df = load_data(spec.data_path)
    df = parse_dates(df, spec.date_column)
    n_periods = df[spec.date_column].nunique()
    ppy = spec.periods_per_year()

    warnings: list[str] = []
    per_channel: list[dict] = []
    structure: dict[str, dict] = {}
    roi_priors: dict[str, dict] = {}

    media_channels = spec.channels_with_role(
        ChannelRole.paid_media, ChannelRole.paid_reach_frequency, ChannelRole.organic_media
    )
    media_cols = [c.model_input_column for c in media_channels]
    spend_map = spec.spend_columns()
    spend_shares = compute_spend_shares(df, [v for v in spend_map.values() if v in df.columns])

    experiments_by_channel: dict[str, list] = {}
    for exp in spec.experiments:
        experiments_by_channel.setdefault(exp.channel, []).append(exp)

    alpha_a, alpha_b, lam_a, lam_b, beta_sigma = [], [], [], [], []

    if n_periods < min_obs:
        warnings.append(
            f"Only {n_periods} periods of data (< {min_obs}); all channel priors widened "
            f"by {short_mult}x. Treat every ROAS from this model as directional."
        )

    for ch in media_channels:
        col = ch.model_input_column
        ch_type = ch.channel_type or classify_channel(col)
        taxonomy = get_channel_type(ch_type)
        cat = _get_channel_prior(ch_type, catalog)

        # ---- carryover, stated as a half-life ---------------------------- #
        halflife_weeks = ch.halflife_periods or cat.get(
            "halflife_weeks", taxonomy.typical_halflife_weeks
        )
        halflife = _rescale_halflife(halflife_weeks, ppy) if ch.halflife_periods is None else ch.halflife_periods
        alpha_mu = halflife_to_alpha(halflife) if halflife > 0 else cat["alpha_mu"]
        alpha_mu = float(np.clip(alpha_mu, 0.01, 0.95))
        alpha_sigma = float(cat.get("alpha_sigma", 0.15))
        lam_mu = float(cat.get("lam_mu", 2.5))
        lam_sigma = float(cat.get("lam_sigma", 0.8))

        # ---- widen where the data cannot support confidence -------------- #
        stats = _channel_stats(df[col]) if col in df.columns else _channel_stats(pd.Series(dtype=float))
        reasons: list[str] = []

        if stats["pct_zero"] > SPARSE_ZERO_FRACTION:
            alpha_sigma *= sparse_mult
            lam_sigma *= sparse_mult
            reasons.append(f"sparse ({stats['pct_zero']*100:.0f}% zero periods)")
        if n_periods < min_obs:
            alpha_sigma *= short_mult
            lam_sigma *= short_mult
        is_flat = stats["cv"] < FLAT_SPEND_CV and stats["n_active"] > 0
        is_always_on = bool(ch.always_on) or (ch.always_on is None and taxonomy.usually_always_on)
        if is_flat or is_always_on:
            alpha_sigma *= always_on_mult
            lam_sigma *= always_on_mult
            if is_flat:
                reasons.append(f"flat spend (CV={stats['cv']:.2f})")
                warnings.append(
                    f"{col}: spend barely varies (CV={stats['cv']:.2f}). Its effect is not "
                    "identified from observational data — the posterior will mostly reproduce "
                    "the prior. Only an experiment or a deliberate spend change can resolve it."
                )
            else:
                reasons.append("always-on channel")
                warnings.append(
                    f"{col}: treated as always-on, so its variation carries limited information "
                    "about what would happen at zero spend. The saturation curve near the origin "
                    "is an extrapolation; do not quote marginal ROAS far below observed spend."
                )

        alpha_sigma = float(np.clip(alpha_sigma, 0.02, 0.30))
        lam_sigma = float(max(lam_sigma, 0.2))

        # ---- structure ---------------------------------------------------- #
        l_max = ch.l_max or suggested_l_max(halflife)
        adstock_kind = ch.adstock or cat.get("default_adstock") or taxonomy.default_adstock
        saturation_kind = ch.saturation or spec.architecture.default_saturation
        structure[col] = {
            "channel_type": ch_type,
            "role": ch.role.value,
            "adstock": adstock_kind,
            "saturation": saturation_kind,
            "l_max": l_max,
            "halflife_periods": round(halflife, 2),
            "funnel_stage": (ch.funnel_stage.value if ch.funnel_stage else taxonomy.funnel_stage),
        }

        # ---- moment matching --------------------------------------------- #
        try:
            a_alpha, b_alpha = beta_moment_match(alpha_mu, alpha_sigma)
        except ValueError as e:
            warnings.append(f"{col}: beta_moment_match failed ({e}); using Beta(2, 6)")
            a_alpha, b_alpha = 2.0, 6.0
        try:
            a_lam, b_lam = gamma_moment_match(lam_mu, lam_sigma)
        except ValueError as e:
            warnings.append(f"{col}: gamma_moment_match failed ({e}); using Gamma(4, 1)")
            a_lam, b_lam = 4.0, 1.0

        spend_col = ch.spend_column
        share = spend_shares.get(spend_col, 1.0 / max(len(media_cols), 1)) if spend_col else 0.0
        # saturation_beta sigma proportional to spend share: a channel that buys 2%
        # of the media cannot plausibly move the target as much as one buying 40%.
        beta_sig = float(max(share, 0.02))

        alpha_a.append(round(a_alpha, 4))
        alpha_b.append(round(b_alpha, 4))
        lam_a.append(round(a_lam, 4))
        lam_b.append(round(b_lam, 4))
        beta_sigma.append(round(beta_sig, 4))

        # ---- ROI prior (for ROI-parameterised frameworks) ----------------- #
        exps = experiments_by_channel.get(ch.column, []) + experiments_by_channel.get(ch.label, [])
        roi_entry: dict[str, Any] | None = None
        if ch.role in (ChannelRole.paid_media, ChannelRole.paid_reach_frequency):
            low = float(cat.get("roi_prior_low", 0.3))
            high = float(cat.get("roi_prior_high", 6.0))
            source = "catalog"
            measured = next((e for e in exps if e.roi_point is not None), None)
            if measured is not None:
                if measured.roi_ci_low and measured.roi_ci_high:
                    low, high = float(measured.roi_ci_low), float(measured.roi_ci_high)
                else:
                    low, high = measured.roi_point * 0.6, measured.roi_point * 1.6
                source = f"experiment ({measured.design.value})"
            mu_ln, sigma_ln = lognormal_from_range(max(low, 1e-3), max(high, low * 1.01), mass=0.90)
            roi_entry = {
                "distribution": "LogNormal",
                "mu": round(mu_ln, 4),
                "sigma": round(sigma_ln, 4),
                "implied_p05": round(lognormal_quantile(mu_ln, sigma_ln, 0.05), 3),
                "implied_median": round(lognormal_quantile(mu_ln, sigma_ln, 0.50), 3),
                "implied_p95": round(lognormal_quantile(mu_ln, sigma_ln, 0.95), 3),
                "source": source,
            }
            roi_priors[col] = roi_entry

        per_channel.append({
            "column": col,
            "spend_column": spend_col,
            "channel_type": ch_type,
            "role": ch.role.value,
            "halflife_periods": round(halflife, 2),
            "alpha_mu": round(alpha_mu, 4),
            "alpha_sigma": round(alpha_sigma, 4),
            "lam_mu": round(lam_mu, 4),
            "lam_sigma": round(lam_sigma, 4),
            "alpha_beta_params": [round(a_alpha, 4), round(b_alpha, 4)],
            "lam_gamma_params": [round(a_lam, 4), round(b_lam, 4)],
            "beta_sigma": round(beta_sig, 4),
            "spend_share_pct": round(share * 100, 1),
            "l_max": l_max,
            "adstock": adstock_kind,
            "saturation": saturation_kind,
            "pct_zero": round(stats["pct_zero"] * 100, 1),
            "spend_cv": round(stats["cv"], 3),
            "widened_because": reasons,
            "roi_prior": roi_entry,
            "has_experiment": bool(exps),
            "rationale": cat.get("rationale", taxonomy.notes),
        })

    model_config = _build_model_config(
        spec, media_cols, alpha_a, alpha_b, lam_a, lam_b, beta_sigma, warnings
    )

    calibration_targets = [
        {
            "channel": exp.channel,
            "design": exp.design.value,
            "has_absolute_lift": exp.lift_absolute is not None,
            "has_roi": exp.roi_point is not None,
            "ready_for_lift_test_api": exp.lift_absolute is not None
            and exp.lift_se is not None
            and exp.spend_during_test is not None,
        }
        for exp in spec.experiments
    ]
    for t in calibration_targets:
        if t["has_absolute_lift"] and not t["ready_for_lift_test_api"]:
            warnings.append(
                f"Experiment for {t['channel']} has a lift value but is missing "
                "lift_se and/or spend_during_test, so it cannot be used as a "
                "likelihood constraint. Collect both or downgrade it to an ROI prior."
            )
    if not spec.experiments:
        warnings.append(
            "No experiments in the spec. Every channel's effect rests on observational "
            "identification alone; ROAS point estimates should be presented as ranges "
            "and paired with a plan to run at least one geo test."
        )

    result = {
        "generated_at": datetime.now().isoformat(),
        "framework": spec.framework.value,
        "granularity": spec.granularity.value,
        "n_periods": int(n_periods),
        "n_channels": len(media_cols),
        "model_config": model_config,
        "roi_priors": roi_priors,
        "structure": structure,
        "per_channel_audit": per_channel,
        "calibration_targets": calibration_targets,
        "warnings": warnings,
        "prior_predictive_guidance": _prior_predictive_guidance(),
    }

    ws = ensure_workspace(base)
    priors_dir = Path(ws) / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    with open(priors_dir / "model_config.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(priors_dir / "prior_audit_report.md", "w") as f:
        f.write(_render_priors_report(result, spec))

    return result


def _build_model_config(
    spec: MMMSpec,
    media_cols: list[str],
    alpha_a: list[float],
    alpha_b: list[float],
    lam_a: list[float],
    lam_b: list[float],
    beta_sigma: list[float],
    warnings: list[str],
) -> dict:
    """Assemble the pymc-marketing ``model_config`` dict (JSON-serialisable)."""
    arch = spec.architecture

    # The intercept prior is the baseline prior. Media and baseline compete for
    # the same variance, so an intercept prior centred on 1.0 (the whole target)
    # quietly tells the model that media explains nothing.
    expected_media = arch.expected_media_contribution
    intercept_mu = 0.5 if expected_media is None else float(np.clip(1.0 - expected_media, 0.05, 0.95))
    if expected_media is not None:
        warnings.append(
            f"Intercept prior centred at {intercept_mu:.2f} of scaled target, matching the "
            f"stated expectation that media drives {expected_media:.0%}. Verify this against "
            "the prior predictive decomposition before fitting."
        )

    if arch.likelihood == "StudentT":
        likelihood = {
            "distribution": "StudentT",
            "nu": arch.student_t_nu,
            "sigma": {"distribution": "HalfNormal", "sigma": 0.5},
        }
    elif arch.likelihood == "LogNormal":
        likelihood = {"distribution": "LogNormal", "sigma": {"distribution": "HalfNormal", "sigma": 0.5}}
    else:
        likelihood = {"distribution": "Normal", "sigma": {"distribution": "HalfNormal", "sigma": 0.5}}

    cfg: dict[str, Any] = {
        "adstock_alpha": {
            "distribution": "Beta", "alpha": alpha_a, "beta": alpha_b, "dims": "channel",
        },
        "saturation_lam": {
            "distribution": "Gamma", "alpha": lam_a, "beta": lam_b, "dims": "channel",
        },
        "saturation_beta": {
            "distribution": "HalfNormal", "sigma": beta_sigma, "dims": "channel",
        },
        "intercept": {"distribution": "Normal", "mu": intercept_mu, "sigma": 0.3},
        "likelihood": likelihood,
        "_metadata": {
            "n_channels": len(media_cols),
            "channel_order": media_cols,
            "link": arch.link.value,
            "generated_at": datetime.now().isoformat(),
        },
    }
    if spec.control_columns():
        cfg["gamma_control"] = {"distribution": "Normal", "mu": 0.0, "sigma": 0.5, "dims": "control"}
    if spec.seasonality.yearly_fourier_modes:
        cfg["gamma_fourier"] = {"distribution": "Laplace", "mu": 0.0, "b": 0.3, "dims": "fourier_mode"}
    return cfg


def tighten_priors_from_posterior(
    idata_path: str | Path,
    tighten_factor: float = 0.7,
    params: tuple[str, ...] = ("adstock_alpha", "saturation_lam", "saturation_beta"),
) -> dict[str, Any]:
    """Re-centre priors on a previous posterior — for brownfield refreshes only.

    This is legitimate when the new fit covers *new data* (a quarterly refresh)
    and illegitimate when it re-fits the *same data*, which double-counts the
    evidence and shrinks intervals that should not shrink. The caller is
    responsible for knowing which situation it is in; the returned payload
    records the factor so the DS report can disclose it.
    """
    import arviz as az

    idata = az.from_netcdf(str(idata_path))
    posterior = idata["posterior"] if hasattr(idata, "children") else idata.posterior

    out: dict[str, Any] = {"tighten_factor": tighten_factor, "params": {}}
    for name in params:
        if name not in posterior:
            continue
        arr = posterior[name]
        sample_dims = [d for d in ("chain", "draw") if d in arr.dims]
        mean = arr.mean(dim=sample_dims)
        std = arr.std(dim=sample_dims) * tighten_factor
        out["params"][name] = {
            "mu": np.atleast_1d(mean.values).round(4).tolist(),
            "sigma": np.atleast_1d(std.values).round(4).tolist(),
            "halflife_periods": (
                [round(alpha_to_halflife(float(v)), 2) for v in np.atleast_1d(mean.values)]
                if name == "adstock_alpha"
                else None
            ),
        }
    return out


def _prior_predictive_guidance() -> str:
    return (
        "Before fitting, draw from the prior predictive and check three things:\n"
        "  1. Coverage — the 90% prior predictive band contains the observed target range. "
        "A band 10x too wide means the sampler will waste its time; too narrow means the "
        "prior will override the data.\n"
        "  2. Decomposition — the implied media share of the target is plausible "
        "(typically 5-40% for an established brand). If the priors imply media drives 90% "
        "of sales, the intercept prior is wrong, not the channel priors.\n"
        "  3. Sign and shape — sampled response curves bend the way you expect and no "
        "channel's prior contribution exceeds total observed target.\n"
        "Code: prior = mmm.sample_prior_predictive(X=X, y=y, samples=500)"
    )


def _render_priors_report(result: dict, spec: MMMSpec) -> str:
    lines = [
        "# MMM Prior Recommendations",
        "",
        f"**Company**: {spec.company_name} | **Industry**: {spec.industry} | **Region**: {spec.region}  ",
        f"**Framework**: {result['framework']} | **Granularity**: {result['granularity']} "
        f"| **Periods**: {result['n_periods']}  ",
        f"**Generated**: {result['generated_at'][:19]}",
        "",
        "---",
        "",
    ]

    if result["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in result["warnings"]]
        lines.append("")

    lines += [
        "## Per-Channel Priors",
        "",
        "Half-life is the interpretable form of the carryover prior: after this many "
        "periods, half the effect of a burst has decayed.",
        "",
        "| Channel | Type | Role | Half-life | alpha mu | alpha sd | lam mu | lam sd | Spend % | l_max | Widened |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ch in result["per_channel_audit"]:
        widened = "; ".join(ch["widened_because"]) or "-"
        lines.append(
            f"| `{ch['column']}` | {ch['channel_type']} | {ch['role']} "
            f"| {ch['halflife_periods']} | {ch['alpha_mu']:.3f} | {ch['alpha_sigma']:.3f} "
            f"| {ch['lam_mu']:.2f} | {ch['lam_sigma']:.2f} | {ch['spend_share_pct']:.1f}% "
            f"| {ch['l_max']} | {widened} |"
        )

    if result["roi_priors"]:
        lines += [
            "",
            "## ROI Priors",
            "",
            "Stated as a 90% range so a marketer can challenge them. Used directly by "
            "ROI-parameterised frameworks (Meridian `media_prior_type=\"roi\"`); for "
            "pymc-marketing they are a sanity check on the implied posterior ROAS.",
            "",
            "| Channel | 5th pct | Median | 95th pct | Source |",
            "|---|---|---|---|---|",
        ]
        for col, r in result["roi_priors"].items():
            lines.append(
                f"| `{col}` | {r['implied_p05']} | {r['implied_median']} | {r['implied_p95']} | {r['source']} |"
            )

    lines += ["", "## Structure", "", "| Channel | Adstock | Saturation | l_max | Funnel |", "|---|---|---|---|---|"]
    for col, s in result["structure"].items():
        lines.append(f"| `{col}` | {s['adstock']} | {s['saturation']} | {s['l_max']} | {s['funnel_stage']} |")

    if result["calibration_targets"]:
        lines += ["", "## Calibration Targets", "", "| Channel | Design | Usable as likelihood constraint |", "|---|---|---|"]
        for t in result["calibration_targets"]:
            lines.append(f"| {t['channel']} | {t['design']} | {'yes' if t['ready_for_lift_test_api'] else 'no'} |")

    lines += [
        "",
        "## Prior Predictive Check",
        "",
        "```",
        result["prior_predictive_guidance"],
        "```",
        "",
        "---",
        "*Generated by agent-mmm prior recommendation engine*",
        "",
    ]
    return "\n".join(lines)
