"""Prior recommendation engine for MMM channel parameters."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agent_mmm.spec import MMMSpec
from agent_mmm.utils.moment_match import beta_moment_match, gamma_moment_match
from agent_mmm.utils.channel_classifier import classify_channel
from agent_mmm.utils.io import load_data, parse_dates
from agent_mmm.workspace import ensure_workspace

_CATALOG_PATH = Path(__file__).parent.parent.parent / "references" / "prior_catalog.yaml"


def _load_prior_catalog() -> dict:
    if not _CATALOG_PATH.exists():
        return {}
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f) or {}


def _get_channel_prior(channel_type: str, catalog: dict) -> dict:
    """Retrieve prior params for a channel type. Falls back to 'digital_display'."""
    priors = catalog.get("channel_priors", {})
    return priors.get(channel_type) or priors.get("digital_display") or {
        "alpha_mu": 0.25, "alpha_sigma": 0.12,
        "lam_mu": 3.0, "lam_sigma": 0.8,
        "rationale": "default fallback",
    }


def compute_spend_shares(df: pd.DataFrame, channel_cols: list[str]) -> dict[str, float]:
    """Compute each channel's share of total spend."""
    totals = {c: float(df[c].sum()) for c in channel_cols if c in df.columns}
    grand_total = sum(totals.values())
    if grand_total == 0:
        return {c: 1.0 / len(channel_cols) for c in channel_cols}
    return {c: v / grand_total for c, v in totals.items()}


def recommend_priors(
    spec: MMMSpec,
    audit_findings: dict | None = None,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Generate prior parameter recommendations for all channels in spec.

    Returns a dict containing:
    - model_config: ready-to-use dict for MMM model_config (JSON-serializable)
    - per_channel_audit: detailed prior audit per channel
    - warnings: list of warning strings
    """
    catalog = _load_prior_catalog()
    sparse_multiplier = catalog.get("sparse_channel_sigma_multiplier", 1.5)

    df = load_data(spec.data_path)
    df = parse_dates(df, spec.date_column)
    channel_cols = spec.channel_columns()
    n_rows = len(df)

    warnings: list[str] = []
    per_channel: list[dict] = []

    # Spend shares for saturation_beta scaling
    spend_shares = compute_spend_shares(df, channel_cols)

    # Per-channel prior computation
    alpha_a_list, alpha_b_list = [], []
    lam_a_list, lam_b_list = [], []
    beta_sigma_list = []

    for ch in channel_cols:
        channel_meta = next((c for c in spec.channels if c.column == ch), None)
        ch_type = (channel_meta.channel_type if channel_meta and channel_meta.channel_type
                   else classify_channel(ch))

        base_prior = _get_channel_prior(ch_type, catalog)
        alpha_mu = base_prior["alpha_mu"]
        alpha_sigma = base_prior["alpha_sigma"]
        lam_mu = base_prior["lam_mu"]
        lam_sigma = base_prior["lam_sigma"]

        # Widen priors for sparse channels
        is_sparse = False
        if ch in df.columns:
            pct_zero = float((df[ch] == 0).mean())
            if pct_zero > 0.30:
                is_sparse = True
                alpha_sigma *= sparse_multiplier
                lam_sigma *= sparse_multiplier
                warnings.append(f"{ch}: sparse channel ({pct_zero*100:.1f}% zeros) — widened priors by {sparse_multiplier}x")

        # Widen priors if insufficient data
        if n_rows < catalog.get("min_obs_for_tight_priors", 52):
            alpha_sigma *= 1.3
            lam_sigma *= 1.3
            if ch == channel_cols[0]:  # warn once
                warnings.append(f"Short dataset ({n_rows} rows): widened all priors by 1.3x")

        # Moment-match to distribution parameters
        try:
            a_alpha, b_alpha = beta_moment_match(alpha_mu, alpha_sigma)
        except ValueError as e:
            warnings.append(f"{ch}: beta_moment_match error ({e}) — using fallback params")
            a_alpha, b_alpha = 2.0, 6.0

        try:
            a_lam, b_lam = gamma_moment_match(lam_mu, lam_sigma)
        except ValueError as e:
            warnings.append(f"{ch}: gamma_moment_match error ({e}) — using fallback params")
            a_lam, b_lam = 4.0, 1.0

        # saturation_beta sigma = spend share (spend-share sigma trick)
        beta_sig = spend_shares.get(ch, 1.0 / len(channel_cols))

        alpha_a_list.append(round(a_alpha, 4))
        alpha_b_list.append(round(b_alpha, 4))
        lam_a_list.append(round(a_lam, 4))
        lam_b_list.append(round(b_lam, 4))
        beta_sigma_list.append(round(beta_sig, 4))

        per_channel.append({
            "column": ch,
            "channel_type": ch_type,
            "is_sparse": is_sparse,
            "alpha_mu": alpha_mu,
            "alpha_sigma": alpha_sigma,
            "lam_mu": lam_mu,
            "lam_sigma": lam_sigma,
            "alpha_beta_params": [round(a_alpha, 4), round(b_alpha, 4)],
            "lam_gamma_params": [round(a_lam, 4), round(b_lam, 4)],
            "beta_sigma": round(beta_sig, 4),
            "spend_share_pct": round(spend_shares.get(ch, 0) * 100, 1),
            "rationale": base_prior.get("rationale", ""),
        })

    # Build model_config dict (JSON-serializable — numpy arrays serialized as lists)
    model_config = {
        "adstock_alpha": {
            "distribution": "Beta",
            "alpha": alpha_a_list,
            "beta": alpha_b_list,
            "dims": "channel",
            "_channel_order": channel_cols,
        },
        "saturation_lam": {
            "distribution": "Gamma",
            "alpha": lam_a_list,
            "beta": lam_b_list,
            "dims": "channel",
        },
        "saturation_beta": {
            "distribution": "HalfNormal",
            "sigma": beta_sigma_list,
            "dims": "channel",
        },
        "intercept": {
            "distribution": "Normal",
            "mu": 0.5,
            "sigma": 0.5,
        },
        "gamma_control": {
            "distribution": "Normal",
            "mu": 0.0,
            "sigma": 0.5,
            "dims": "control",
        },
        "gamma_fourier": {
            "distribution": "Laplace",
            "mu": 0.0,
            "b": 0.3,
            "dims": "fourier_mode",
        },
        "likelihood": {
            "distribution": "StudentT",
            "nu": 5,
            "sigma": {"distribution": "HalfNormal", "sigma": 0.5},
        },
        "_metadata": {
            "n_channels": len(channel_cols),
            "channel_order": channel_cols,
            "generated_at": datetime.now().isoformat(),
        },
    }

    result = {
        "generated_at": datetime.now().isoformat(),
        "n_channels": len(channel_cols),
        "model_config": model_config,
        "per_channel_audit": per_channel,
        "warnings": warnings,
        "prior_predictive_guidance": _prior_predictive_guidance(spec),
    }

    # Save artifacts
    ws = ensure_workspace(base)
    priors_dir = Path(ws) / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)

    with open(priors_dir / "model_config.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    report = _render_priors_report(result, spec)
    with open(priors_dir / "prior_audit_report.md", "w") as f:
        f.write(report)

    return result


def _prior_predictive_guidance(spec: MMMSpec) -> str:
    """Return a code snippet for running prior predictive checks."""
    return (
        "Run prior predictive check before fitting:\n"
        "  prior_pc = model.sample_prior_predictive(X=X, y=y_series, samples=500)\n"
        "  Check: 90% credible interval should contain observed target range.\n"
        "  If interval is too wide: tighten alpha_sigma and lam_sigma by 0.5x.\n"
        "  If interval is too narrow: widen by 1.5x.\n"
        "  Target range: [target.min(), target.max()] after MaxAbsScaler normalization → [0, 1]."
    )


def _render_priors_report(result: dict, spec: MMMSpec) -> str:
    lines = [
        "# MMM Prior Recommendations",
        "",
        f"**Company**: {spec.company_name} | **Industry**: {spec.industry} | **Region**: {spec.region}",
        f"**Generated**: {result['generated_at'][:19]}  ",
        f"**Channels**: {result['n_channels']}",
        "",
        "---",
        "",
    ]

    if result["warnings"]:
        lines += ["## Warnings", ""]
        for w in result["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines += [
        "## Per-Channel Prior Parameters",
        "",
        "| Channel | Type | alpha mu | alpha sigma | lam mu | lam sigma | Spend Share | Notes |",
        "|---------|------|----------|-------------|--------|-----------|-------------|-------|",
    ]

    for ch in result["per_channel_audit"]:
        sparse_flag = " sparse" if ch["is_sparse"] else ""
        lines.append(
            f"| `{ch['column']}` | {ch['channel_type']} "
            f"| {ch['alpha_mu']:.2f} | {ch['alpha_sigma']:.3f} "
            f"| {ch['lam_mu']:.2f} | {ch['lam_sigma']:.3f} "
            f"| {ch['spend_share_pct']:.1f}% "
            f"| {sparse_flag} |"
        )

    lines += [
        "",
        "## Adstock Configuration",
        "",
        "| Channel Type | Recommended Adstock | l_max |",
        "|-------------|---------------------|-------|",
        "| SEM, Social, Display, Meta | GeometricAdstock | 5 |",
        "| YouTube, Audio | GeometricAdstock | 8 |",
        "| OOH, TV, Print | DelayedAdstock | 12 |",
        "",
        "Note: pymc-marketing requires one adstock type for all channels in one model.",
        "Recommendation: Use `GeometricAdstock(l_max=8)` as default unless all channels are offline.",
        "",
        "## Prior Predictive Check",
        "",
        f"```\n{result['prior_predictive_guidance']}\n```",
        "",
        "## Python Code to Apply These Priors",
        "",
        "```python",
        "import numpy as np",
        "from pymc_extras.prior import Prior",
        "from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation",
        "from pymc_marketing.mmm.multidimensional import MMM",
        "",
    ]

    channels = result["model_config"]["_metadata"]["channel_order"]
    cfg = result["model_config"]
    lines += [
        f"channels = {channels}",
        "",
        "model_config = {",
        '    "adstock_alpha": Prior("Beta",',
        f'        alpha=np.array({cfg["adstock_alpha"]["alpha"]}),',
        f'        beta=np.array({cfg["adstock_alpha"]["beta"]}),',
        '        dims="channel"),',
        '    "saturation_lam": Prior("Gamma",',
        f'        alpha=np.array({cfg["saturation_lam"]["alpha"]}),',
        f'        beta=np.array({cfg["saturation_lam"]["beta"]}),',
        '        dims="channel"),',
        '    "saturation_beta": Prior("HalfNormal",',
        f'        sigma=np.array({cfg["saturation_beta"]["sigma"]}),',
        '        dims="channel"),',
        '    "intercept": Prior("Normal", mu=0.5, sigma=0.5),',
        '    "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),',
        '    "gamma_fourier": Prior("Laplace", mu=0, b=0.3, dims="fourier_mode"),',
        '    "likelihood": Prior("StudentT", nu=5, sigma=Prior("HalfNormal", sigma=0.5)),',
        "}",
        "",
        "model = MMM(",
        '    date_column="date",',
        f'    channel_columns=channels,',
        f'    target_column="{spec.target_column}",',
        "    adstock=GeometricAdstock(l_max=8),",
        "    saturation=LogisticSaturation(),",
        f"    yearly_seasonality={spec.seasonality.yearly_fourier_modes},",
        "    model_config=model_config,",
        "    adstock_first=True,",
        ")",
        "```",
        "",
        "---",
        "*Generated by agent-mmm prior recommendation engine*",
        "",
    ]
    return "\n".join(lines)
