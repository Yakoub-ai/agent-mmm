"""Data Science report: full diagnostics, model spec, convergence stats, reproducibility."""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

from agent_mmm.spec import MMMSpec
from agent_mmm.workspace import ensure_workspace


def generate_ds_report(
    spec: MMMSpec,
    run_id: str,
    metrics: dict | None = None,
    diagnostics: dict | None = None,
    base: str | Path = ".",
) -> str:
    """Generate Data Science reproducibility and diagnostics report."""
    ws = ensure_workspace(base)
    report_date = datetime.now().strftime("%B %d, %Y")

    lines = [
        "# Marketing Mix Model — Data Science Report",
        "",
        f"**Company**: {spec.company_name}  ",
        f"**Report Date**: {report_date}  ",
        f"**Run ID**: `{run_id}`  ",
        "",
        "---",
        "",
        "## Model Specification",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| MMM Type | {spec.mmm_type.value} |",
        f"| Data Path | `{spec.data_path}` |",
        f"| Target Column | `{spec.target_column}` |",
        f"| Target Unit | {spec.target_unit.label} ({spec.target_unit.kind.value}) |",
        f"| Channels | {len(spec.channel_columns())} |",
        f"| Controls | {len(spec.control_columns())} |",
        f"| Granularity | {spec.granularity.value} |",
        f"| Fourier Modes | {spec.seasonality.yearly_fourier_modes} |",
        f"| Multi-geo | {spec.geo.is_panel} |",
        "",
        "### Channel Configuration",
        "",
        "| Column | Type | Label | Active |",
        "|--------|------|-------|--------|",
    ]

    from agent_mmm.utils.channel_classifier import classify_channel

    for ch in spec.channels:
        ch_type = ch.channel_type or classify_channel(ch.column)
        lines.append(
            f"| `{ch.column}` | {ch_type} | {ch.label or '—'} | {'yes' if ch.is_active else 'no'} |"
        )

    lines += ["", "### Control Variables", ""]
    if spec.controls:
        lines += ["| Column | Label | Source |", "|--------|-------|--------|"]
        for ctrl in spec.controls:
            lines.append(f"| `{ctrl.column}` | {ctrl.label or '—'} | {ctrl.source} |")
    else:
        lines.append("*No controls configured.*")

    lines += ["", "## Fit Metrics", ""]
    if metrics:
        for k, v in metrics.items():
            if k not in (
                "run_id",
                "completed_at",
                "sampler_config",
                "prior_pc",
                "idata_path",
                "model_path",
                "save_error",
                "provenance",
                "spec",
                "calibration",
                "started_at",
                "framework",
            ):
                lines.append(f"- **{k}**: {v}")
    else:
        lines.append("*Metrics not available.*")

    lines += ["", "## Diagnostics", ""]
    if diagnostics:
        tier = diagnostics.get("summary", {}).get("tier", "UNKNOWN")
        lines.append(f"**Overall Tier**: {tier}")

        conv = diagnostics.get("checks", {}).get("convergence", {})
        if conv.get("available"):
            lines += [
                "",
                "### Convergence",
                "",
                f"- Max rhat: `{conv.get('max_rhat', 'N/A')}`",
                f"- Min ESS_bulk: `{conv.get('min_ess_bulk', 'N/A')}`",
                f"- Divergences: `{conv.get('n_divergences', 'N/A')}`",
            ]

        ov = diagnostics.get("checks", {}).get("overfit", {})
        if ov.get("available"):
            lines += [
                "",
                "### Generalisation",
                "",
                f"- In-sample R2: `{ov.get('in_sample_r2')}`",
                f"- CV R2: `{ov.get('cv_r2')}`",
                f"- Gap: `{ov.get('gap')}` (threshold: 0.20)",
                f"- Overfit: `{ov.get('overfit')}`",
            ]
        elif ov.get("note"):
            lines += ["", "### Generalisation", "", f"- {ov['note']}"]

        # The baseline decides whether any channel number below it is readable,
        # so it belongs above the attribution section, not in an appendix.
        base = diagnostics.get("checks", {}).get("baseline", {})
        if base.get("available"):
            lines += [
                "",
                "### Baseline health",
                "",
                f"- Baseline share of target: `{base.get('baseline_share')}`",
                f"- Periods with a negative baseline: `{base.get('negative_periods')}` "
                f"({base.get('negative_pct')}%)",
                f"- Baseline drift start-to-end: `{base.get('trend_drift_pct')}%`",
                f"- Components counted: {', '.join(base.get('components', []))}",
            ]

        dec = diagnostics.get("checks", {}).get("decomposition", {})
        if dec.get("available"):
            lines += ["", "### Decomposition", "", "| Component | Share | Total |", "|---|---|---|"]
            for comp, share in (dec.get("shares") or {}).items():
                total = (dec.get("totals") or {}).get(comp)
                lines.append(f"| {comp} | {share} | {total} |")

        contraction = diagnostics.get("checks", {}).get("prior_contraction", {})
        real = {
            k: v for k, v in contraction.items()
            if isinstance(v, dict) and "contraction" in v
        }
        if real:
            lines += [
                "",
                "### What the data taught us",
                "",
                "Contraction is `1 - sd(posterior)/sd(prior)`. Below 0.2 the posterior is "
                "largely the prior restated, and that parameter is an assumption rather "
                "than a finding.",
                "",
                "| Parameter | Prior sd | Posterior sd | Contraction |",
                "|---|---|---|---|",
            ]
            for k, v in sorted(real.items(), key=lambda kv: kv[1]["contraction"]):
                flag = " (prior-dominated)" if v["prior_dominated"] else ""
                lines.append(
                    f"| {k} | {v['prior_sd']} | {v['posterior_sd']} | "
                    f"{v['contraction']:.2f}{flag} |"
                )

        pl = diagnostics.get("checks", {}).get("attribution_plausibility", {})
        if pl.get("available"):
            lines += [
                "",
                "### Attribution plausibility",
                "",
                f"- Media share of modelled target: `{pl.get('media_share')}`",
            ]
            if pl.get("channel_shares_pct"):
                lines += ["", "| Channel | Share of media effect |", "|---|---|"]
                for ch, pct in pl["channel_shares_pct"].items():
                    lines.append(f"| {ch} | {pct}% |")

        if diagnostics.get("errors"):
            lines += ["", "### Errors", ""]
            for e in diagnostics["errors"]:
                lines.append(f"- {e}")

        if diagnostics.get("warnings"):
            lines += ["", "### Warnings", ""]
            for w in diagnostics["warnings"]:
                lines.append(f"- {w}")

    # Calibration: whether any channel is anchored to a measurement, and which.
    cal = (metrics or {}).get("calibration") or {}
    lines += ["", "## Calibration", ""]
    if cal.get("applied"):
        lines += [
            f"- {cal.get('n_tests')} lift test(s) attached as likelihood constraints.",
            f"- Channels calibrated: {', '.join(cal.get('channels', []))}",
            "",
            "Calibrated channels carry an external anchor; the rest rest on observational "
            "identification alone and deserve wider ranges in any recommendation.",
        ]
    else:
        lines += [
            f"- No lift-test constraints applied. {cal.get('reason', '')}".rstrip(),
            "",
            "Without an experiment, several very different attributions fit this data "
            "equally well; the one reported is the one the priors preferred. This is the "
            "model's largest limitation.",
        ]

    ppc = (metrics or {}).get("prior_pc") or {}
    if ppc.get("ran") and ppc.get("coverage_90") is not None:
        lines += [
            "",
            "## Prior predictive check",
            "",
            f"- 90% band coverage of observed periods: `{ppc.get('coverage_90')}`",
            f"- Band width vs data range: `{ppc.get('band_width_vs_data_range')}x`",
        ]
        for note in ppc.get("notes", []):
            lines.append(f"- {note}")

    # Reproducibility
    py_ver = sys.version.split()[0]
    try:
        import pymc_marketing

        pmm_ver = pymc_marketing.__version__
    except ImportError:
        pmm_ver = "not installed"
    try:
        import arviz as az

        az_ver = az.__version__
    except ImportError:
        az_ver = "not installed"

    lines += [
        "",
        "## Reproducibility",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Generated at**: {report_date}",
        f"- **Python**: {py_ver}",
        f"- **pymc-marketing**: {pmm_ver}",
        f"- **arviz**: {az_ver}",
        f"- **Model / inference data**: `./mmm-workspace/runs/{run_id}/model.nc` "
        "(an `xarray.DataTree`)",
        f"- **Metrics**: `./mmm-workspace/runs/{run_id}/metrics.json`",
        "",
        "---",
        f"*Generated by agent-mmm | Run `{run_id}` | {report_date}*",
        "",
    ]

    prov = (metrics or {}).get("provenance") or {}
    if prov:
        lines += [
            f"- **Data fingerprint**: `{prov.get('data_fingerprint')}`",
            f"- **Random seed**: `{prov.get('random_seed')}`",
        ]
        env = prov.get("environment") or {}
        if env:
            versions = ", ".join(f"{k} {v}" for k, v in env.items() if v)
            lines.append(f"- **Environment at fit time**: {versions}")

    report_str = "\n".join(lines)
    ws_reports = ws / "reports"
    ws_reports.mkdir(parents=True, exist_ok=True)
    with open(ws_reports / "ds.md", "w") as f:
        f.write(report_str)
    return report_str
