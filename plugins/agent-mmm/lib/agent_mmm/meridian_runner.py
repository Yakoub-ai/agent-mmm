"""Fit Google Meridian in its own environment and write framework-neutral results.

Meridian (TensorFlow/JAX) cannot share an environment with PyMC, so it runs as a
separate process in ``.venv-meridian``::

    .venv-meridian/bin/python -P <plugin>/lib/agent_mmm/meridian_runner.py \\
        --spec mmm-workspace/spec.yaml --run-id 2026-11-14T09-00_meridian --holdout auto

From the main environment, :func:`launch` builds that argv (no shell) and runs it.

The runner uses the same holdout split, the same channel names (the spec's
model-input columns) and the same scoring code (``results_schema``) as the
pymc backend, so the two results files compare directly.

Verified against google-meridian 2.2. Meridian is imported inside functions so
this module can be imported from the main environment to call :func:`launch`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MERIDIAN_VENV_PYTHON = ".venv-meridian/bin/python"


# --------------------------------------------------------------------------- #
# Launching from the main environment
# --------------------------------------------------------------------------- #
def launch(
    spec_path: str,
    run_id: str,
    base: str | Path = ".",
    holdout: str = "auto",
    python: Optional[str] = None,
    chains: int = 4,
    adapt: int = 500,
    burnin: int = 500,
    keep: int = 500,
    seed: int = 42,
    timeout: int = 6 * 3600,
) -> Path:
    """Run the Meridian fit in its own interpreter and return the results path."""
    base = Path(base).resolve()
    py = python or os.environ.get("MMM_MERIDIAN_PYTHON") or str(base / MERIDIAN_VENV_PYTHON)
    if not Path(py).exists():
        raise FileNotFoundError(f"No Meridian environment at {py}. Run /mmm-setup with profile 'meridian'.")
    argv = [
        py, "-P", str(Path(__file__).resolve()),
        "--spec", spec_path, "--run-id", run_id, "--base", str(base), "--holdout", str(holdout),
        "--chains", str(chains), "--adapt", str(adapt), "--burnin", str(burnin),
        "--keep", str(keep), "--seed", str(seed),
    ]
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    p = subprocess.run(argv, cwd=base, env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"Meridian runner failed with exit code {p.returncode}")
    return base / "mmm-workspace" / "runs" / run_id / "results.json"


# --------------------------------------------------------------------------- #
# Fitting (inside .venv-meridian)
# --------------------------------------------------------------------------- #
def _flat(a: Any) -> "Any":
    import numpy as np

    a = np.asarray(a, dtype=float)
    return a.reshape(-1, *a.shape[2:])          # (chain, draw, ...) -> (S, ...)


def build_input(spec: Any, raw: "Any"):
    """Map the spec onto Meridian's data loader. Returns (input_data, channels, notes)."""
    from meridian.data import load

    from agent_mmm.spec import ChannelRole, TargetUnitKind

    notes: list[str] = []
    df = raw.copy()
    df[spec.date_column] = df[spec.date_column].dt.strftime("%Y-%m-%d")

    paid = spec.channels_with_role(ChannelRole.paid_media)
    media_cols, spend_cols, channels = [], [], []
    for c in paid:
        if not c.spend_column:
            notes.append(f"{c.column}: no spend column; Meridian needs spend for paid media, skipped.")
            continue
        media_cols.append(c.model_input_column)
        spend_cols.append(c.spend_column)
        channels.append(c.model_input_column)
    skipped = [c.column for c in spec.active_channels()
               if c.role in (ChannelRole.paid_reach_frequency, ChannelRole.organic_media,
                             ChannelRole.non_media_treatment)]
    if skipped:
        notes.append(f"Not passed to Meridian in this runner (RF/organic/non-media): {skipped}. "
                     "They are absent from the comparison for Meridian.")
    # One spend column cannot back two channels in Meridian.
    if len(set(spend_cols)) != len(spend_cols):
        raise ValueError("Two Meridian channels share a spend column; give each its own.")

    kwargs: dict[str, Any] = dict(
        time=spec.date_column, kpi=spec.target_column,
        media=media_cols, media_spend=spend_cols,
        controls=spec.control_columns() or None,
    )
    if spec.geo.is_panel:
        kwargs["geo"] = spec.geo.geo_column
        if spec.geo.population_column:
            kwargs["population"] = spec.geo.population_column
        else:
            notes.append("Geo panel without population: Meridian assumes equal population.")
    if spec.target_unit.revenue_per_unit_column:
        kwargs["revenue_per_kpi"] = spec.target_unit.revenue_per_unit_column

    coord = load.CoordToColumns(**kwargs)
    kpi_type = "revenue" if spec.target_unit.kind == TargetUnitKind.monetary else "non_revenue"
    loader = load.DataFrameDataLoader(
        df=df, coord_to_columns=coord, kpi_type=kpi_type,
        media_to_channel=dict(zip(media_cols, channels)),
        media_spend_to_channel=dict(zip(spend_cols, channels)),
    )
    return loader.load(), channels, notes


def _quiet() -> None:
    import logging
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("absl").setLevel(logging.ERROR)


def fit_and_extract(spec_path: str, run_id: str, base: str, holdout: str, chains: int,
                    adapt: int, burnin: int, keep: int, seed: int) -> Path:
    _quiet()
    import numpy as np
    import pandas as pd
    from meridian.analysis import analyzer
    from meridian.model import model, spec as mspec

    from agent_mmm.results_schema import DEFAULT_MULTIPLIERS, build_results, holdout_mask
    from agent_mmm.spec import TargetUnitKind, load_spec
    from agent_mmm.utils.io import load_data, parse_dates

    base_p = Path(base)
    spec = load_spec(base_p / spec_path if not Path(spec_path).is_absolute() else spec_path)
    data_path = Path(spec.data_path)
    raw = parse_dates(load_data(data_path if data_path.is_absolute() else base_p / data_path), spec.date_column)
    data, channels, notes = build_input(spec, raw)

    times = [str(t)[:10] for t in data.kpi.coords["time"].values]
    hmask = holdout_mask(len(times), spec.periods_per_year(), holdout if holdout != "0" else 0,
                         dates=times, start=spec.validation.holdout_start, end=spec.validation.holdout_end)
    n_geo = data.kpi.sizes.get("geo", 1)
    holdout_id = np.tile(hmask, (n_geo, 1)) if spec.geo.is_panel else hmask
    model_spec = mspec.ModelSpec(
        holdout_id=holdout_id if hmask.any() else None,
        max_lag=spec.architecture.default_l_max,
    )
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    mmm.sample_posterior(n_chains=chains, n_adapt=adapt, n_burnin=burnin, n_keep=keep, seed=seed)

    try:  # 2.2+: model_context; older versions take the model positionally
        a = analyzer.Analyzer(model_context=mmm.model_context, inference_data=mmm.inference_data)
    except (TypeError, AttributeError):
        a = analyzer.Analyzer(mmm)
    use_kpi = spec.target_unit.kind != TargetUnitKind.monetary
    train_times = [t for t, h in zip(times, hmask) if not h]

    expected = _flat(a.expected_outcome(aggregate_times=False, use_kpi=use_kpi))            # (S, T)
    contrib = _flat(a.incremental_outcome(aggregate_times=False, use_kpi=use_kpi,
                                          include_non_paid_channels=False))                # (S, T, C)
    contrib = contrib[:, ~hmask, :]
    curves = []
    for m in DEFAULT_MULTIPLIERS:
        if m == 0:
            curves.append(np.zeros((expected.shape[0], len(channels))))
            continue
        curves.append(_flat(a.incremental_outcome(
            selected_times=train_times, scaling_factor1=m, use_kpi=use_kpi,
            include_non_paid_channels=False)))
    stacked = np.stack(curves, axis=-1)                                                     # (S, C, M)

    dates = pd.to_datetime(times)
    by_date = raw.groupby(spec.date_column)
    spend_map = spec.spend_columns()
    spend = np.column_stack([
        by_date[spend_map[ch]].sum().reindex(dates).to_numpy(dtype=float) for ch in channels
    ])
    actual = by_date[spec.target_column].sum().reindex(dates).to_numpy(dtype=float)

    diag: dict[str, Any] = {}
    try:
        rh = a.get_rhat()
        vals = [float(np.nanmax(np.asarray(v))) for v in rh.values() if np.isfinite(np.asarray(v)).any()]
        diag["max_rhat"] = max(vals) if vals else None
        idata = mmm.inference_data
        diag["divergences"] = int(np.asarray(idata.sample_stats["diverging"]).sum())
        diag["n_chains"], diag["n_draws"] = chains, keep
        import arviz as az
        ess = az.ess(idata.posterior)
        ev = [float(np.nanmin(np.asarray(ess[v]))) for v in ess.data_vars if np.isfinite(np.asarray(ess[v])).any()]
        diag["min_ess"] = min(ev) if ev else None
    except Exception as e:
        diag["error"] = str(e)

    notes.append("Meridian uses its default ROI priors here; pymc-marketing uses the plugin's prior "
                 "engine. Part of any difference between them comes from the priors.")
    try:
        from importlib.metadata import version
        mer_version = version("google-meridian")
    except Exception:
        mer_version = None
    res = build_results(
        framework="meridian", run_id=run_id, target=spec.target_column,
        target_label=spec.target_unit.label,
        kpi_kind="revenue" if not use_kpi else "non_revenue",
        channels=channels, dates=times, actual=actual, expected=expected,
        contributions=contrib, spend=spend, holdout=hmask,
        curve_draws={ch: stacked[:, j, :] for j, ch in enumerate(channels)},
        diagnostics=diag,
        provenance={"data_path": str(spec.data_path), "meridian": mer_version,
                    "sampler": {"chains": chains, "adapt": adapt, "burnin": burnin, "keep": keep, "seed": seed}},
        notes=notes,
    )
    run_dir = base_p / "mmm-workspace" / "runs" / run_id
    out = res.save(run_dir / "results.json")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "framework": "meridian", "completed_at": datetime.now().isoformat(),
        "results_path": str(out), "holdout_fit": {k: v for k, v in res.fit.items() if "holdout" in k},
        "diagnostics": diag,
    }, indent=2, default=str), encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="meridian_runner")
    ap.add_argument("--spec", default="mmm-workspace/spec.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base", default=".")
    ap.add_argument("--holdout", default="auto")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--adapt", type=int, default=500)
    ap.add_argument("--burnin", type=int, default=500)
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if any(c in a.run_id for c in "/\\") or a.run_id in ("", ".", ".."):
        ap.error("--run-id must be a plain name")
    holdout = a.holdout if a.holdout in ("auto", "0") else str(int(a.holdout))
    out = fit_and_extract(a.spec, a.run_id, a.base, holdout, a.chains, a.adapt, a.burnin, a.keep, a.seed)
    print(f"Meridian results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
