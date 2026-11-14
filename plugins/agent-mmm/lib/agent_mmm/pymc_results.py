"""Extract framework-neutral results from a fitted pymc-marketing MMM.

Hands raw arrays in original units to ``results_schema.build_results``; no
metric is computed here, so pymc and Meridian are scored by the same code.

Verified against pymc-marketing 1.4:

* ``channel_contribution`` and ``mu`` are on the scaled target; multiply by
  ``target_scale`` (``mu_original_scale`` exists under a log link and is used
  instead).
* Response curves re-run the fitted graph on scaled spend with
  ``sample_posterior_predictive(var_names=["channel_contribution"])``. Channel
  contributions are separable, so scaling every channel by ``m`` in one call
  gives each channel's response at ``m``.
* Holdout predictions use ``include_last_observations=True`` so adstock carries
  over from the last training periods.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from agent_mmm.results_schema import DEFAULT_MULTIPLIERS, ModelResults, build_results
from agent_mmm.spec import MMMSpec, TargetUnitKind

logger = logging.getLogger(__name__)
SAMPLE_DIMS = ("chain", "draw", "sample")


def _to_array(da: Any, keep: tuple[str, ...]) -> np.ndarray:
    """xarray -> (samples, *keep), summing every other non-sample dim (e.g. geo)."""
    sdims = [d for d in da.dims if d in SAMPLE_DIMS]
    other = [d for d in da.dims if d not in sdims and d not in keep]
    if other:
        da = da.sum(other)
    if len(sdims) == 2:
        da = da.stack(_s=("chain", "draw"))
    elif sdims:
        da = da.rename({sdims[0]: "_s"})
    else:
        da = da.expand_dims("_s")
    return np.asarray(da.transpose("_s", *keep).values, dtype=float)


def _expected_var(model: Any) -> tuple[str, bool]:
    post = model.idata["posterior"]
    if "mu_original_scale" in post.data_vars:
        return "mu_original_scale", True
    return "mu", False


def _diagnostics(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import arviz as az

        post = model.idata["posterior"]
        post = post.to_dataset() if hasattr(post, "to_dataset") else post
        params = [v for v in post.data_vars if "date" not in post[v].dims]
        rhat = az.rhat(post[params])
        ess = az.ess(post[params])
        out["max_rhat"] = float(max(float(rhat[v].max()) for v in params))
        out["min_ess"] = float(min(float(ess[v].min()) for v in params))
    except Exception as e:  # diagnostics must never lose a fit
        out["error"] = f"rhat/ess unavailable: {e}"
    try:
        ss = model.idata["sample_stats"]
        out["divergences"] = int(np.asarray(ss["diverging"]).sum())
        out["n_chains"] = int(ss.sizes.get("chain", 0))
        out["n_draws"] = int(ss.sizes.get("draw", 0))
    except Exception:
        pass
    return out


def extract_pymc_results(
    model: Any,
    spec: MMMSpec,
    X: pd.DataFrame,
    y: pd.Series,
    raw: pd.DataFrame,
    holdout_dates: set,
    run_id: str,
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    provenance: Optional[dict] = None,
) -> ModelResults:
    """``X``/``y``/``raw`` cover every period; the model was fitted on non-holdout rows."""
    date_col = spec.date_column
    channels = spec.channel_columns()
    scale = float(model.get_scales_as_xarray()["target_scale"])
    var, original = _expected_var(model)
    k = 1.0 if original else scale

    dates = sorted(pd.to_datetime(X[date_col]).unique())
    dkeys = [str(d)[:10] for d in dates]
    holdout = np.array([d in holdout_dates for d in dkeys])
    row_hold = pd.to_datetime(X[date_col]).dt.strftime("%Y-%m-%d").isin(holdout_dates).to_numpy()
    X_train = X[~row_hold]

    # Expected outcome: posterior for training periods, forecast for holdout.
    exp_train = _to_array(model.idata["posterior"][var], ("date",)) * k
    if holdout.any():
        pp = model.sample_posterior_predictive(
            X[row_hold], include_last_observations=True, var_names=[var],
            extend_idata=False, progressbar=False,
        )
        exp_hold = _to_array(pp[var], ("date",)) * k
        n = min(len(exp_train), len(exp_hold))
        expected = np.concatenate([exp_train[:n], exp_hold[:n]], axis=1)
    else:
        expected = exp_train

    contributions = _to_array(model.idata["posterior"]["channel_contribution"], ("date", "channel")) * scale

    # Response curves over the training window.
    per_m = []
    for m in multipliers:
        Xs = X_train.copy()
        Xs[channels] = Xs[channels].astype(float) * m
        pp = model.sample_posterior_predictive(
            Xs, var_names=["channel_contribution"], extend_idata=False, progressbar=False,
        )
        per_m.append(_to_array(pp["channel_contribution"], ("channel",)) * scale)   # (S, C)
    stacked = np.stack(per_m, axis=-1)                                                # (S, C, M)
    curve_draws = {ch: stacked[:, j, :] for j, ch in enumerate(channels)}

    # Spend and actuals per period, summed over geos.
    raw = raw.copy()
    raw[date_col] = pd.to_datetime(raw[date_col])
    by_date = raw.groupby(date_col)
    spend_map = spec.spend_columns()
    spend = np.full((len(dates), len(channels)), np.nan)
    for j, ch in enumerate(channels):
        col = spend_map.get(ch)
        if col and col in raw.columns:
            spend[:, j] = by_date[col].sum().reindex(dates).to_numpy(dtype=float)
    actual = (
        pd.DataFrame({date_col: pd.to_datetime(X[date_col]).to_numpy(), "y": y.to_numpy()})
        .groupby(date_col)["y"].sum().reindex(dates).to_numpy(dtype=float)
    )

    kind = "revenue" if spec.target_unit.kind == TargetUnitKind.monetary else "non_revenue"
    notes = []
    if spec.geo.is_panel:
        notes.append("Geo panel: results are summed across geos for comparison.")
    return build_results(
        framework="pymc-marketing", run_id=run_id, target=spec.target_column,
        target_label=spec.target_unit.label, kpi_kind=kind, channels=channels,
        dates=dkeys, actual=actual, expected=expected, contributions=contributions,
        spend=spend, holdout=holdout, curve_draws=curve_draws, multipliers=multipliers,
        diagnostics=_diagnostics(model), provenance=provenance, notes=notes,
    )
