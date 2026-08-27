"""Fit pipeline: prior predictive -> calibrate -> fit -> posterior predictive -> save.

The order matters and each step earns its place:

* **Prior predictive first.** Looking at what the priors imply before seeing the
  data is the only chance to catch a prior that says media drives 300% of sales.
  Afterwards you cannot un-see the answer.
* **Calibration before fitting.** Lift-test constraints are part of the model, not
  a post-hoc adjustment, so they have to be attached to the built graph.
* **Rescale before scoring.** pymc-marketing fits on max-abs-scaled data and
  returns normalised posterior predictive draws. Metrics computed on the raw
  output are metrics of the wrong quantity.
* **Per-draw metrics.** E[f(x)] != f(E[x]): R² of the posterior mean flatters the
  model, because averaging draws cancels noise the model never actually removed.
* **Record provenance.** A run nobody can reproduce is an anecdote.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_mmm.model_factory import build_mmm, prepare_data
from agent_mmm.spec import MMMSpec
from agent_mmm.workspace import ensure_run_dir, new_run_id

logger = logging.getLogger(__name__)

# Named sampler profiles. target_accept trades wall time for a smaller step
# size; raising it is the first fix for divergences, not a default to always max.
SAMPLER_SMOKE = {"draws": 200, "tune": 300, "chains": 2, "target_accept": 0.90}
SAMPLER_QUICK = {"draws": 500, "tune": 1000, "chains": 4, "target_accept": 0.90}
SAMPLER_CV = {"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.95}
SAMPLER_FINAL = {"draws": 2000, "tune": 3000, "chains": 4, "target_accept": 0.97}


def data_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the modelling frame, so a run can be tied to its input."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("pymc_marketing", "pymc", "arviz", "pytensor", "numpy", "pandas"):
        try:
            meta[name] = __import__(name).__version__
        except Exception:
            meta[name] = None
    return meta


def _lift_test_frame(spec: MMMSpec) -> pd.DataFrame | None:
    """Build the lift-test frame pymc-marketing expects, or None if unusable.

    A lift test constrains the saturation curve between ``x`` and ``x + delta_x``.
    That needs all three of a starting spend, a spend delta, and a measured
    outcome delta with its uncertainty — an experiment missing any of them can
    inform a prior but cannot be a likelihood term.
    """
    rows = []
    for e in spec.experiments:
        if e.lift_absolute is None or e.lift_se is None or not e.spend_during_test:
            continue
        ch = spec.channel_by_name(e.channel)
        rows.append({
            "channel": (ch.model_input_column if ch else e.channel),
            "x": float(e.baseline_spend or 0.0),
            "delta_x": float(e.spend_during_test),
            "delta_y": float(e.lift_absolute),
            "sigma": float(e.lift_se),
        })
    return pd.DataFrame(rows) if rows else None


def run_fit(
    spec: MMMSpec,
    model_config_dict: dict | None = None,
    model_config_path: str | Path | None = None,
    priors: dict | None = None,
    run_id: str | None = None,
    sampler_config: dict | None = None,
    base: str | Path = ".",
    skip_prior_pc: bool = False,
    prior_pc_samples: int = 500,
    apply_lift_tests: bool = True,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit one model end to end and write its artefacts under ``runs/<run_id>/``."""
    run_id = run_id or new_run_id()
    sampler_config = sampler_config or {
        "draws": spec.sampler.draws,
        "tune": spec.sampler.tune,
        "chains": spec.sampler.chains,
        "target_accept": spec.sampler.target_accept,
    }
    run_dir = ensure_run_dir(run_id, base)

    if priors is None:
        if model_config_dict is not None:
            priors = {"model_config": model_config_dict}
        elif model_config_path is not None:
            with open(model_config_path) as f:
                raw = json.load(f)
            priors = raw if "model_config" in raw else {"model_config": raw}

    X, y = prepare_data(spec, df=df)
    model = build_mmm(spec, priors=priors, df=df)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "framework": spec.framework.value,
        "sampler_config": sampler_config,
        "spec": {
            "company": spec.company_name,
            "channels": spec.channel_columns(),
            "controls": spec.control_columns(),
            "target": spec.target_column,
            "granularity": spec.granularity.value,
        },
        "provenance": {
            "data_path": str(spec.data_path),
            "data_fingerprint": data_fingerprint(X.assign(**{spec.target_column: y.values})),
            "random_seed": spec.sampler.random_seed,
            "environment": environment_metadata(),
        },
    }

    # The graph has to exist before priors can be sampled or constraints attached.
    model.build_model(X, y)

    # --- 1. Prior predictive -------------------------------------------- #
    if not skip_prior_pc:
        try:
            model.sample_prior_predictive(X=X, y=y, samples=prior_pc_samples, extend_idata=True)
            metrics["prior_pc"] = _summarise_prior_predictive(model, y, spec)
        except Exception as e:
            logger.warning("Prior predictive check failed: %s", e)
            metrics["prior_pc"] = {"ran": False, "error": str(e)}

    # --- 2. Calibration -------------------------------------------------- #
    lift = _lift_test_frame(spec) if apply_lift_tests else None
    if lift is not None:
        try:
            model.add_lift_test_measurements(lift)
            metrics["calibration"] = {
                "applied": True,
                "n_tests": int(len(lift)),
                "channels": sorted(lift["channel"].unique().tolist()),
            }
        except Exception as e:
            logger.warning("Could not attach lift tests: %s", e)
            metrics["calibration"] = {"applied": False, "error": str(e)}
    else:
        metrics["calibration"] = {
            "applied": False,
            "reason": "No experiment in the spec has all of baseline spend, spend delta, "
                      "measured lift and its standard error.",
        }

    # --- 3. Fit ----------------------------------------------------------- #
    logger.info("Fitting %s with %s", run_id, sampler_config)
    model.fit(X, y, random_seed=spec.sampler.random_seed, **sampler_config)

    # --- 4. Posterior predictive ------------------------------------------ #
    try:
        model.sample_posterior_predictive(X, extend_idata=True)
    except Exception as e:
        logger.warning("Posterior predictive sampling failed: %s", e)

    metrics.update(compute_insample_metrics(model, y, spec))
    metrics["completed_at"] = datetime.now().isoformat()

    # --- 5. Persist -------------------------------------------------------- #
    model_path = run_dir / "model.nc"
    try:
        model.save(str(model_path))
        metrics["model_path"] = str(model_path)
        metrics["idata_path"] = str(model_path)
    except Exception as e:
        logger.warning("Could not save the model: %s", e)
        metrics["model_path"] = None
        metrics["save_error"] = (
            f"{e}. Saving a DataTree to netCDF needs h5netcdf or netCDF4 installed."
        )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    r2 = metrics.get("r2_insample")
    logger.info("Run %s complete. In-sample R2=%s", run_id, "n/a" if r2 is None else f"{r2:.3f}")
    return metrics


def _summarise_prior_predictive(model: Any, y: pd.Series, spec: MMMSpec) -> dict[str, Any]:
    """Does the prior predictive cover the data without swamping it?

    Coverage far below 90% means the priors rule out what actually happened.
    Coverage at 100% with a band orders of magnitude wider than the data means
    the priors are uninformative and the sampler will spend its time in regions
    the business would call absurd.
    """
    try:
        prior_group = model.idata["prior_predictive"]
        var = spec.target_column if spec.target_column in prior_group else list(prior_group.data_vars)[0]
        draws = np.asarray(prior_group[var])
        scale = float(model.get_scales_as_xarray()["target_scale"])
        obs = y.to_numpy(dtype=float)

        flat = draws.reshape(-1, draws.shape[-1]) if draws.ndim > 1 else draws.reshape(1, -1)
        if flat.shape[-1] != len(obs):
            flat = flat.reshape(-1, len(obs))
        lo = np.percentile(flat, 5, axis=0) * scale
        hi = np.percentile(flat, 95, axis=0) * scale
        covered = float(np.mean((obs >= lo) & (obs <= hi)))
        width_ratio = float(np.mean(hi - lo) / (obs.max() - obs.min())) if obs.max() > obs.min() else None

        notes = []
        if covered < 0.8:
            notes.append(
                f"Only {covered:.0%} of observed periods fall inside the 90% prior band. The "
                "priors assign little probability to what actually happened — widen them or "
                "revisit the intercept."
            )
        if width_ratio is not None and width_ratio > 10:
            notes.append(
                f"The prior band is {width_ratio:.0f}x wider than the observed range. Such vague "
                "priors slow sampling and let the model consider allocations nobody would act on."
            )
        return {
            "ran": True,
            "coverage_90": round(covered, 3),
            "band_width_vs_data_range": round(width_ratio, 2) if width_ratio else None,
            "notes": notes,
        }
    except Exception as e:
        return {"ran": True, "error": str(e)}


def compute_insample_metrics(model: Any, y: pd.Series, spec: MMMSpec) -> dict[str, Any]:
    """R², MAPE and wMAPE computed per posterior draw, then summarised.

    Reported as a mean with a credible interval: a single number hides whether
    the model fits confidently or merely fits on average.
    """
    try:
        pp = model.idata["posterior_predictive"]
        var = spec.target_column if spec.target_column in pp else list(pp.data_vars)[0]
        draws = np.asarray(pp[var])
        scale = float(model.get_scales_as_xarray()["target_scale"])
        obs = y.to_numpy(dtype=float)

        flat = draws.reshape(-1, draws.shape[-1])
        if flat.shape[-1] != len(obs):
            flat = flat.reshape(-1, len(obs))
        preds = flat * scale

        ss_tot = float(np.sum((obs - obs.mean()) ** 2))
        r2 = 1 - np.sum((obs - preds) ** 2, axis=1) / ss_tot if ss_tot > 0 else np.zeros(len(preds))
        nonzero = obs != 0
        mape = np.mean(np.abs((obs[nonzero] - preds[:, nonzero]) / obs[nonzero]), axis=1)
        wmape = np.sum(np.abs(obs - preds), axis=1) / np.sum(np.abs(obs))

        def _summary(a: np.ndarray) -> dict[str, float]:
            return {
                "mean": round(float(np.mean(a)), 4),
                "p05": round(float(np.percentile(a, 5)), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
            }

        return {
            "r2_insample": round(float(np.mean(r2)), 4),
            "r2_insample_hdi": _summary(r2),
            "mape_insample": round(float(np.mean(mape)), 4),
            "mape_insample_hdi": _summary(mape),
            "wmape_insample": round(float(np.mean(wmape)), 4),
            "target_scale": scale,
            "metric_note": (
                "Computed per posterior-predictive draw then averaged. This includes "
                "observation noise, so it is stricter than R2 of the posterior mean — and it "
                "is the honest number, because the mean of many draws cancels noise the model "
                "never actually removed. Compare like with like across runs."
            ),
        }
    except Exception as e:
        return {"r2_insample": None, "mape_insample": None, "metrics_error": str(e)}


_compute_insample_metrics = compute_insample_metrics
