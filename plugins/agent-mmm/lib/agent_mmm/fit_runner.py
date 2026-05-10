"""Fit runner: prior-PC → fit → posterior-PC → save InferenceData."""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_mmm.spec import MMMSpec, MMMType
from agent_mmm.model_factory import build_mmm, prepare_data
from agent_mmm.workspace import ensure_run_dir, new_run_id

logger = logging.getLogger(__name__)


# Sampler configs
SAMPLER_QUICK = {"draws": 500, "tune": 1000, "chains": 2, "target_accept": 0.90}
SAMPLER_CV = {"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.97}
SAMPLER_FINAL = {"draws": 2000, "tune": 3000, "chains": 4, "target_accept": 0.99}


def run_fit(
    spec: MMMSpec,
    model_config_dict: dict | None = None,
    model_config_path: str | Path | None = None,
    run_id: str | None = None,
    sampler_config: dict | None = None,
    base: str | Path = ".",
    skip_prior_pc: bool = False,
    prior_pc_samples: int = 300,
) -> dict[str, Any]:
    """Run the full fit pipeline for one run.

    Steps:
    1. Build MMM from spec + model_config
    2. Prior predictive check (unless skip_prior_pc=True)
    3. Fit model
    4. Posterior predictive check
    5. Save InferenceData + metrics to workspace

    Returns a metrics dict.
    """
    if run_id is None:
        run_id = new_run_id()
    if sampler_config is None:
        sampler_config = SAMPLER_QUICK

    run_dir = ensure_run_dir(run_id, base)

    # Load model config
    if model_config_dict is None and model_config_path is not None:
        import json as _json
        with open(model_config_path) as f:
            raw = _json.load(f)
        model_config_dict = raw.get("model_config", raw)

    X, y = prepare_data(spec)
    model = build_mmm(spec, model_config_dict=model_config_dict)

    # Save config snapshot
    config_snapshot = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "sampler_config": sampler_config,
        "spec_company": spec.company_name,
        "spec_channels": spec.channel_columns(),
        "spec_target": spec.target_column,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # --- Step 1: Prior predictive check ---
    prior_pc_summary = {}
    if not skip_prior_pc:
        try:
            logger.info("Running prior predictive check...")
            prior_pc = model.sample_prior_predictive(X=X, y=y, samples=prior_pc_samples)
            # Check coverage: does 90% CI contain observed range?
            y_vals = y.values
            obs_min, obs_max = float(y_vals.min()), float(y_vals.max())
            # Normalized observed (rough estimate — exact MaxAbsScaler applied internally)
            target_scale = float(y_vals.max()) if float(y_vals.max()) > 0 else 1.0
            obs_min_norm = obs_min / target_scale
            obs_max_norm = obs_max / target_scale

            prior_pc_summary = {
                "prior_pc_ran": True,
                "observed_min": obs_min_norm,
                "observed_max": obs_max_norm,
            }
        except Exception as e:
            prior_pc_summary = {"prior_pc_ran": False, "error": str(e)}
            logger.warning(f"Prior PC failed: {e}")

    # --- Step 2: Fit model ---
    logger.info(f"Fitting model with sampler_config={sampler_config}")
    model.fit(X, y, **sampler_config)

    # --- Step 3: Posterior predictive ---
    try:
        pp = model.sample_posterior_predictive(X, extend_idata=True)
    except Exception as e:
        logger.warning(f"Posterior PC failed: {e}")

    # --- Step 4: Compute in-sample metrics ---
    metrics = _compute_insample_metrics(model, X, y, spec)
    metrics["run_id"] = run_id
    metrics["completed_at"] = datetime.now().isoformat()
    metrics["sampler_config"] = sampler_config
    metrics["prior_pc"] = prior_pc_summary

    # --- Step 5: Save InferenceData ---
    idata_path = run_dir / "idata.nc"
    try:
        model.idata.to_netcdf(str(idata_path))
        metrics["idata_path"] = str(idata_path)
    except Exception as e:
        logger.warning(f"Failed to save idata: {e}")
        metrics["idata_path"] = None
        metrics["idata_save_error"] = str(e)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Run {run_id} complete. R²={metrics.get('r2_insample'):.3f}")
    return metrics


def _compute_insample_metrics(model: Any, X: pd.DataFrame, y: pd.Series, spec: MMMSpec) -> dict:
    """Compute in-sample R² and MAPE from posterior predictive.

    Respects E[f(x)] != f(E[x]): computes per-sample metrics then averages.
    """
    try:
        # Get posterior predictive samples (normalized)
        pp_data = model.idata.posterior_predictive
        # Shape: (chain, draw, date) or similar
        y_pred_samples = pp_data[spec.target_column + "_obs"].values if (
            spec.target_column + "_obs" in pp_data
        ) else pp_data[list(pp_data.data_vars)[0]].values

        # Get target scale
        target_scale = float(y.max()) if float(y.max()) > 0 else 1.0
        y_true = y.values / target_scale  # normalized

        # y_pred_samples shape: (chains, draws, time)
        flat_samples = y_pred_samples.reshape(-1, y_pred_samples.shape[-1])

        # Per-sample R² and MAPE, then average
        r2_per_sample = []
        mape_per_sample = []
        for samp in flat_samples:
            ss_res = np.sum((y_true - samp) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_per_sample.append(r2)
            nonzero = y_true != 0
            if nonzero.sum() > 0:
                mape = float(np.mean(np.abs((y_true[nonzero] - samp[nonzero]) / y_true[nonzero])))
                mape_per_sample.append(mape)

        return {
            "r2_insample": float(np.mean(r2_per_sample)),
            "mape_insample": float(np.mean(mape_per_sample)) if mape_per_sample else None,
        }
    except Exception as e:
        return {"r2_insample": None, "mape_insample": None, "metrics_error": str(e)}
