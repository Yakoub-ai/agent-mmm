"""Iterative improvement loop: tournament + posterior-informed prior refinement."""
from __future__ import annotations
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agent_mmm.workspace import ensure_workspace, new_run_id

logger = logging.getLogger(__name__)

# Variant parameters for tournament
TOURNAMENT_L_MAX_OPTIONS = [5, 8, 12]
TOURNAMENT_FOURIER_OPTIONS = [6, 8, 12]
TOURNAMENT_PRIOR_WIDTH_MULTIPLIERS = [0.8, 1.0, 1.2]

# Scoring constants
CONVERGENCE_PENALTY = 0.7   # multiplier when convergence fails
PLAUSIBILITY_PENALTY = 0.8  # multiplier when attribution implausible

# Improvement plateau threshold
PLATEAU_DELTA = 0.01


def score_run(metrics: dict, diagnostics: dict) -> float:
    """Compute composite score for a run.

    score = cv_r2 * (1 - overfit_gap_penalty) * convergence_ok * plausibility_ok

    All factors in [0, 1]. Higher = better model.
    """
    # CV R² (primary signal)
    ov = diagnostics.get("checks", {}).get("overfit", {})
    cv_r2 = ov.get("cv_r2") if ov.get("available") else None
    in_sample_r2 = ov.get("in_sample_r2") or metrics.get("r2_insample") or 0.0

    if cv_r2 is None:
        # No CV available — use penalized in-sample R²
        cv_r2 = in_sample_r2 * 0.85  # conservative estimate

    cv_r2 = max(0.0, float(cv_r2))

    # Overfit gap penalty
    overfit_gap = max(0.0, float(ov.get("gap") or 0.0))
    overfit_penalty = max(0.0, 1.0 - overfit_gap * 2)  # 0.20 gap → 0.60 penalty

    # Convergence factor
    summary = diagnostics.get("summary", {})
    convergence_factor = 1.0
    if not summary.get("rhat_ok", True) or not summary.get("divergences_ok", True):
        convergence_factor = CONVERGENCE_PENALTY

    # Plausibility factor
    plausibility = diagnostics.get("checks", {}).get("attribution_plausibility", {})
    plausibility_factor = 1.0
    if plausibility.get("dominant_channel"):
        plausibility_factor = PLAUSIBILITY_PENALTY

    score = cv_r2 * overfit_penalty * convergence_factor * plausibility_factor
    return round(score, 4)


def generate_variants(
    base_model_config: dict,
    n_variants: int = 6,
    round_number: int = 1,
) -> list[dict]:
    """Generate N candidate model config variants for the tournament.

    Varies: l_max, yearly_fourier_modes, and prior_width_multiplier.
    In round 1: explores the full space. In later rounds: focuses around the current best.
    """
    import itertools

    # Build a grid and sample from it
    grid = list(itertools.product(
        TOURNAMENT_L_MAX_OPTIONS,
        TOURNAMENT_FOURIER_OPTIONS,
        TOURNAMENT_PRIOR_WIDTH_MULTIPLIERS,
    ))

    rng = np.random.default_rng(round_number * 100)
    selected = rng.choice(len(grid), size=min(n_variants, len(grid)), replace=False)

    variants = []
    for i, idx in enumerate(selected):
        l_max, fourier, width_mult = grid[idx]
        cfg = copy.deepcopy(base_model_config)
        cfg["_variant"] = {
            "l_max": l_max,
            "yearly_fourier_modes": fourier,
            "prior_width_multiplier": width_mult,
            "variant_id": f"r{round_number}_v{i+1}",
        }
        # Apply width multiplier to all sigma/alpha/beta arrays in prior dists
        for key in ["adstock_alpha", "saturation_lam", "saturation_beta"]:
            if key in cfg:
                entry = cfg[key]
                for sigma_key in ["sigma", "alpha", "beta"]:
                    if sigma_key in entry and isinstance(entry[sigma_key], list):
                        entry[sigma_key] = [v * width_mult for v in entry[sigma_key]]
        variants.append(cfg)

    return variants


def tighten_priors_from_posterior(
    model_config: dict,
    posterior_stats: dict,
    tighten_factor: float = 0.7,
) -> dict:
    """Update prior mu/sigma from posterior mean/std.

    For each channel parameter in model_config:
    - New mu ≈ posterior mean (moment-match back to prior)
    - New sigma ≈ posterior std * tighten_factor (tighter but not overly tight)

    Uses moment-matching to convert (mu, sigma) back to distribution params.
    """
    from agent_mmm.utils.moment_match import beta_moment_match, gamma_moment_match

    new_config = copy.deepcopy(model_config)

    for param_key in ["adstock_alpha", "saturation_lam"]:
        if param_key not in posterior_stats or param_key not in new_config:
            continue

        post = posterior_stats[param_key]  # {"mean": [...], "std": [...]}
        means = post.get("mean", [])
        stds = post.get("std", [])

        if not means or not stds:
            continue

        a_list, b_list = [], []
        for mu, sigma in zip(means, stds):
            new_sigma = max(sigma * tighten_factor, 0.01)
            try:
                if "alpha" in param_key:
                    a, b = beta_moment_match(float(mu), float(new_sigma))
                else:
                    a, b = gamma_moment_match(float(mu), float(new_sigma))
                a_list.append(round(a, 4))
                b_list.append(round(b, 4))
            except (ValueError, ZeroDivisionError):
                # Keep original if moment-match fails
                entry = new_config[param_key]
                a_list.append(entry.get("alpha", [2.0])[len(a_list)] if isinstance(entry.get("alpha"), list) else 2.0)
                b_list.append(entry.get("beta", [4.0])[len(b_list)] if isinstance(entry.get("beta"), list) else 4.0)

        if a_list:
            new_config[param_key]["alpha"] = a_list
            new_config[param_key]["beta"] = b_list

    return new_config


def extract_posterior_stats(idata_path: str | Path) -> dict:
    """Extract per-channel posterior mean and std from InferenceData for prior tightening."""
    try:
        import arviz as az
        import numpy as np
        idata = az.from_netcdf(str(idata_path))
        posterior = idata.posterior
        stats = {}
        for var in ["adstock_alpha", "saturation_lam"]:
            if var in posterior.data_vars:
                arr = posterior[var].values  # (chains, draws, channels)
                flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1, 1)
                stats[var] = {
                    "mean": np.mean(flat, axis=0).tolist(),
                    "std": np.std(flat, axis=0).tolist(),
                }
        return stats
    except Exception as e:
        logger.warning(f"Failed to extract posterior stats: {e}")
        return {}


def run_tournament(
    spec_path: str | Path,
    model_config_path: str | Path,
    max_rounds: int = 3,
    n_variants_per_round: int = 6,
    patience: int = 2,
    base: str | Path = ".",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full tournament + posterior-informed refinement loop.

    Parameters
    ----------
    spec_path: path to spec.yaml
    model_config_path: path to model_config.json from prior_engine
    max_rounds: maximum number of improvement rounds
    n_variants_per_round: number of model variants per round
    patience: stop if no improvement for this many rounds
    base: workspace base directory
    dry_run: if True, generate variants but don't actually fit (for testing)

    Returns leaderboard dict.
    """
    from agent_mmm.spec import load_spec

    spec = load_spec(spec_path)

    with open(model_config_path) as f:
        raw = json.load(f)
    base_model_config = raw.get("model_config", raw)

    ws = ensure_workspace(base)
    leaderboard: list[dict] = []
    best_score = 0.0
    no_improvement_rounds = 0
    current_config = base_model_config

    result = {
        "started_at": datetime.now().isoformat(),
        "max_rounds": max_rounds,
        "n_variants_per_round": n_variants_per_round,
        "rounds": [],
        "best_run_id": None,
        "best_score": 0.0,
        "leaderboard": [],
    }

    for round_num in range(1, max_rounds + 1):
        logger.info(f"Tournament round {round_num}/{max_rounds}")

        variants = generate_variants(current_config, n_variants=n_variants_per_round, round_number=round_num)
        round_results = []

        for i, variant_config in enumerate(variants):
            variant_info = variant_config.pop("_variant", {})
            variant_id = variant_info.get("variant_id", f"r{round_num}_v{i}")
            run_id = f"{new_run_id()}_tour_{variant_id}"

            if dry_run:
                # Synthetic metrics for testing
                rng = np.random.default_rng(round_num * 10 + i)
                metrics = {
                    "run_id": run_id,
                    "r2_insample": float(rng.uniform(0.65, 0.92)),
                    "mape_insample": float(rng.uniform(0.08, 0.20)),
                }
                diagnostics = {
                    "summary": {"tier": "PASS", "rhat_ok": True, "ess_ok": True, "divergences_ok": True},
                    "checks": {
                        "overfit": {
                            "available": True,
                            "in_sample_r2": metrics["r2_insample"],
                            "cv_r2": metrics["r2_insample"] - float(rng.uniform(0.05, 0.18)),
                            "gap": float(rng.uniform(0.05, 0.18)),
                            "overfit": False,
                        },
                        "attribution_plausibility": {"available": False},
                    },
                }
            else:
                # Real fit
                try:
                    from agent_mmm.fit_runner import run_fit, SAMPLER_QUICK
                    from agent_mmm.diagnostics import run_diagnostics

                    metrics = run_fit(
                        spec,
                        model_config_dict=variant_config,
                        run_id=run_id,
                        sampler_config=SAMPLER_QUICK,
                        base=base,
                    )
                    diagnostics = run_diagnostics(
                        run_id=run_id,
                        idata_path=metrics.get("idata_path"),
                        metrics_path=str(ws / "runs" / run_id / "metrics.json"),
                        base=base,
                    )
                except Exception as e:
                    logger.warning(f"Variant {variant_id} failed: {e}")
                    metrics = {"run_id": run_id, "r2_insample": 0.0}
                    diagnostics = {"summary": {"tier": "FAIL"}, "checks": {}}

            s = score_run(metrics, diagnostics)
            entry = {
                "run_id": run_id,
                "round": round_num,
                "variant_info": variant_info,
                "score": s,
                "r2_insample": metrics.get("r2_insample"),
                "r2_cv": diagnostics.get("checks", {}).get("overfit", {}).get("cv_r2"),
                "overfit_gap": diagnostics.get("checks", {}).get("overfit", {}).get("gap"),
                "tier": diagnostics.get("summary", {}).get("tier"),
            }
            round_results.append(entry)
            leaderboard.append(entry)
            logger.info(f"  {variant_id}: score={s:.3f}")

        # Sort round by score, pick winner
        round_results.sort(key=lambda x: x["score"], reverse=True)
        winner = round_results[0]
        result["rounds"].append({
            "round": round_num,
            "winner_run_id": winner["run_id"],
            "winner_score": winner["score"],
            "all_variants": round_results,
        })

        # Check improvement
        if winner["score"] > best_score + PLATEAU_DELTA:
            best_score = winner["score"]
            result["best_run_id"] = winner["run_id"]
            result["best_score"] = best_score
            no_improvement_rounds = 0

            # Tighten priors from winner's posterior for next round
            if not dry_run and winner.get("run_id"):
                idata_path = ws / "runs" / winner["run_id"] / "idata.nc"
                if idata_path.exists():
                    post_stats = extract_posterior_stats(idata_path)
                    current_config = tighten_priors_from_posterior(current_config, post_stats)
        else:
            no_improvement_rounds += 1
            logger.info(f"No improvement (patience {no_improvement_rounds}/{patience})")

        if no_improvement_rounds >= patience:
            logger.info("Plateau reached — stopping tournament")
            result["stopped_reason"] = "plateau"
            break

    # Sort leaderboard
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    result["leaderboard"] = leaderboard
    result["completed_at"] = datetime.now().isoformat()

    # Save leaderboard
    with open(ws / "leaderboard.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result
