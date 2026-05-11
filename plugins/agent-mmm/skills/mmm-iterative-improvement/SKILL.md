---
name: mmm-iterative-improvement
description: |
  MMM iterative improvement mechanics: tournament-based model selection and posterior-informed prior tightening. Use when designing or running the improvement loop, understanding tournament scoring, debugging why improvement stalled, or explaining the refinement strategy to stakeholders.
---

# Iterative MMM Improvement

## Architecture

The improvement loop has two phases:

### Phase 1: Tournament
Run N variants of the model simultaneously (different hyperparameter configurations), score each, select the winner.

### Phase 2: Posterior-Informed Refinement
Use the winner's posterior distribution to tighten priors, then re-run the tournament from the better starting point. Repeat until score plateaus.

## Variant Generation

Each tournament round generates variants by grid-searching:

| Axis | Values Tried |
|------|-------------|
| `l_max` (adstock window) | e.g., 4, 8, 12 |
| Fourier modes | e.g., 6, 8, 12 |
| Prior width multiplier | e.g., 0.7, 1.0, 1.5 (scale sigma) |

6 variants per round is a good default for weekly data with moderate compute.

## Composite Scoring Function

```python
score = cv_r2 * overfit_penalty * convergence_factor * plausibility_factor

# Components:
cv_r2 = cross_validation_r_squared  # from TimeSliceCrossValidator
overfit_penalty = max(0, 1 - overfit_gap * 2)  # 0 if gap >= 0.5
convergence_factor = 1.0 if converged else 0.5  # binary penalty for non-convergence
plausibility_factor = 1.0 if plausible else 0.8  # soft penalty for implausible attribution
```

Higher score = better. Score of 1.0 is perfect (CV R²=1, no overfit, converged, plausible).

## Plateau Detection

The loop stops early if:
- Score improves by < 0.01 for `patience` consecutive rounds (default patience=2)
- `max_rounds` is reached (default 3)

## Posterior-Informed Prior Tightening

After selecting the round winner, extract posterior statistics and tighten priors:

```python
# For each channel parameter (adstock_alpha, saturation_lam):
new_mu = posterior_mean
new_sigma = posterior_std * tighten_factor  # default 0.7

# Then moment-match back to distribution parameters:
# adstock_alpha: Beta(alpha, beta) via beta_moment_match(new_mu, new_sigma)
# saturation_lam: Gamma(alpha, beta) via gamma_moment_match(new_mu, new_sigma)
```

**Safety check**: If tightened sigma would be < 0.01, floor it at 0.01 to avoid degenerate distributions.

## Leaderboard

All runs are tracked in `mmm-workspace/leaderboard.json`:
```json
{
  "runs": [
    {
      "run_id": "2026-05-10T14-22_v01",
      "round": 1,
      "variant": "l_max=8_fourier=8_width=1.0",
      "score": 0.74,
      "cv_r2": 0.82,
      "overfit_gap": 0.04,
      "converged": true,
      "n_divergences": 0
    }
  ],
  "best_run_id": "2026-05-10T14-22_v01",
  "best_score": 0.74
}
```

## Brownfield Improvement

For brownfield projects, the first tournament round starts with tightened priors from the existing `idata.nc`. This is equivalent to a posterior-informed refinement step from the existing model. Subsequent rounds continue the normal tournament logic.

## Running the Loop

```python
from agent_mmm.iter_loop import run_tournament

run_tournament(
    spec_path="mmm-workspace/spec.yaml",
    model_config_path="mmm-workspace/priors/model_config.json",
    max_rounds=3,
    n_variants_per_round=6,
    patience=2,
    base=".",
)
```

Run via slash command: `/mmm-improve`
