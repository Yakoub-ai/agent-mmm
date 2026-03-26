---
name: mmm-diagnostics
description: |
  Convergence diagnostics, fit evaluation, and debugging for Marketing Mix Models. Use when checking rhat, ESS, divergences, evaluating model fit metrics (R-squared, MAPE, wMAPE), debugging sampling issues, assessing overfitting, running model comparison (LOO/WAIC), or validating cross-validation results. Also activate when the user encounters convergence warnings, poor model fit, or needs to understand why a model isn't working.
---

# MMM Diagnostics & Debugging

## Convergence Checklist

Run these checks immediately after fitting:

| Check | Threshold | Ideal | How to Check |
|-------|-----------|-------|-------------|
| rhat | < 1.05 | < 1.02 | `az.summary(model.idata)["r_hat"].max()` |
| ESS bulk | > 400 | > 1000 | `az.summary(model.idata)["ess_bulk"].min()` |
| ESS tail | > 400 | > 1000 | `az.summary(model.idata)["ess_tail"].min()` |
| Divergences | 0 | 0 | `model.idata.sample_stats["diverging"].sum()` |
| Energy | No bimodality | Smooth | `az.plot_energy(model.idata)` |

```python
import arviz as az

# Quick convergence summary
summary = az.summary(model.idata)
print(f"Max rhat: {summary['r_hat'].max():.4f}")
print(f"Min ESS bulk: {summary['ess_bulk'].min():.0f}")
print(f"Divergences: {model.idata.sample_stats['diverging'].sum().values}")
```

## Fit Quality Metrics

### The Cardinal Rule
**E[f(x)] != f(E[x])** -- Always compute metrics per posterior sample, then average.

```python
# CORRECT: per-sample metrics
from pymc_marketing.mmm.evaluation import compute_summary_metrics
metrics = compute_summary_metrics(y_true, y_pred_samples)

# WRONG: metrics on posterior mean
y_pred_mean = pp.mean(dim=("chain", "draw"))
r2 = r2_score(y_true, y_pred_mean)  # Biased upward!
```

### Metric Thresholds

| Metric | In-Sample | CV (Out-of-Sample) | Notes |
|--------|-----------|---------------------|-------|
| R² | > 0.80 | > 0.60 | Primary fit quality indicator |
| MAPE | < 10% | < 15% | Percentage error (sensitive to small values) |
| wMAPE | < 10% | < 15% | Weighted MAPE (more robust) |
| NRMSE | < 0.15 | < 0.20 | Normalized root mean squared error |

### Overfit Detection
```python
overfit_gap = train_r2 - val_r2
# Gap > 0.20: likely overfitting
# Gap < 0.05: good generalization
# Gap < 0: underfitting (rare) or data issues
```

## Guardrails

| Metric | Threshold | Action if Violated |
|--------|-----------|-------------------|
| mean_val_r2 | > 0.60 | Model underfitting or poor generalization |
| rhat_max | < 1.05 | Chains haven't converged -- increase tune/target_accept |
| max_overfit_gap | < 0.20 | Train-test gap too large -- regularize or simplify |
| min_ess_bulk | > 400 | Insufficient effective samples -- increase draws |

## Validation Tiers (Increasing Rigor)

1. **In-sample R² > 0.80** -- basic fit quality
2. **LOO/WAIC** -- information-theoretic model comparison (lower = better)
3. **TSCV CV R² > 0.60** -- out-of-sample generalization with expanding window
4. **Contribution sanity** -- shares sum to ~1.0, magnitudes and signs plausible
5. **Prior dominance check** -- `posterior_std / prior_std < 0.8` means data informed beliefs

### LOO/WAIC Comparison
```python
import arviz as az
loo = az.loo(model.idata)
waic = az.waic(model.idata)
print(f"LOO: {loo.loo:.1f} (se: {loo.loo_se:.1f})")
print(f"WAIC: {waic.waic:.1f} (se: {waic.waic_se:.1f})")

# Compare models
az.compare({"model_a": idata_a, "model_b": idata_b})
```

### Prior vs Posterior Check
```python
# Posterior should be narrower than prior (data updated beliefs)
model.plot.prior_vs_posterior(var="adstock_alpha")
model.plot.prior_vs_posterior(var="saturation_lam")

# Quantitative check
prior_std = model.idata.prior["adstock_alpha"].std(dim=("chain", "draw"))
posterior_std = model.idata.posterior["adstock_alpha"].std(dim=("chain", "draw"))
ratio = posterior_std / prior_std
# ratio < 0.8: data informed (good)
# ratio > 0.95: prior dominated (bad -- prior too strong or insufficient data)
```

## Debugging Decision Tree

```
Model won't converge?
├── Divergences > 0?
│   ├── Increase target_accept → 0.99
│   ├── Still diverging? → Check priors (too wide/contradictory?)
│   └── Still diverging? → Reduce model complexity
├── rhat > 1.05?
│   ├── Increase tune steps (3000+)
│   ├── Check for multimodality (az.plot_trace)
│   └── Consider reparameterization
├── ESS < 400?
│   ├── Increase draws (2000+)
│   ├── Increase tune steps
│   └── Check for high autocorrelation
└── All chains stuck?
    ├── Check data for extreme values
    ├── Check prior-data conflict
    └── Try different initialization

Model fits but poor R²?
├── In-sample R² < 0.60?
│   ├── Missing important channels or controls
│   ├── Wrong target variable
│   ├── Structural break in data
│   └── Need time-varying intercept
├── CV R² << In-sample R²?
│   ├── Overfitting: too many parameters
│   ├── Tighten priors (more informative)
│   ├── Reduce channels
│   └── Check for data leakage in CV setup
└── Contribution shares implausible?
    ├── Prior specification wrong
    ├── Collinearity between channels
    ├── Insufficient data for channel
    └── Wrong adstock/saturation choice
```

## Diagnostic Visualizations

```python
# Core diagnostics
model.plot.posterior_predictive()          # In-sample fit
model.plot.residuals_over_time()           # Temporal patterns in residuals
model.plot.prior_vs_posterior(var="adstock_alpha")  # Data learning

# ArviZ diagnostics
az.plot_trace(model.idata, var_names=["adstock_alpha", "saturation_lam"])
az.plot_energy(model.idata)                # Sampler health
az.plot_posterior(model.idata, var_names=["adstock_alpha"])
az.plot_pair(model.idata, var_names=["adstock_alpha", "saturation_lam"])  # Correlations

# Residual analysis
model.plot.residuals_posterior_distribution()  # Should be ~Normal
```

## Cross-Validation Diagnostics

```python
# Per-fold metrics
for i, result in enumerate(cv._cv_results):
    pp = result.idata.posterior_predictive
    y_pred = pp["y"].mean(dim=("chain", "draw")).values
    r2 = r2_score(result.y_test, y_pred)
    print(f"Fold {i+1}: R²={r2:.3f}")

# Plot CV predictions
model.plot.cv_predictions()
model.plot.cv_crps()          # Continuous Ranked Probability Score
model.plot.param_stability()  # Parameter stability across folds
```

## Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Divergences | Warning during sampling | Increase target_accept to 0.99 |
| Low ESS | < 400 effective samples | Increase draws to 2000+ |
| High rhat | > 1.05 | Increase tune to 3000+ |
| Prior dominance | Posterior = Prior | Widen priors or get more data |
| Overfitting | Train R² >> CV R² by 0.20+ | Tighten priors, reduce channels |
| Poor fit | R² < 0.60 | Add controls, check target, add channels |
| Bimodal posterior | Multiple peaks in trace | Check for model identifiability issues |
| Slow sampling | Hours to converge | Reduce l_max, fewer channels, simpler model |
