---
name: mmm-api-reference
description: |
  Complete pymc-marketing API reference for writing and reviewing MMM code. Use when you need exact constructor signatures, method names, parameter lists, or code patterns for pymc-marketing v0.18.2+. Load this skill when writing new model code, debugging import errors, checking method signatures, or reviewing code that uses pymc-marketing. Also activate when you need to look up plotting methods, evaluation functions, cross-validation setup, or transformation classes.
---

# pymc-marketing MMM API Reference (v0.18.2+)

## 1. Imports

```python
# Core MMM class (ALWAYS use multidimensional)
from pymc_marketing.mmm.multidimensional import MMM

# Transformations
from pymc_marketing.mmm import (
    GeometricAdstock,       # Exponential decay
    DelayedAdstock,         # Weibull CDF peak-then-decay
    WeibullAdstock,         # Weibull parameterization
    NoAdstock,              # No carryover
    LogisticSaturation,     # 1 - exp(-lam * x)
    HillSaturation,         # S-curve with inflection
    MichaelisMentenSaturation,
    TanhSaturation,
    NoSaturation,
)

# Priors
from pymc_extras.prior import Prior

# Budget optimization
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer, CustomModelWrapper, Constraint

# Cross-validation
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

# Evaluation
from pymc_marketing.mmm.evaluation import (
    compute_summary_metrics,
    calculate_metric_distributions,
    summarize_metric_distributions,
)

# Summary
from pymc_marketing.mmm.summary import MMMSummaryFactory, MMMIDataWrapper

# Diagnostics
import arviz as az
```

## 2. MMM Constructor

```python
model = MMM(
    date_column: str,                     # Required: datetime column name
    channel_columns: list[str],           # Required: media spend column names
    adstock: AdstockTransformation,       # Required: carryover transform
    saturation: SaturationTransformation, # Required: diminishing returns transform
    target_column: str = "y",             # Name of target column (default "y")
    control_columns: list[str] = None,    # Additional regressors
    yearly_seasonality: int = None,       # Fourier modes for yearly patterns
    time_varying_intercept: bool = False, # HSGP dynamic baseline
    time_varying_media: bool = False,     # HSGP time-dependent channel effect
    model_config: dict = None,            # Prior and likelihood config
    sampler_config: dict = None,          # PyMC sampling parameters
    adstock_first: bool = True,           # Adstock before saturation (recommended)
    scaling: Scaling|dict = None,         # Data scaling config
    dag: str = None,                      # DOT format DAG for causal ID
    treatment_nodes: list[str] = None,    # Causal treatment variables
    outcome_node: str = None,             # Causal outcome variable
    dims: dict = None,                    # Additional model dimensions
)
```

### Default model_config
```python
{
    "intercept": Prior("Normal", mu=0, sigma=2),
    "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
    "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
    "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "adstock_alpha": Prior("Beta", alpha=1, beta=3),
    "saturation_lam": Prior("Gamma", alpha=3, beta=1),
    "saturation_beta": Prior("HalfNormal", sigma=2),
}
```

## 3. Transformation Classes

### GeometricAdstock
```python
adstock = GeometricAdstock(l_max=5)
# alpha in [0,1]: higher = longer memory
# w_t = alpha^t (exponential decay)
```

### DelayedAdstock
```python
adstock = DelayedAdstock(l_max=12)
# Weibull CDF: w_t = 1 - exp(-(t/lam)^k)
# Peak-then-decay pattern, good for offline media
```

### LogisticSaturation
```python
saturation = LogisticSaturation()
# f(x) = 1 - exp(-lam * x)
# lam: steepness, beta: output scale
```

## 4. Fitting Methods

```python
# Prior predictive check
prior = model.sample_prior_predictive(X=X, y=y, samples=500)

# Fit model
model.fit(X, y, draws=2000, tune=3000, chains=4, target_accept=0.99)

# Posterior predictive (in-sample)
pp = model.sample_posterior_predictive(X, extend_idata=True)
# WARNING: Returns NORMALIZED [0,1] values

# Out-of-sample prediction
pp_test = model.sample_posterior_predictive(X_test, extend_idata=False)
y_pred = pp_test.mean(dim=("chain", "draw")).values

# Get scale factor
scale = model.get_scales_as_xarray()["target_scale"]
```

## 5. Attribution Methods

```python
# Channel contributions (original scale)
model.add_original_scale_contribution_variable(["channel_contribution"])
contrib = model.idata.posterior["channel_contribution_original_scale"]
# dims: (chain, draw, date, channel)

# Model summary
model.table()           # Rich table
summary = model.summary  # Property

# MMMSummaryFactory
wrapper = MMMIDataWrapper(model.idata)
factory = MMMSummaryFactory(data=wrapper, model=model)
factory.contributions()
factory.roas()
factory.saturation_curves()
factory.adstock_curves()
factory.total_contribution()
factory.channel_spend()
```

## 6. Curve Sampling

```python
# Saturation curves with posterior uncertainty
sat_curve = model.sample_saturation_curve(max_value=2.0, num_points=100)
# Returns xarray: dims (chain, draw, x, channel)

# Adstock curves
ads_curve = model.sample_adstock_curve()
```

## 7. Cross-Validation

```python
cv = TimeSliceCrossValidator(
    n_init=80,              # Min training observations
    forecast_horizon=13,    # Weeks to predict
    date_column="date",
    step_size=13,           # Step between folds
    sampler_config={"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.97},
)

# Requires MMMBuilder protocol
class ModelBuilder:
    def build_model(self, X, y):
        model = create_and_fit_model(X, y)
        return model

n_folds = cv.get_n_splits(X, pd.Series(y))
combined_idata = cv.run(X, pd.Series(y), mmm=ModelBuilder())

# Per-fold results
for result in cv._cv_results:
    result.y_train, result.y_test
    result.idata.posterior_predictive
```

## 8. Budget Optimization

```python
# Wrap fitted model
wrapper = CustomModelWrapper(
    base_model=model.model,
    idata=model.idata,
    channels=channel_columns,
)

# Create optimizer
optimizer = BudgetOptimizer(model=wrapper, num_periods=len(X))

# Run optimization
opt_alloc, opt_result = optimizer.allocate_budget(
    total_budget=total_budget,
    budget_bounds=budget_bounds,  # dict: {channel: (min, max)}
)

# Custom constraints
optimizer.set_constraints([
    Constraint(type="ineq", fun=constraint_fn),
])
```

## 9. Evaluation

```python
# Per-sample metric distributions (correct approach)
dist = calculate_metric_distributions(y_true, y_pred_samples)
summary = summarize_metric_distributions(dist)
metrics = compute_summary_metrics(y_true, y_pred_samples)

# sklearn point metrics (on posterior mean -- less correct)
from sklearn.metrics import r2_score, mean_absolute_percentage_error
```

## 10. Plotting Suite (model.plot.*)

```python
# Predictive checks
model.plot.posterior_predictive()
model.plot.prior_predictive()

# Parameter diagnostics
model.plot.prior_vs_posterior(var="adstock_alpha")
model.plot.posterior_distribution(var="saturation_beta")
model.plot.channel_parameter(var="adstock_alpha")

# Decomposition
model.plot.waterfall_components_decomposition()
model.plot.contributions_over_time()
model.plot.channel_contribution_share_hdi()

# Residuals
model.plot.residuals_over_time()
model.plot.residuals_posterior_distribution()

# Curves
model.plot.saturation_curves(curve=model.sample_saturation_curve())
model.plot.saturation_curves_scatter()
model.plot.saturation_scatterplot()
model.plot.marginal_curve()
model.plot.uplift_curve()

# Cross-validation
model.plot.cv_crps()
model.plot.cv_predictions()
model.plot.param_stability()

# Budget
model.plot.budget_allocation()
model.plot.allocated_contribution_by_channel_over_time()
model.plot.sensitivity_analysis()

# Interactive (Plotly)
model.plot_interactive.posterior_predictive()
```

## 11. ArviZ Diagnostics

```python
import arviz as az

# Convergence
az.summary(model.idata)                    # rhat, ESS, mean, HDI
az.plot_trace(model.idata, var_names=[...]) # Trace plots
az.plot_energy(model.idata)                 # NUTS energy diagnostic
az.plot_posterior(model.idata, var_names=[...])
az.plot_pair(model.idata, var_names=[...])  # Parameter correlations

# Model comparison
loo = az.loo(model.idata)
waic = az.waic(model.idata)
az.compare({"model_a": idata_a, "model_b": idata_b})
```

## 12. Migration from Legacy API

| Feature | Legacy `pymc_marketing.mmm.MMM` | Multidimensional |
|---------|------|------|
| Import | `from pymc_marketing.mmm import MMM` | `from pymc_marketing.mmm.multidimensional import MMM` |
| Target column | Not a constructor param | `target_column="y"` |
| Channel priors | `adstock.function_priors["alpha"] = Prior(...)` | `model_config={"adstock_alpha": Prior(...)}` |
| Contributions | `model.compute_channel_contribution_original_scale()` | `model.add_original_scale_contribution_variable(...)` |
| Shares | `model.get_channel_contribution_share_samples()` | `contrib / contrib.sum("channel")` |
| Budget opt | `model.optimize_budget()` | `BudgetOptimizer` + `CustomModelWrapper` |
| Plots | `model.plot_posterior_predictive()` | `model.plot.posterior_predictive()` |
| Params | `model.format_recovered_transformation_parameters()` | `model.table()` |
| Forward pass | `model.get_channel_contribution_forward_pass_grid()` | `model.sample_saturation_curve()` |

## 13. Sensitivity Analysis

```python
results = model.sensitivity.run_sweep()
```
