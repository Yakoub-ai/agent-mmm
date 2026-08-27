---
name: mmm-api-reference
description: |
  Exact pymc-marketing API reference for MMM code, verified against v1.1.0. Use when writing or reviewing pymc-marketing code, checking constructor signatures, method names, return types, import paths, or plotting/evaluation/optimisation calls. Also use when migrating 0.x code to 1.x, debugging ImportError or AttributeError from pymc-marketing, or working with ArviZ 1.x DataTree inference data.
---

# pymc-marketing API Reference (verified against v1.1.0)

**1.0 was a breaking release.** If you learned this library before August 2026, most
of what you remember about imports, inference data and budget optimisation is wrong.
Check §1 before writing a line.

Environment as of v1.1.0: Python >= 3.12, PyMC 6.2+, PyTensor 3.2+, ArviZ 1.2+,
NumPy 2.x, pymc-extras 0.14+.

---

## 1. What changed in 1.0 (read this first)

| Concern | 0.x | 1.x |
|---|---|---|
| Import | `from pymc_marketing.mmm.multidimensional import MMM` | `from pymc_marketing.mmm import MMM` |
| Legacy class | `pymc_marketing.mmm.MMM` (old single-dim) | **removed** |
| `multidimensional` module | canonical | **deprecated shim**, emits `FutureWarning` |
| Inference data | `arviz.InferenceData` | `xarray.DataTree` |
| Budget optimiser wrapper | `MultiDimensionalBudgetOptimizerWrapper` | `BudgetOptimizerWrapper`, or just `mmm.budget_optimizer(...)` |
| `allocate_budget` return | tuple | `BudgetOptimizationResult` (changed again in 1.1) |
| `az.waic` | available | **removed from ArviZ** — use `az.loo` |
| `az.summary(hdi_prob=)` | `hdi_prob` | `ci_prob`; default interval is ETI-89, columns `eti89_lb`/`eti89_ub` |
| Model deprecated prior module | `pymc_marketing.prior` | removed — use `pymc_extras.prior` |

```python
# CORRECT for 1.x
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

# WRONG — deprecated, warns, and will be removed
from pymc_marketing.mmm.multidimensional import MMM
```

---

## 2. Top-level exports

`from pymc_marketing.mmm import ...`

**Core** — `MMM`, `MMMBuilder`, `MMMPlotSuiteFacade`

**Adstock** — `GeometricAdstock`, `DelayedAdstock`, `WeibullCDFAdstock`,
`WeibullPDFAdstock`, `BinomialAdstock`, `NoAdstock`, `AdstockTransformation`

**Saturation** — `LogisticSaturation`, `InverseScaledLogisticSaturation`,
`HillSaturation`, `HillSaturationSigmoid`, `MichaelisMentenSaturation`,
`TanhSaturation`, `TanhSaturationBaselined`, `RootSaturation`, `LogSaturation`,
`NoSaturation`, `SaturationTransformation`

**Per-channel structure** — `MediaTransformation`, `MediaConfig`, `MediaConfigList`

**Time components** — `LinearTrend`, `YearlyFourier`, `MonthlyFourier`,
`WeeklyFourier`, `HSGP`, `HSGPPeriodic`, `SoftPlusHSGP`, `CovFunc`,
`PeriodicCovFunc`, `approx_hsgp_hyperparams`

**Scaling** — `Scaling`, `FixedScaling`, `DataDerivedScaling`, `VariableScaling`

**Effects** — `MediaMuEffect`, `ControlMuEffect`, `DataVarMuEffect`

**Analysis** — `Incrementality`, `IncrementalitySpec`, `SensitivityAnalysis`,
`TimeSliceCrossValidator`, `TimeSliceCrossValidationResult`

**Optimisation** — `BudgetOptimizer`, `BudgetOptimizerWrapper`,
`BudgetOptimizationResult`, `OptimizationVariable`, `OptimizationVariables`

**Utilities** — `merge_inference_data`, `merge_models_and_idata`,
`FancyLinearRegression`

Submodules worth knowing: `causal`, `counterfactual`, `decomposition`, `events`,
`incrementality`, `lift_test`, `spend_reach`, `builders`, `summary`, `evaluation`,
`time_slice_cross_validation`.

---

## 3. `MMM` constructor

Keyword-only. Every argument below is verified against the 1.1.0 signature.

```python
MMM(
    *,
    date_column: str,                        # required
    channel_columns: list[str],              # required, min length 1
    adstock: AdstockTransformation,          # required
    saturation: SaturationTransformation,    # required
    target_column: str = "y",
    time_varying_intercept: bool | HSGPBase = False,
    time_varying_media: bool | HSGPBase = False,
    dims: tuple[str, ...] | None = None,     # TUPLE of extra panel dims
    scaling: Scaling | dict | None = None,
    model_config: dict | None = None,
    sampler_config: dict | None = None,
    control_columns: list[str] | None = None,
    yearly_seasonality: int | None = None,
    adstock_first: bool = True,
    dag: str | None = None,                  # DOT-format causal graph
    treatment_nodes: list[str] | None = None,
    outcome_node: str | None = None,
    link: Literal["identity", "log"] = "identity",
    cost_per_unit: pd.DataFrame | None = None,
)
```

Three arguments people get wrong:

* **`dims` is a tuple.** `dims="geo"` iterates the string into `('g','e','o')`.
  Write `dims=("geo",)`.
* **`link="log"`** switches to a multiplicative model with a LogNormal likelihood.
  Components then combine on the log scale, so a single component has no standalone
  additive contribution — use `compute_counterfactual_contributions_dataset()`, not
  `channel_contribution * target_scale`.
* **`cost_per_unit`** converts non-spend channel inputs (impressions, GRPs) into money
  so ROAS and the optimiser stay in currency. Wide DataFrame indexed by
  `(date, *dims)`, columns are channel names; missing channels default to 1.0.

### Default `model_config`

```python
{
  "intercept":      Prior("Normal", mu=0, sigma=2, dims=()),
  "likelihood":     Prior("Normal", sigma=Prior("HalfNormal", sigma=2), dims="date"),
  "gamma_control":  Prior("Normal", mu=0, sigma=2, dims="control"),
  "gamma_fourier":  Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
  "adstock_alpha":  Prior("Beta", alpha=1, beta=3, dims="channel"),
  "saturation_lam": Prior("Gamma", alpha=3, beta=1, dims="channel"),
  "saturation_beta":Prior("HalfNormal", sigma=2, dims="channel"),
}
```

`default_sampler_config` is `{}`.

---

## 4. Transformations

### Adstock

```python
GeometricAdstock(
    l_max: int,                    # required, > 0
    normalize: bool = True,
    mode: ConvMode = ConvMode.After,
    priors: dict | None = None,
    prefix: str | None = None,
    parametrization: Literal["alpha", "halflife"] | None = None,
)
```

| Class | Parameters | Shape | Use for |
|---|---|---|---|
| `GeometricAdstock` | `alpha` ~ Beta(1,3) | Monotone decay from t=0 | Digital, search, social |
| `DelayedAdstock` | `alpha`, `theta` ~ HalfNormal(1) | Peak at lag `theta`, then decay | TV, print, direct mail, cinema |
| `WeibullCDFAdstock` | `lam`, `k` | Flexible S-shaped accumulation | When the shape is genuinely unknown |
| `WeibullPDFAdstock` | `lam` ~ Gamma(mu=2,σ=1), `k` ~ Gamma(mu=3,σ=1) | Flexible peak-and-decay | Same, with a delayed peak |
| `BinomialAdstock` | `alpha` | Bounded-support decay | Meridian parity |
| `NoAdstock` | — | Instant, no carryover | Price, distribution, promotions |

**Half-life parameterisation (new in 1.1)** — `parametrization="halflife"` reparameterises
`GeometricAdstock`/`DelayedAdstock` so the sampled parameter is the half-life in periods
rather than a decay rate. Prefer it when a stakeholder has to read the posterior: "half
the TV effect is gone after 3.4 weeks" is a claim a marketer can challenge; `alpha=0.82`
is not. The relationship is `alpha = 0.5 ** (1 / halflife)`.

**Choosing `l_max`** — set it so the truncated tail is negligible:
`l_max >= ceil(log(0.05) / log(alpha))` keeps 95% of the impulse. Too small and the
decay parameter is biased downwards, because the model cannot express the tail it
truncated.

### Saturation

| Class | Parameters | Shape |
|---|---|---|
| `LogisticSaturation` | `lam`, `beta` | `beta * (1 - exp(-lam*x))`, concave from zero |
| `InverseScaledLogisticSaturation` | `lam`, `beta` | Same curve, `lam` is the **half-saturation point** — far easier to set a prior on |
| `HillSaturation` | `slope`, `kappa`, `beta` | S-curve; `slope > 1` gives a threshold |
| `HillSaturationSigmoid` | `sigma`, `beta`, `lam` | Sigmoid parameterisation |
| `MichaelisMentenSaturation` | `alpha`, `lam` | `alpha*x/(lam+x)`; `alpha` = asymptote |
| `TanhSaturationBaselined` | `x0`, `gain`, `r`, `beta` | Anchored at a reference spend |
| `RootSaturation` | `alpha`, `beta` | Power curve |
| `LogSaturation` | `beta` | `beta*log(1+x)`, never saturates fully |
| `NoSaturation` | — | Linear |

Concave-from-zero (Logistic, Michaelis-Menten) says the first pound is the most
productive. S-shaped (Hill with `slope > 1`) says there is a minimum effective dose
below which nothing happens. That is a real phenomenon for broad-reach media, but it
is very hard to identify without spend near zero — do not choose Hill for its
flexibility, choose it because you have data in the threshold region.

### Per-channel transformations

`MMM(adstock=..., saturation=...)` applies **one** pair to every channel. For genuinely
different behaviour per channel:

```python
from pymc_marketing.mmm import (
    MediaTransformation, MediaConfig, MediaConfigList,
    GeometricAdstock, DelayedAdstock, LogisticSaturation,
)

online = MediaTransformation(
    adstock=GeometricAdstock(l_max=6), saturation=LogisticSaturation(),
    adstock_first=True, dims="channel",
)
offline = MediaTransformation(
    adstock=DelayedAdstock(l_max=16), saturation=LogisticSaturation(),
    adstock_first=True, dims="channel",
)
media = MediaConfigList([
    MediaConfig(name="online",  columns=["sem", "social"], media_transformation=online),
    MediaConfig(name="offline", columns=["tv", "ooh"],     media_transformation=offline),
])
```

This changes variable names (they take the config `name` as a prefix), so downstream
helpers keyed on `adstock_alpha` need updating. Weigh that against a shared `l_max` set
to the widest requirement, which costs compute but not correctness.

---

## 5. Fitting

```python
mmm.build_model(X, y)                      # explicit build, needed before lift tests
mmm.sample_prior_predictive(X=X, y=y, samples=500, extend_idata=True, combined=True)

idata = mmm.fit(
    X, y,
    method="mcmc",            # mcmc | map | demz | advi | fullrank_advi
    progressbar=None,
    random_seed=42,
    sample_kwargs=None,       # variational methods only
    draws=2000, tune=3000, chains=4, target_accept=0.95,   # -> pm.sample
)                              # returns xr.DataTree
```

* `X` must contain the date column, every channel column, every control column, and
  one categorical column per entry in `dims`.
* **`y` must be a Series named exactly `target_column`.** A mismatch fails deep inside
  model building with an unhelpful error.
* `method="map"` gives a point estimate in seconds — useful for a smoke test of the
  pipeline, useless for a decision, because it has no uncertainty.
* `approximate_fit(...)` exists for variational fitting on large panels.

### Posterior predictive

```python
pp = mmm.sample_posterior_predictive(
    X, extend_idata=True, combined=True,
    include_last_observations=False,   # True to carry adstock into a forecast window
    clone_model=True,
)
# -> xr.Dataset, var named after target_column, dims (date, sample)
```

**The returned values are normalised.** Multiply by `target_scale` before computing
any metric or showing any chart:

```python
scale = float(mmm.get_scales_as_xarray()["target_scale"])
y_hat = pp["y"] * scale
```

### Inference data layout (`xr.DataTree`)

Groups after a full run: `posterior`, `posterior_predictive`, `prior`,
`prior_predictive`, `sample_stats`, `observed_data`, `constant_data`, `fit_data`.

Posterior variables in a standard fit:
`adstock_alpha`, `saturation_lam`, `saturation_beta`, `intercept_contribution`,
`channel_contribution`, `fourier_contribution`, `yearly_seasonality_contribution`,
`gamma_fourier`, `y_sigma`, `total_media_contribution_original_scale`.

`constant_data`: `channel_data`, `channel_scale`, `target_data`, `target_scale`, `dayofyear`.

Note the names: it is `intercept_contribution`, not `intercept`, and the likelihood
sigma is `<target>_sigma`.

```python
idata["posterior"]["channel_contribution"]     # dims (chain, draw, date, channel), normalised
list(idata.children)                            # group names
```

### Persistence

```python
mmm.save("model.nc")          # needs h5netcdf or netCDF4 installed
mmm2 = MMM.load("model.nc")
```

`DataTree.to_netcdf` has no writer without one of those packages, and the error appears
only at save time — after the sampling you just paid for. Install it up front.

---

## 6. Calibration with experiments

The single highest-value API in the library. Build the model first.

```python
mmm.build_model(X, y)
df_lift = pd.DataFrame([
    # one row per measured lift; add a column per entry in mmm.dims
    {"channel": "tv", "x": 0.0, "delta_x": 60_000.0, "delta_y": 1_500.0, "sigma": 400.0},
])
mmm.add_lift_test_measurements(df_lift, dist=pmd.Gamma, name="lift_measurements")
```

It conditions `saturation(x + delta_x) - saturation(x)` on the measured `delta_y` with
uncertainty `sigma`, i.e. it pins the response curve at a point somebody actually
measured. Call it more than once and pass a distinct `name` each time.

ROAS / cost-per-target calibration:

```python
mmm.add_cost_per_target_calibration(
    data=spend_frame,                 # X-like, channel values in spend units
    calibration_data=pd.DataFrame([
        {"channel": "tv", "cost_per_target": 42.0, "sigma": 8.0},
    ]),
    target_column="cost_per_target",
    target_per_cost=False,            # True calibrates contribution/spend (ROAS)
)
```

Events:

```python
mmm.add_events(df_events, prefix="promo", effect=EventEffect(...))
```

Helpers in `pymc_marketing.mmm.lift_test`: `scale_lift_measurements`,
`scale_channel_lift_measurements`, `assert_monotonic`, `exact_row_indices`,
`add_saturation_observations`.

---

## 7. Attribution

### Incrementality (preferred)

```python
incr = mmm.incrementality                          # an Incrementality instance
roas = incr.contribution_over_spend(frequency="quarterly")   # ROAS
cac  = incr.spend_over_contribution(frequency="monthly")     # CPA / CAC
mroi = incr.marginal_contribution_over_spend(frequency="quarterly")
inc  = incr.compute_incremental_contribution(frequency="monthly", include_carryover=True)
joint = incr.compute_joint_incremental_contribution(frequency="quarterly")
```

Each accepts `start_date`, `end_date`, `include_carryover`, `num_samples` and returns an
xarray object with the full posterior, so you get intervals rather than a point.

This is counterfactual: it re-runs the model with spend perturbed and takes the
difference, so it respects the link function and counts carryover that lands outside the
window. Dividing `channel_contribution` by spend does neither, and under a log link it is
simply wrong.

### Direct contributions

```python
scale = float(mmm.get_scales_as_xarray()["target_scale"])
contrib = mmm.idata["posterior"]["channel_contribution"] * scale       # identity link only

df = mmm.compute_mean_contributions_over_time(central_tendency="median")
ds = mmm.compute_counterfactual_contributions_dataset()                # link-aware, any link
mmm.add_original_scale_contribution_variable(var=["channel_contribution", "y"])
```

Under a **log link**, `add_original_scale_contribution_variable` on a component gives
`exp(component) * target_scale` — a multiplicative factor, not an additive contribution.
Use `compute_counterfactual_contributions_dataset()` instead, and expect per-component
counterfactuals to sum to *more* than the total, because interactions are counted by every
component that participates in them.

### Summary factory

`mmm.summary` is a ready `MMMSummaryFactory` — no wrapper construction needed.

```python
mmm.summary.contributions()          mmm.summary.roas()
mmm.summary.channel_share_hdi()      mmm.summary.total_contribution()
mmm.summary.channel_spend()          mmm.summary.waterfall()
mmm.summary.saturation_curves()      mmm.summary.adstock_curves()
mmm.summary.residuals_over_time()    mmm.summary.residuals_distribution()
mmm.summary.prior_vs_posterior()     mmm.summary.posterior_predictive()
mmm.summary.change_over_time()       mmm.summary.sensitivity_analysis()
mmm.table()                          # rich console table
```

`MMMSummaryFactory(data, model=None, hdi_probs=(0.94,), output_format="pandas")`;
`MMMIDataWrapper` now lives at `pymc_marketing.data.idata.mmm_wrapper`.

---

## 8. Response curves and sensitivity

```python
curve = mmm.sample_saturation_curve(
    max_value=1.0,            # in SCALED space; original_max / channel_scale.mean()
    num_points=100,
    num_samples=500,
    original_scale=True,      # y in target units; x stays scaled
    random_state=42,
)
ads = mmm.sample_adstock_curve(amount=1.0)

mmm.sensitivity.run_sweep(
    "channel_data",
    sweep_values=np.linspace(0, 1.5, 12),
    var_names="channel_contribution",
    sweep_type="multiplicative",       # multiplicative | additive | absolute
)
mmm.plot.sensitivity_analysis()

mmm.effective_carryover_lags()   # l_max plus any extra carryover declared by effects
```

---

## 9. Budget optimisation

```python
opt = mmm.budget_optimizer("2026-01-05", "2026-03-30")   # builds the optimisation model
result = opt.allocate_budget(
    total_budget=1_000_000,
    budget_bounds={"tv": (100_000, 400_000), "sem": (50_000, 200_000)},
    x0=None,
    minimize_kwargs=None,
    return_if_fail=False,
)   # -> BudgetOptimizationResult (changed in 1.1; older code expected a tuple)
```

`mmm.budget_optimizer(start, end)` computes `num_periods` and pulls `adstock_periods`
from the fitted adstock, so carry-in and carry-over are handled. Constructing
`BudgetOptimizer` by hand means getting those right yourself.

Direct construction, when you need custom objectives:

```python
BudgetOptimizer(
    num_periods=13, model=pm_model, idata=idata,
    adstock_periods=8, carry_in_periods=0,
    response_variable="total_media_contribution_original_scale",
    utility_function=...,
)
opt.set_constraints([...])
```

Optimising over a window shorter than the carryover understates every channel with a
long tail: the objective only counts the response that lands inside the window.
`effective_carryover_lags()` tells you how wide the window has to be.

---

## 10. Validation

```python
from pymc_marketing.mmm import TimeSliceCrossValidator

cv = TimeSliceCrossValidator(
    n_init=80, forecast_horizon=13, date_column="date", step_size=13,
    sampler_config={"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.95},
)

class Builder:                       # must satisfy the MMMBuilder protocol
    def build_model(self, X, y):
        return make_mmm()

n_folds = cv.get_n_splits(X, y)
combined = cv.run(X, y, mmm=Builder())
mmm.plot.cv_predictions(); mmm.plot.cv_crps(); mmm.plot.param_stability()
```

```python
from pymc_marketing.mmm.evaluation import (
    compute_summary_metrics, calculate_metric_distributions, summarize_metric_distributions,
)
metrics = compute_summary_metrics(y_true, y_pred_samples)   # per-draw, then summarised
```

---

## 11. ArviZ 1.x

```python
import arviz as az

az.summary(idata, var_names=["adstock_alpha"], ci_prob=0.89)
# columns: mean, sd, eti89_lb, eti89_ub, ess_bulk, ess_tail, r_hat, mcse_mean, mcse_sd

az.loo(idata)          # az.waic no longer exists
az.compare({"a": idata_a, "b": idata_b})
az.bfmi(idata); az.ess(idata); az.rhat(idata); az.hdi(idata); az.eti(idata)
az.plot_trace(...); az.plot_energy(...)
az.from_netcdf(path); az.from_zarr(path)
```

`az.summary` returns ETI-89 by default, not HDI-3%/97%. `hdi_prob=` is now `ci_prob=`.
`arviz.InferenceData` is gone; groups are `DataTree` children:

```python
posterior = idata["posterior"].to_dataset()
"prior" in idata.children
```

---

## 12. Other components

```python
from pymc_marketing.mmm import LinearTrend, YearlyFourier, Scaling, FixedScaling, DataDerivedScaling

LinearTrend(n_changepoints=10, include_intercept=False, dims=None, priors=None, prefix="")
YearlyFourier(n_order=6, prior=Prior("Laplace", mu=0, b=1))

# Fixed scaling keeps a production refresh comparable to the previous one:
# with data-derived scaling, a new peak silently rescales every prior.
Scaling(target=FixedScaling(dims=(), value=50_000.0),
        channel=DataDerivedScaling(method="max", dims=()))
```

**YAML model definition** — `pymc_marketing.mmm.builders`:

```python
from pymc_marketing.mmm.builders import build_mmm_from_yaml, MMMYamlConfig
mmm = build_mmm_from_yaml("model.yaml", X=X, y=y)
# keys: model, effects, extra_vars, data, original_scale_vars, idata_path
```

**R2D2 variance-decomposition prior** — `pymc_marketing.r2d2`:

```python
from pymc_marketing.r2d2 import R2D2
r2d2 = R2D2(
    r2=Prior("Beta", mu=0.8, sigma=0.2),
    total_sigma=Prior("LogNormal", mu=np.log(y.std()), sigma=0.1),
    dims={"channel": "channel", "control": "control"},
)
```

Instead of setting each coefficient prior independently and hoping the implied
decomposition is sane, R2D2 states the total explained variance and splits it across
components with a Dirichlet. That is the direct way to encode "media should account
for roughly a fifth of the variation". Every covariate you want covered must appear
in `dims`.

**Causal identification**:

```python
mmm = MMM(..., dag="digraph {tv -> sales; price -> sales; price -> tv;}",
          treatment_nodes=["tv"], outcome_node="sales")
```

Needs the `dag` extra (`dowhy`, `networkx`, `pygraphviz`). `pymc_marketing.mmm.causal`
also provides `BuildModelFromDAG`, `CausalGraphModel` and `TBFPC` (Bayes-factor causal
discovery oriented on a target).

---

## 13. Plotting

`mmm.plot.*` (matplotlib), `mmm.plot_interactive.*` (Plotly), `mmm.plot_suite`.

```
posterior_predictive           prior_predictive            prior_vs_posterior
posterior_distribution         channel_parameter           waterfall_components_decomposition
contributions_over_time        channel_contribution_share_hdi
residuals_over_time            residuals_posterior_distribution
saturation_curves              saturation_curves_scatter   saturation_scatterplot
marginal_curve                 uplift_curve
cv_crps                        cv_predictions              param_stability
budget_allocation              allocated_contribution_by_channel_over_time
sensitivity_analysis
```

---

## 14. Failure modes and their causes

| Symptom | Cause |
|---|---|
| `ImportError: cannot import name 'MMM' from 'pymc_marketing.mmm.multidimensional'` | 1.x path is `pymc_marketing.mmm` |
| `AttributeError: module 'arviz' has no attribute 'waic'` | Removed in ArviZ 1.x; use `az.loo` |
| `DataTree.__init__() got an unexpected keyword argument 'posterior'` | Use `xr.DataTree.from_dict({...})` |
| `ValueError: cannot write NetCDF files ... none of the suitable backend libraries` | `pip install h5netcdf` |
| Metrics look impossibly good/bad | Posterior predictive is normalised; multiply by `target_scale` |
| Model builds but `y` errors | `y.name` must equal `target_column` |
| `dims` behaves oddly, coords look like single letters | `dims` was a string; it must be a tuple |
| Contributions do not sum to the target under `link="log"` | Expected — use `compute_counterfactual_contributions_dataset()` |
| Optimiser understates long-carryover channels | Window shorter than `effective_carryover_lags()` |

---

## The agent_mmm library

Ships at `plugins/agent-mmm/lib/agent_mmm/`. Install: `pip install -e "plugins/agent-mmm[pymc]"`.

| Module | Entry point | Purpose |
|---|---|---|
| `spec` | `load_spec`, `MMMSpec` | Framework-agnostic project definition |
| `data_audit` | `run_audit(spec, base)` | Contract → shape → integrity → identifiability → signal → semantics |
| `data_prep` | `prepare_dataset(spec, df)` | Aggregation, dense reindexing, explicit imputation, calendar features |
| `controls_engine` | `recommend_controls(spec, audit, base)` | External-factor recommendations |
| `prior_engine` | `recommend_priors(spec, audit, base)` | Half-life priors, ROI priors, structure |
| `backends` | `get_backend(framework)` | pymc-marketing (executable), Meridian / Robyn (codegen) |
| `model_factory` | `compile_spec(spec, priors)` | Spec → framework, with unsupported features reported |
| `fit_runner` | `run_fit(spec, priors, ...)` | Prior PC → calibrate → fit → posterior PC → save |
| `diagnostics` | `run_diagnostics(run_id, ...)`, `decompose(idata)` | Convergence, baseline, contraction, plausibility |
| `iter_loop` | `run_tournament(...)` | Variant tournament and refinement |
| `reports.*` | `generate_*_report(...)` | CMO / CFO / MOps / DS |

Workspace: `./mmm-workspace/` — `spec.yaml`, `audit/`, `controls/`, `priors/`,
`runs/<id>/`, `leaderboard.json`, `reports/`.
