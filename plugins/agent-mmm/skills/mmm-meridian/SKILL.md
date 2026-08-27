---
name: mmm-meridian
description: |
  Google Meridian reference and practice guide, verified against google-meridian 1.8.0. Use when building, reviewing or debugging a Meridian model, wiring InputData and CoordToColumns, setting ROI/mROI/contribution priors, configuring knots and adstock/saturation specs, handling reach-and-frequency channels, running the Analyzer or BudgetOptimizer, or translating a model between Meridian and pymc-marketing.
---

# Google Meridian (verified against v1.8.0)

Meridian is Google's open-source Bayesian MMM: TensorFlow Probability, geo-hierarchical
by default, with media effects parameterised **by ROI** rather than by coefficient. That
last choice is the important one — the prior sits directly on the quantity stakeholders
argue about.

Install: `pip install google-meridian` (Python ≥ 3.10; pulls TensorFlow 2.21 and a pinned
`tfp-nightly`). GPU: `google-meridian[and-cuda]`. There is also a `[jax]` extra.

---

## 1. When Meridian is the right choice

**Choose Meridian when:**
* You have a geo panel with population data. Hierarchical geo pooling is its core
  strength and the main reason to prefer it.
* Reach and frequency matter. Meridian is the only major open framework that models RF
  natively and answers "what frequency should we buy?".
* You want ROI priors. Calibrating to a geo experiment is a one-line prior change rather
  than a reverse-engineered coefficient.
* You need organic media and non-media treatments kept structurally separate from paid
  media.
* You want a built-in EDA guardrail that refuses to fit data with critical problems.

**Choose something else when:**
* The model is national with no geo dimension — you lose the main advantage and keep the
  TensorFlow dependency.
* You need a multiplicative (log-link) model. Meridian's media terms are additive.
* You need custom likelihoods, arbitrary priors on arbitrary parameters, or a mediation
  structure. pymc-marketing is a modelling toolkit; Meridian is a well-designed
  application.
* You need Fourier seasonality specifically. Meridian uses knots for time effects.

---

## 2. Data model

Meridian's `InputData` distinguishes variable kinds explicitly — the structural feature
that stops a price column from acquiring a ROAS.

| Slot | Dims | Meaning |
|---|---|---|
| `kpi` | (geo, time) | Outcome. `kpi_type` is `"revenue"` or `"non_revenue"` |
| `revenue_per_kpi` | (geo, time) | Value per KPI unit; turns a non-revenue KPI's ROI into money |
| `population` | (geo,) | Required for geo models; enables per-capita scaling |
| `media` | (geo, media_time, channel) | Paid exposure (impressions, GRPs) |
| `media_spend` | (geo, time, channel) | Paid cost |
| `reach`, `frequency`, `rf_spend` | (geo, ·, rf_channel) | Reach-and-frequency channels |
| `organic_media` | (geo, media_time, channel) | Unpaid activity with carryover |
| `organic_reach`, `organic_frequency` | | Organic RF |
| `non_media_treatments` | (geo, time, channel) | Price, promo, distribution |
| `controls` | (geo, time, variable) | Confounders |

`n_media_times >= n_times`: media may extend earlier than the KPI so pre-window exposure
drives adstock into the modelled period. This is a genuine advantage — most frameworks
lose the first `l_max` periods to a cold start.

```python
from meridian.data import load

coord_to_columns = load.CoordToColumns(
    time="date", geo="geo", kpi="revenue", population="population",
    revenue_per_kpi=None,
    media=["tv_grps", "yt_impressions"], media_spend=["tv_spend", "yt_spend"],
    reach=["ctv_reach"], frequency=["ctv_frequency"], rf_spend=["ctv_spend"],
    organic_media=["email_sends"],
    non_media_treatments=["price_index", "acv"],
    controls=["competitor_spend", "consumer_confidence"],
)
loader = load.DataFrameDataLoader(
    df=df, kpi_type="revenue", coord_to_columns=coord_to_columns,
    media_to_channel={"tv_grps": "TV", "yt_impressions": "YouTube"},
    media_spend_to_channel={"tv_spend": "TV", "yt_spend": "YouTube"},
)
data = loader.load()
```

Loaders: `DataFrameDataLoader`, `CsvDataLoader`, `XrDatasetDataLoader`. The panel must be
rectangular — every (geo, time) cell present.

**National models** use a single constant geo. Everything works; the hierarchy is inert.

---

## 3. `ModelSpec`

```python
from meridian.model import spec as model_spec

spec = model_spec.ModelSpec(
    prior=prior,                                   # PriorDistribution
    media_effects_dist="log_normal",               # or "normal"; geo random effects
    hill_before_adstock=False,
    max_lag=8,
    unique_sigma_for_each_geo=False,
    media_prior_type="roi",                        # roi | mroi | contribution | coefficient
    rf_prior_type="roi",
    roi_calibration_period=None,                   # (n_media_times, n_media_channels) bool
    rf_roi_calibration_period=None,
    organic_media_prior_type="contribution",       # contribution | coefficient
    organic_rf_prior_type="contribution",
    non_media_treatments_prior_type="contribution",
    non_media_baseline_values=None,                # float | "min" | "max" per channel
    knots=None,
    baseline_geo=None,
    holdout_id=None,                               # bool mask; excludes KPI, keeps media
    control_population_scaling_id=None,
    non_media_population_scaling_id=None,
    adstock_decay_spec="geometric",                # "geometric" | "binomial" | per-channel map
    saturation_spec="hill",                        # "hill" | "none" | per-channel map
    enable_aks=False,                              # automatic knot selection
)
```

### The four prior types

| `media_prior_type` | Prior on | Use when |
|---|---|---|
| `"roi"` (default) | Return per unit spend | You have ROI beliefs or experiments. **Default choice.** |
| `"mroi"` | Marginal ROI at current spend | You care about the next pound, not the average |
| `"contribution"` | Share of outcome | You believe a channel drives ~x% of sales |
| `"coefficient"` | `beta_m` directly | You are porting a coefficient-parameterised model |

`roi_calibration_period` restricts the ROI prior to the window a test actually covered —
essential when a channel was tested in one quarter only.

### Knots: the most consequential setting

`knots` controls the flexibility of time effects, which is Meridian's baseline.

* `knots=None` on a **geo** model means **one knot per time period** — the most flexible
  baseline available, able to absorb almost any variation media might otherwise explain.
* `knots=None` on a national model means 1 (a constant).
* `knots=k` places k equally spaced knots; `knots=[0, 25, 51]` places them explicitly.
* `enable_aks=True` selects the number automatically; it errors if `knots` is also set.

**Never leave `knots=None` on a geo model without deciding it.** Fewer knots means a
stiffer baseline and more variance available for media; more knots means the opposite.
This is a prior about how much marketing does, expressed as a smoothing parameter. Fit at
two settings and compare the media share before choosing.

### Adstock and saturation

`adstock_decay_spec`: `"geometric"` or `"binomial"`, globally or per channel. There is no
delayed/Weibull option — map a delayed request to `"binomial"`, which has a delayed peak,
and expect a slightly different carryover shape from an equivalent pymc-marketing model.

`saturation_spec`: `"hill"` or `"none"`, globally or per channel. `"none"` is right for
channels genuinely in their linear range.

`hill_before_adstock=False` (default) applies adstock first. Does not apply to RF channels.

---

## 4. Priors

`PriorDistribution` holds a TFP distribution per parameter family. Defaults verified
against 1.8.0:

| Parameter | Default | Meaning |
|---|---|---|
| `roi_m`, `roi_rf` | `LogNormal(0.2, 0.9)` | ROI; median ≈ 1.2, 90% range ≈ 0.3–5.3 |
| `mroi_m`, `mroi_rf` | `LogNormal(0.0, 0.5)` | Marginal ROI |
| `contribution_m/rf/om/orf` | `Beta(1, 99)` | Contribution share; mean 1% |
| `contribution_n` | `TruncatedNormal(0, 0.1, -1, 1)` | Non-media contribution, can be negative |
| `beta_m`, `beta_rf`, `beta_om`, `beta_orf` | `HalfNormal(5)` | Coefficients (non-negative) |
| `eta_m/rf/om/orf` | `HalfNormal(1)` | Geo-level random-effect scale |
| `gamma_c`, `gamma_n` | `Normal(0, 5)` | Control and non-media coefficients |
| `xi_c`, `xi_n` | `HalfNormal(5)` | Their geo random-effect scales |
| `alpha_m/rf/om/orf` | `Uniform(0, 1)` | Adstock decay |
| `ec_m`, `ec_om` | `TruncatedNormal(0.8, 0.8, 0.1, 10)` | Hill half-saturation |
| `ec_rf`, `ec_orf` | `LogNormal(0.7, 0.4)` shifted by 0.1 | Hill half-saturation, RF |
| `slope_m`, `slope_om` | `Deterministic(1.0)` | Hill slope fixed → concave |
| `slope_rf`, `slope_orf` | `LogNormal(0.7, 0.4)` | Hill slope, RF (S-curve allowed) |
| `knot_values` | `Normal(0, 5)` | Time effects |
| `tau_g_excl_baseline` | `Normal(0, 5)` | Geo intercepts vs baseline geo |
| `sigma` | `HalfNormal(5)` | Residual sd |

Note `slope_m` is **deterministic at 1.0**: paid media gets a concave Hill curve, not an
S-curve, unless you override it. RF channels do allow an S-curve, which is the right
default given how frequency behaves.

Helpers for stating a prior the way a human would:

```python
from meridian.model import prior_distribution as pd

pd.lognormal_dist_from_range(low=1.0, high=6.0, mass_percent=0.9)   # "ROAS is 1-6, 90% sure"
pd.lognormal_dist_from_mean_std(mean=3.0, std=1.0)
```

```python
import numpy as np, tensorflow_probability as tfp
tfd = tfp.distributions
# Channel order must match data.media_channel exactly.
prior = pd.PriorDistribution(roi_m=tfd.LogNormal([0.5, 0.2], [0.4, 0.9], name="roi_m"))
```

---

## 5. Fitting

```python
from meridian.model import model

mmm = model.Meridian(input_data=data, model_spec=spec)
mmm.sample_prior(500)
mmm.sample_posterior(n_chains=4, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)
health = mmm.review()          # post-fit health checks -> mmm.health_summary
# or: mmm.sample_posterior_and_review(...)
model.save_mmm(mmm, "model.pkl")
```

`n_chains` accepts a list (e.g. `[2, 2]`) to split sampling into separate calls and cap
memory. `posterior_thinning()` reduces a large posterior for downstream analysis.

**The EDA guardrail.** Meridian runs its own checks before fitting and refuses to proceed
on critical findings. Check types: `PAIRWISE_CORRELATION`, `STANDARD_DEVIATION`,
`MULTICOLLINEARITY`, `KPI_INVARIABILITY`, `COST_PER_MEDIA_UNIT`,
`VARIABLE_GEO_TIME_COLLINEARITY`, `POPULATION_CORRELATION`, `PRIOR_PROBABILITY`,
`DATA_ADEQUACY`. Read `mmm.eda_outcomes` rather than working around a refusal — the
checks map closely onto the problems that produce confidently wrong MMMs.

---

## 6. Analysis

```python
from meridian.analysis import analyzer, optimizer, summarizer, visualizer

an = analyzer.Analyzer(mmm)
an.rhat_summary(bad_rhat_threshold=1.2)
an.predictive_accuracy()
an.summary_metrics()                    # ROI, mROI, CPIK, contribution, effectiveness
an.baseline_summary_metrics()
an.negative_baseline_probability()      # check this before believing any ROI
an.incremental_outcome(...)
an.roi(); an.marginal_roi(); an.cpik()
an.response_curves(spend_multipliers=np.linspace(0, 2, 20))
an.adstock_decay(); an.hill_curves()
an.optimal_freq()                       # RF channels only
an.expected_vs_actual_data(split_by_holdout_id=True)
```

`negative_baseline_probability()` exists because the failure is common: a negative
baseline means the model is borrowing from an impossible counterfactual to pay media, and
every ROI downstream is inflated.

**CPIK** (cost per incremental KPI) is the non-revenue counterpart of ROI — the right
headline metric when the KPI is signups or policies.

```python
opt = optimizer.BudgetOptimizer(mmm)
res = opt.optimize(fixed_budget=True, budget=1_000_000,
                   spend_constraint_lower=0.3, spend_constraint_upper=0.3,
                   use_optimal_frequency=True)
res.output_optimization_summary("optimization.html", ".")

summarizer.Summarizer(mmm).output_model_results_summary("summary.html", ".")
```

Scenarios: `fixed_budget=True` (reallocate a fixed pot), `fixed_budget=False` with
`target_roi` or `target_mroi` (flexible budget to a return threshold). The second is the
more useful question and the one most teams never ask.

Visualizers: `ModelDiagnostics`, `ModelFit`, `MediaEffects`, `MediaSummary`,
`ReachAndFrequency` — each produces Altair charts and the DataFrames behind them.

---

## 7. Meridian ↔ pymc-marketing

| Concept | Meridian | pymc-marketing |
|---|---|---|
| Geo panel | native, population-weighted | `dims=("geo",)` |
| Media prior | ROI / mROI / contribution / coefficient | coefficient (`saturation_beta`) |
| Adstock | geometric, binomial | geometric, delayed, Weibull CDF/PDF, binomial |
| Saturation | Hill (slope fixed at 1 for paid media) | logistic, Hill, MM, tanh, root, log |
| Time effects | knots | Fourier, LinearTrend, HSGP |
| Reach & frequency | native + optimal frequency | not modelled |
| Organic media | own slot | `channel_columns` + role discipline |
| Non-media treatments | own slot | `control_columns` |
| Experiment calibration | ROI prior + `roi_calibration_period` | `add_lift_test_measurements` (likelihood) |
| Link function | additive only | identity or log |
| Pre-period media | `n_media_times > n_times` | not supported |
| Custom priors | fixed parameter families | any `Prior` on any parameter |
| Backend | TensorFlow Probability | PyMC / PyTensor |

**Porting caveats.** A delayed adstock has no exact Meridian equivalent; Fourier
seasonality has no equivalent at all; a log-link model cannot be ported. Expect different
numbers, and reconcile on the decision (which channels to grow) rather than on the point
estimates.

`agent_mmm.backends.get_backend("meridian").translate(spec, priors=...)` generates the
code and lists every spec feature Meridian cannot express.

---

## 8. Common failures

| Symptom | Cause |
|---|---|
| Fit refused before sampling | EDA guardrail found a critical issue — read `mmm.eda_outcomes` |
| Baseline absorbs almost everything | `knots=None` on a geo model; set an explicit number |
| ROI posterior ≈ ROI prior | The channel is not identified (flat spend, collinearity). Widen, or test |
| `negative_baseline_probability` is high | Missing driver; every ROI is inflated |
| Loader errors on shape | Panel is not rectangular, or media/KPI time axes disagree |
| ROI in KPI units, not money | Non-revenue KPI without `revenue_per_kpi` |
| A price variable appears with a ROI | It was loaded as `media` instead of `non_media_treatments` |
| Frequency recommendation missing | Channel loaded as `media` rather than `reach`/`frequency` |
