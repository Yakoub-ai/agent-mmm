---
name: mmm-model-building
description: |
  Constructing an MMM with pymc-marketing 1.x — model architecture, adstock and saturation choice, prior specification, likelihood and link function, seasonality and trend, panel dimensions, and the fitting strategy. Use when building a new model, choosing transformations, writing model_config, setting priors from channel knowledge, deciding between additive and multiplicative structure, or designing the prior-predictive-to-fit sequence.
---

# Building an MMM with pymc-marketing 1.x

Written for pymc-marketing >= 1.0, where `from pymc_marketing.mmm import MMM` is the
correct import, `idata` is an `xarray.DataTree`, and the legacy 0.x MMM class no longer
exists. See `mmm-api-reference` for exact signatures.

The model is a series of decisions, each of which is a claim about how marketing works.
This skill is about making those claims deliberately.

---

## 1. The structure

```
target = baseline + Σ_channels beta_c · saturation(adstock(x_c)) + noise
baseline = intercept + trend + seasonality + controls + non-media treatments
```

Four decisions determine everything downstream:

1. **What enters as media** and what enters as baseline — see `mmm-channel-semantics`.
2. **How flexible the baseline is** — see `mmm-baseline-and-trend`. This is the single
   largest lever on the media share.
3. **The shape of the media response** — adstock and saturation, below.
4. **How much the priors constrain** — below, and `mmm-experimentation-calibration`.

---

## 2. Constructor

```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

mmm = MMM(
    date_column="date",
    channel_columns=["tv_grps", "sem_spend", "social_spend"],
    target_column="revenue",
    adstock=GeometricAdstock(l_max=12),
    saturation=LogisticSaturation(),
    control_columns=["price_index", "acv", "is_holiday"],
    yearly_seasonality=6,
    adstock_first=True,
    dims=("geo",),                # TUPLE — a bare string iterates into characters
    link="identity",              # "log" for a multiplicative model
    model_config=model_config,
)
```

Requirements that fail unhelpfully if missed:

* `y` must be a Series named exactly `target_column`.
* `X` must contain the date column, every channel, every control, and one categorical
  column per entry in `dims`.
* Panel data must be rectangular — every (geo, date) cell present.
* Target and channels are max-abs scaled internally; **controls are not**. Standardise
  controls yourself or the default `gamma_control` prior means something you did not
  intend.

---

## 3. Adstock

Carryover is a property of how the ad is consumed. Attention-bought media has a long tail;
intent-bought media does not.

| Class | Shape | Use for |
|---|---|---|
| `GeometricAdstock` | Monotone decay from t=0 | Digital, search, social — effect starts immediately |
| `DelayedAdstock` | Peak at lag `theta`, then decay | TV, print, direct mail, cinema — response builds after exposure |
| `WeibullCDFAdstock` / `WeibullPDFAdstock` | Flexible accumulation / peak-and-decay | When the shape is genuinely unknown and you have the data to identify it |
| `BinomialAdstock` | Bounded-support decay | Meridian parity |
| `NoAdstock` | Instant | Price, distribution, promotions |

**Specify carryover as a half-life, not a decay rate.** "Half the TV effect is gone after
3.5 weeks" is a claim a marketer can challenge; `alpha = 0.82` is not. They are the same
number: `alpha = 0.5 ** (1/halflife)`. pymc-marketing 1.1 supports
`parametrization="halflife"` on `GeometricAdstock` and `DelayedAdstock`, which puts the
half-life directly in the posterior.

Weekly starting points: search 0.3–0.5, social 1.5–2, video 2.5–3, TV 3.5, OOH 4.

**`l_max` must be wide enough.** Keep ~95% of the impulse:
`l_max >= ceil(log(0.05) / log(alpha))`. Truncating too early biases the decay estimate
downwards, because the model cannot express the tail it truncated. For TV at half-life 3.5
that means `l_max` around 16, not 8.

**One adstock for all channels, or several?** `MMM(adstock=...)` applies one to every
channel. `MediaConfigList` allows per-channel transformations but renames the variables
(they take the config name as a prefix), which breaks helpers keyed on `adstock_alpha`.
The pragmatic default is one adstock type with `l_max` set to the widest requirement:
short-carryover channels pay a little compute, and no channel's tail is truncated.

---

## 4. Saturation

Saturation is a property of the addressable audience. Brand search saturates almost
immediately; national video barely saturates within realistic budgets.

| Class | Shape | Notes |
|---|---|---|
| `LogisticSaturation` | `beta(1 - e^{-lam·x})`, concave from zero | Default. `lam` is steepness in scaled space |
| `InverseScaledLogisticSaturation` | Same curve | `lam` is the **half-saturation point** — much easier to set a prior on |
| `HillSaturation` | S-curve, `slope > 1` gives a threshold | Only when you have data near zero spend |
| `MichaelisMentenSaturation` | `alpha·x/(lam+x)` | `alpha` is the asymptote — interpretable |
| `TanhSaturationBaselined` | Anchored at a reference spend | Useful for brownfield continuity |
| `LogSaturation` | `beta·log(1+x)` | Never fully saturates |
| `NoSaturation` | Linear | Genuinely linear range |

**Concave vs S-shaped is a substantive claim.** Concave says the first pound is the most
productive. S-shaped says there is a minimum effective dose below which nothing happens.
Threshold effects are real for broad-reach media, but they are only identifiable if the
data contains spend in the threshold region. Choosing Hill for its flexibility, without
low-spend observations, means the prior decides where the threshold is — and the budget
optimiser will take that seriously.

**`lam` in scaled space.** Channels are max-abs scaled, so `x` runs roughly 0–1 and
`lam ≈ 3` means ~95% of the maximum effect at the channel's historical peak spend;
`lam ≈ 1` means still near-linear at peak. Prefer `InverseScaledLogisticSaturation` when
you want to reason about this directly.

**Curves outside the observed range are extrapolation.** The optimiser will explore spend
levels you never ran; the curve there is the prior's shape, not a finding. Bound the
optimiser, and say in the report where the data stops.

---

## 5. Priors

### Half-life first, distribution second

```python
import numpy as np
from pymc_extras.prior import Prior
from agent_mmm.utils.channel_classifier import halflife_to_alpha, suggested_l_max
from agent_mmm.utils.moment_match import beta_moment_match, gamma_moment_match

halflives = {"tv_grps": 3.5, "sem_spend": 0.5, "social_spend": 1.5}
alpha_mu = np.array([halflife_to_alpha(h) for h in halflives.values()])
alpha_sd = np.array([0.10, 0.10, 0.13])
a, b = zip(*[beta_moment_match(m, s) for m, s in zip(alpha_mu, alpha_sd)])
```

### The intercept prior is a prior about media

In scaled space the target runs roughly 0–1. An intercept prior centred at 1.0 tells the
model the baseline explains everything and media explains nothing; centred at 0 it says
the opposite. Neither is neutral.

```python
expected_media_share = 0.25
"intercept": Prior("Normal", mu=1 - expected_media_share, sigma=0.3)
```

Or state the variance decomposition directly with an R2D2 prior
(`pymc_marketing.r2d2.R2D2`), which splits total explained variance across components via
a Dirichlet instead of hoping independent coefficient priors imply something sane. Every
covariate you want covered must appear in its `dims`.

### Spend-share sigma for `saturation_beta`

A channel buying 2% of the media cannot plausibly move the target as much as one buying
40%. Scaling the prior width by spend share encodes that without asserting an effect size:

```python
shares = np.array([X[c].sum() for c in channels]); shares = shares / shares.sum()
"saturation_beta": Prior("HalfNormal", sigma=np.maximum(shares, 0.02), dims="channel")
```

### Widen where the data cannot support confidence

| Situation | Action |
|---|---|
| >30% zero-spend periods | widen by ~1.5x |
| Fewer than 104 periods | widen by ~1.3x |
| Always-on or flat spend (CV < 0.15) | widen by ~1.4x, and say the posterior reflects the prior |
| Channel with an experiment | tighten, or better, add the lift test as a constraint |

A tight prior on an unidentified channel is not a modelling choice, it is a way of
presenting an assumption as a finding.

### Complete `model_config`

```python
model_config = {
    "adstock_alpha":  Prior("Beta", alpha=np.array(a), beta=np.array(b), dims="channel"),
    "saturation_lam": Prior("Gamma", alpha=lam_a, beta=lam_b, dims="channel"),
    "saturation_beta":Prior("HalfNormal", sigma=shares, dims="channel"),
    "intercept":      Prior("Normal", mu=0.75, sigma=0.3),
    "gamma_control":  Prior("Normal", mu=0, sigma=0.5, dims="control"),
    "gamma_fourier":  Prior("Laplace", mu=0, b=0.3, dims="fourier_mode"),
    "likelihood":     Prior("StudentT", nu=5, sigma=Prior("HalfNormal", sigma=0.5)),
}
```

`agent_mmm.prior_engine.recommend_priors(spec)` generates all of this from the spec plus
the data's own characteristics, and writes an audit report explaining every value.

---

## 6. Likelihood and link

| Likelihood | When |
|---|---|
| `Normal` | Clean data, no spikes |
| `StudentT(nu=5)` | **Default.** Robust to promotional spikes and data errors |
| `StudentT(nu=3)` | Very heavy tails |
| `LogNormal` | With `link="log"` |

Lower `nu` means heavier tails and more tolerance for outliers. Robustness is preferable to
deleting the outlying weeks, which biases the model towards the ordinary.

**Link function.** `link="identity"` is additive: components sum to the target, and each
channel's contribution is `beta · saturation(adstock(x))`. `link="log"` is multiplicative:
components combine on the log scale, which matches how price and media often interact
(media has more effect when price is competitive).

The cost of the log link is that a single component has no standalone additive
contribution. Use `compute_counterfactual_contributions_dataset()` for decomposition, and
expect per-component counterfactuals to sum to *more* than the total — interactions are
counted by every component that participates. Say that in the report rather than letting
a reader assume the decomposition is broken.

---

## 7. Seasonality, trend, and panel dimensions

```python
mmm = MMM(..., yearly_seasonality=6, time_varying_intercept=False, dims=("geo",))
```

* `yearly_seasonality=n` adds `2n` Fourier parameters. Weekly data: 4–8. More modes fit
  sharper shapes and absorb more of the variance media could explain.
* Named events (Christmas, Black Friday, Ramadan) belong in `control_columns` as explicit
  flags, not in the Fourier order. They are step changes, and they are exactly when
  campaigns run.
* `time_varying_intercept=True` (HSGP) gives a smooth varying baseline. Needs 104+
  periods, and the lengthscale should be **coarser than your media flighting** — otherwise
  the baseline absorbs the campaigns.
* `LinearTrend(n_changepoints=k)` as an additive effect is the middle option between a
  fixed intercept and a full HSGP.
* `dims=("geo",)` makes every parameter geo-specific with partial pooling. Requires a
  rectangular panel. See `mmm-multi-geo-panel`.

Fit the model with and without the flexible baseline and compare the media share. If it
moves by a third, your conclusion is a modelling choice and the report must say so.

---

## 8. Fitting sequence

```python
# 1. Build the graph explicitly — needed before priors or constraints.
mmm.build_model(X, y)

# 2. Prior predictive BEFORE seeing the answer. Afterwards you cannot un-see it.
prior = mmm.sample_prior_predictive(X=X, y=y, samples=500, extend_idata=True)
```

Check three things: does the 90% band cover the observed range (below ~80% coverage means
the priors rule out what happened); is the band a sane width (10x the data range means
vague priors and slow sampling); and is the implied media share plausible (if the priors
imply media drives 90% of sales, the intercept prior is wrong, not the channel priors).

```python
# 3. Calibrate. This is what separates an MMM you can act on from a curve fit.
mmm.add_lift_test_measurements(df_lift_test)

# 4. Fit.
mmm.fit(X, y, draws=2000, tune=3000, chains=4, target_accept=0.95, random_seed=42)

# 5. Posterior predictive — returns NORMALISED values.
pp = mmm.sample_posterior_predictive(X, extend_idata=True)
scale = float(mmm.get_scales_as_xarray()["target_scale"])
y_hat = pp["revenue"] * scale
```

**Sampler settings.** Start at `target_accept=0.9`; raise to 0.95–0.99 only in response to
divergences. Maxing it out by default buys slow sampling and hides the geometry problem
that caused the divergences. `draws=1000, tune=1500` is fine for exploration;
`draws=2000, tune=3000, chains=4` for a model that will move budget.

**Persistence.** `mmm.save("model.nc")` needs `h5netcdf` or `netCDF4` installed — and the
error only appears after you have paid for the sampling. Install it up front.

---

## 9. Channel grouping

**Group when the data cannot separate.** Pairwise correlation above ~0.8 or VIF above ~10
means whatever split the model reports is the prior's opinion. Reporting one honest number
beats reporting two fabricated ones.

**Split when the halves behave differently and are flighted differently.** Brand vs generic
search is the highest-value split in most accounts; prospecting vs retargeting is second.

**Count the parameter budget.** Each channel costs ~3 parameters with no new observations.
104 weeks and 10 channels is roughly 3 observations per parameter, and the priors are
doing most of the work. `agent_mmm.data_audit` reports this ratio explicitly.

**No circularity.** Do not include SEM spend in a model whose target is search volume —
search volume drives SEM impressions, not the reverse. See `mmm-causal-design`.

---

## 10. Build order

```
1.  Roles for every column        -> mmm-channel-semantics
2.  DAG and control set           -> mmm-causal-design
3.  Baseline specification        -> mmm-baseline-and-trend
4.  Adstock / saturation / l_max  -> this skill
5.  Priors, in interpretable units-> this skill + prior_engine
6.  Prior predictive check        -> gate before fitting
7.  Lift-test calibration         -> mmm-experimentation-calibration
8.  Fit
9.  Diagnostics, baseline first   -> mmm-diagnostics
10. Validation and sensitivity    -> mmm-validation
```

Steps 1–3 decide whether the answer is right. Steps 4–8 decide how precisely it is
estimated. Time spent tuning the sampler instead of the specification is almost always
misallocated.

---

## Library

```python
from agent_mmm.spec import load_spec
from agent_mmm.prior_engine import recommend_priors
from agent_mmm.model_factory import compile_spec
from agent_mmm.fit_runner import run_fit

spec = load_spec("mmm-workspace/spec.yaml")
priors = recommend_priors(spec)
result = compile_spec(spec, priors=priors)   # .model, .code, .unsupported, .warnings
metrics = run_fit(spec, priors=priors)
```

Slash commands: `/mmm-recommend-priors`, `/mmm-build`, `/mmm-fit`.
