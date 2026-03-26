---
name: mmm-model-building
description: |
  Model construction and prior specification for Marketing Mix Models with pymc-marketing. Use when building a new MMM, choosing adstock/saturation transformations, specifying priors with moment matching, configuring the likelihood, setting up the model constructor, or designing the fitting strategy. Also activate when discussing prior calibration, spend-share sigma, channel prior tables, or model_config dictionaries.
---

# MMM Model Building with pymc-marketing

## Model Constructor

```python
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm import GeometricAdstock, DelayedAdstock, LogisticSaturation
from pymc_extras.prior import Prior

model = MMM(
    date_column="date",
    channel_columns=channel_cols,       # e.g., ["spend_SEM", "spend_TV", ...]
    target_column="target_col_name",    # NEW in multidimensional API
    adstock=GeometricAdstock(l_max=5),  # or DelayedAdstock(l_max=12)
    saturation=LogisticSaturation(),
    control_columns=control_cols,       # e.g., ["is_q4", "is_holiday", "interest_rate"]
    yearly_seasonality=8,               # Fourier modes (8 is good default for weekly)
    model_config=model_config,          # Prior specifications (see below)
    adstock_first=True,                 # Apply adstock before saturation (recommended)
)
```

## Adstock Selection

| Type | Class | Best For | Key Parameter |
|------|-------|----------|---------------|
| Geometric | `GeometricAdstock(l_max=5)` | Digital channels (fast decay) | alpha in [0,1]: higher = longer memory |
| Delayed | `DelayedAdstock(l_max=12)` | Offline channels (delayed onset) | Weibull CDF: peak-then-decay |

**Rule of thumb:**
- `l_max=5` for digital (SEM, Social, Display) -- effects dissipate within 5 weeks
- `l_max=8-12` for offline (TV, OOH, Print) -- effects can linger longer
- Use same adstock type for all channels in one model (pymc-marketing limitation)

## Saturation Selection

| Type | Class | Behavior |
|------|-------|----------|
| Logistic | `LogisticSaturation()` | `f(x) = 1 - exp(-lam * x)` -- smooth diminishing returns (primary choice) |
| Hill | `HillSaturation()` | S-curve with inflection point -- use when there's a minimum effective dose |
| Michaelis-Menten | `MichaelisMentenSaturation()` | Enzymatic kinetics curve |

**Default:** LogisticSaturation for most use cases. `lam` controls steepness, `beta` scales the output.

## Prior Specification

### Moment Matching

Convert intuitive (mu, sigma) to distribution parameters:

**Beta (adstock_alpha) -- bounded [0,1]:**
```python
def beta_moment_match(mu, sigma):
    C = max(mu * (1 - mu) / sigma**2 - 1, 0.5)
    return mu * C, (1 - mu) * C  # alpha, beta params
```

**Gamma (saturation_lam) -- positive real:**
```python
def gamma_moment_match(mu, sigma):
    alpha = (mu / sigma) ** 2
    beta = mu / sigma**2
    return alpha, beta
```

### Channel Prior Guidance

| Channel Type | alpha_mu | alpha_sigma | lam_mu | lam_sigma | Rationale |
|-------------|----------|-------------|--------|-----------|-----------|
| SEM/Search | 0.05-0.15 | 0.05-0.10 | 3.5-5.0 | 0.8-1.2 | Near-zero carryover, high saturation point |
| Social Media | 0.15-0.25 | 0.08-0.12 | 2.5-3.5 | 0.6-1.0 | Short carryover, moderate saturation |
| Digital Display | 0.20-0.30 | 0.10-0.15 | 2.5-3.5 | 0.6-1.0 | Short carryover |
| YouTube/Video | 0.30-0.50 | 0.12-0.18 | 2.0-3.0 | 0.5-0.8 | Medium carryover (video lingers) |
| Meta (FB+IG) | 0.25-0.35 | 0.10-0.15 | 2.5-3.5 | 0.6-1.0 | Short-medium carryover |
| Audio/Radio | 0.35-0.55 | 0.12-0.18 | 1.5-2.5 | 0.4-0.7 | Medium carryover |
| OOH/Outdoor | 0.45-0.75 | 0.15-0.25 | 1.5-2.5 | 0.4-0.7 | Long carryover (repeated exposure) |
| TV | 0.50-0.80 | 0.15-0.25 | 1.0-2.0 | 0.3-0.6 | Longest carryover, gradual saturation |
| Print | 0.40-0.60 | 0.15-0.20 | 1.5-2.5 | 0.4-0.7 | Medium-long carryover |

**Pattern:** Digital = low alpha (fast decay), high lambda (steep saturation). Offline = high alpha (slow decay), low lambda (gradual saturation).

### Spend-Share Sigma for saturation_beta

Scale the prior width by each channel's share of total spend -- higher-spending channels get wider priors:
```python
total_spend = sum(X[ch].sum() for ch in channels)
sigma = np.array([X[ch].sum() / total_spend for ch in channels])
model_config["saturation_beta"] = Prior("HalfNormal", sigma=sigma, dims="channel")
```

### Complete model_config Example

```python
import numpy as np
from pymc_extras.prior import Prior

# Moment-match per-channel priors
alpha_a, alpha_b = [], []
lam_a, lam_b = [], []
for ch in channels:
    mu, sig = channel_priors[ch]["alpha_mu"], channel_priors[ch]["alpha_sigma"]
    C = max(mu * (1 - mu) / sig**2 - 1, 0.5)
    alpha_a.append(mu * C); alpha_b.append((1 - mu) * C)

    mu_l, sig_l = channel_priors[ch]["lam_mu"], channel_priors[ch]["lam_sigma"]
    lam_a.append((mu_l / sig_l)**2); lam_b.append(mu_l / sig_l**2)

model_config = {
    # Per-channel priors
    "adstock_alpha": Prior("Beta", alpha=np.array(alpha_a), beta=np.array(alpha_b), dims="channel"),
    "saturation_lam": Prior("Gamma", alpha=np.array(lam_a), beta=np.array(lam_b), dims="channel"),
    "saturation_beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),

    # Model-level priors
    "intercept": Prior("Normal", mu=0.5, sigma=0.5),
    "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
    "gamma_fourier": Prior("Laplace", mu=0, b=0.3, dims="fourier_mode"),

    # Likelihood -- Student-t for outlier robustness
    "likelihood": Prior("StudentT", nu=5, sigma=Prior("HalfNormal", sigma=0.5)),
}
```

## Likelihood Selection

| Type | When to Use |
|------|-------------|
| `Normal` | Clean data, no outliers expected |
| `StudentT(nu=5)` | **Recommended default** -- robust to outliers (holiday spikes, data anomalies) |
| `StudentT(nu=3)` | Very heavy tails -- extreme outlier tolerance |

Lower `nu` = heavier tails = more outlier tolerance. `nu=5` balances robustness and efficiency.

## Fitting Strategy

### Step 1: Prior Predictive Check
```python
prior = model.sample_prior_predictive(X=X, y=y, samples=500)
# Verify: 90% credible interval contains observed data range
# If not: priors are too narrow or misspecified
```

### Step 2: Fit Model
```python
# Cross-validation runs (lighter sampling)
model.fit(X_train, y_train, draws=1000, tune=1500, chains=4, target_accept=0.97)

# Final fit (full sampling)
model.fit(X, y, draws=2000, tune=3000, chains=4, target_accept=0.99)
```

### Step 3: Posterior Predictive
```python
pp = model.sample_posterior_predictive(X, extend_idata=True)
# REMEMBER: Returns normalized [0,1] values -- scale back for metrics
```

### Step 4: Cross-Validation (expanding window)
```python
from pymc_marketing.mmm.time_slice_cross_validation import TimeSliceCrossValidator

cv = TimeSliceCrossValidator(
    n_init=80, forecast_horizon=13, step_size=13,
    date_column="date",
    sampler_config={"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.97},
)
```

## Seasonality Configuration

- `yearly_seasonality=8`: 8 Fourier modes (good default for weekly data)
- Higher modes capture sharper seasonal patterns but risk overfitting
- Always supplement Fourier with explicit indicator controls (quarterly dummies, holidays)
- Fourier terms alone may not capture sharp drops (Christmas dip, summer lull)

## Time-Varying Parameters (Advanced)

```python
model = MMM(
    ...,
    time_varying_intercept=True,  # HSGP smooth baseline trends
    time_varying_media=True,      # Channel effectiveness changes over time
)
```
- Use when you suspect structural changes in baseline or media effectiveness
- `ls_mu` controls smoothness: 6 (monthly), 13 (quarterly), 26 (semi-annual)
- Requires more data (104+ weeks recommended)

## Channel Grouping Rules

1. **No circularity:** Don't include SEM spend when modeling search volume (GQV) -- SEM responds to search, not vice versa
2. **Control for confounders:** When modeling SEM clicks, include search volume (GQV) as a control, not a channel
3. **Combine correlated channels:** If two channels always co-occur (e.g., FB + IG), combine them
4. **Match to target:** Only include channels that could plausibly affect the target variable
