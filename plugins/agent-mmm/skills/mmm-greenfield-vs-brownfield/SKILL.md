---
name: mmm-greenfield-vs-brownfield
description: |
  Greenfield vs brownfield MMM decision guide. Use when the user asks about starting a new MMM from scratch vs improving an existing one, or when designing the modeling strategy for a company with prior MMM experience.
---

# Greenfield vs Brownfield MMM

## Definitions

**Greenfield**: Building an MMM from scratch with no prior fitted model. All priors are data-driven estimates from the channel type catalog. Wider priors to let the data speak.

**Brownfield**: Improving or auditing an existing MMM. A prior fitted model (InferenceData `.nc` file) is available. Posteriors from the previous run can be used to tighten priors for the next run.

## Decision Guide

| Signal | Likely Answer |
|--------|--------------|
| First time running MMM | Greenfield |
| Have an existing `.nc` idata file | Brownfield |
| Previous model had convergence issues | Brownfield (refit from better priors) |
| Changing channels or target variable | Greenfield (different model structure) |
| Same model, different time period | Brownfield |
| Different agency, same company | May be Brownfield if prior model spec is available |

## Greenfield Strategy

1. Wide priors from `prior_catalog.yaml` channel type defaults
2. Widen further if data is sparse (`data_quality_tier = WARN`)
3. Prior predictive check: 90% interval should contain observed target range
4. Tournament across `l_max`, saturation type, Fourier modes
5. Final fit with `SAMPLER_FINAL` config (draws=2000, tune=3000, chains=4, target_accept=0.99)

## Brownfield Strategy

1. Load existing `idata.nc` → extract posterior summary stats (mean, std per parameter)
2. Use posterior mean as new prior mu, `posterior_std * 0.7` as new sigma (tightening factor)
3. Apply moment-matching to convert (mu, sigma) back to distribution parameters
4. This is also what `/mmm-improve` does in the posterior-informed refinement step
5. Key check: are the new tightened priors still scientifically plausible? Don't over-tighten.

## Prior Tightening Code (Brownfield)

```python
from agent_mmm.iter_loop import tighten_priors_from_posterior
import arviz as az

idata = az.from_netcdf("mmm-workspace/runs/best-run/idata.nc")
posterior_stats = {
    var: {"mean": float(idata.posterior[var].mean()), "std": float(idata.posterior[var].std())}
    for var in ["adstock_alpha", "saturation_lam"]
    if var in idata.posterior
}
tight_config = tighten_priors_from_posterior(existing_model_config, posterior_stats, tighten_factor=0.7)
```

## Series A/B/C Funnel Design

For comprehensive marketing analysis, run independent models per funnel stage. Each uses its own `spec.yaml`:

| Series | Target | Channels | Notes |
|--------|--------|---------|-------|
| A (Sales) | Revenue or conversions | All media spend | Primary business outcome |
| B (Awareness) | Search volume (GQV) | Non-SEM channels only | SEM excluded — it's circular |
| C (SEM Funnel) | SEM clicks | Non-SEM channels | Include search volume as control |

A channel's true ROI can be understated in Series A if it works through awareness intermediation (Meta → search interest → SEM clicks → sales). Series B/C models reveal this indirect effect.
