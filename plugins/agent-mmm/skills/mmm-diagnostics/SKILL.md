---
name: mmm-diagnostics
description: |
  Diagnosing a fitted MMM — convergence (r-hat, ESS, divergences, BFMI), fit quality, overfitting, baseline health, prior-to-posterior learning, and attribution plausibility. Use when checking whether a model is trustworthy, debugging sampling failures, interpreting ArviZ 1.x output, deciding whether results are safe to report, or working through why a model will not converge or fits badly.
---

# Diagnosing an MMM

Read the checks in this order. An answer at a later stage is meaningless if an earlier one
failed — interpreting ROAS from a model with r-hat 1.3 is reading tea leaves, however good
the R² looks.

```
1. Convergence   — did the sampler explore the posterior at all?
2. Fit           — does it reproduce its training data?
3. Generalisation— does it reproduce data it has not seen?
4. Baseline      — is the decomposition structurally sane?
5. Learning      — did the data move the priors?
6. Plausibility  — do the channel effects survive contact with what we know?
```

```python
from agent_mmm.diagnostics import run_diagnostics
findings = run_diagnostics("run-id", idata=mmm.idata, spend_totals=totals)
```

Written for ArviZ >= 1.2, where inference data is an `xarray.DataTree` and `az.waic` no
longer exists.

---

## 1. Convergence

| Check | Target | Fail | Read as |
|---|---|---|---|
| r-hat | ≤ 1.01 | > 1.05 | Chains agree on where the posterior is |
| ESS bulk | ≥ 400 (≥100/chain) | < 200 | Enough independent samples for the mean |
| ESS tail | ≥ 400 | < 200 | Enough for the interval edges |
| Divergences | 0 | > 0 | Sampler could reach every region |
| BFMI | ≥ 0.2 | < 0.2 | Energy distribution explored properly |

```python
import arviz as az
s = az.summary(mmm.idata)   # columns: mean, sd, eti89_lb, eti89_ub, ess_bulk, ess_tail, r_hat, mcse_*
print(s["r_hat"].max(), s["ess_bulk"].min())
print(int(mmm.idata["sample_stats"]["diverging"].sum()))
```

**r-hat 1.01, not 1.05.** The rank-normalised threshold (Vehtari et al. 2021) is 1.01, and
it is what PyMC warns at. The older 1.05 convention is too loose for a model whose output
moves budget.

**Divergences are bias, not noise.** They mean the sampler could not traverse part of the
posterior, so the samples systematically miss a region. Zero is the only acceptable count.

**Look at *which* parameter fails.** A single channel's `saturation_beta` failing is a
different problem from everything failing: it usually means that channel is not
identified, and the fix is in the specification, not the sampler.

### Debugging

```
Divergences
  1. target_accept 0.9 -> 0.95 -> 0.99
  2. Still diverging? The geometry is the problem, not the step size:
     - a channel with almost no spend variation creates a funnel
       (beta and lam trade off with nothing to pin them)
     - a prior in conflict with the data
     - a lift-test constraint the model cannot satisfy
  3. Fix the specification: group collinear channels, drop unidentified ones,
     widen or re-centre the offending prior.

High r-hat
  1. More tune (3000+). Adaptation, not sampling, is usually the shortfall.
  2. az.plot_trace on the worst parameter: multimodal? chains in different places?
  3. Multimodality in MMM usually means two channels can swap roles — a
     collinearity problem with a sampling symptom.

Low ESS
  1. More draws.
  2. If ESS is low only for one parameter, check its autocorrelation:
     high autocorrelation with good r-hat means a badly scaled prior.

Sampling is glacial
  - l_max is large (convolution cost scales with it) — but do not truncate a real tail
  - too many channels for the data
  - target_accept at 0.99 by default: raise it in response to divergences, not as a habit
```

The general rule: **most convergence problems are identifiability problems.** If the data
cannot distinguish two channels, no sampler setting will make it do so.

---

## 2. Fit

```python
from agent_mmm.fit_runner import compute_insample_metrics
compute_insample_metrics(mmm, y, spec)   # R², MAPE, wMAPE — per draw, with intervals
```

**Compute metrics per posterior draw, then summarise.** `E[f(x)] ≠ f(E[x])`: R² of the
posterior mean cancels noise the model never actually removed and is optimistic. Per-draw
metrics on the posterior *predictive* include observation noise, so they are lower — and
they are the honest number. Compare like with like across runs.

| Metric | Reasonable | Read as |
|---|---|---|
| In-sample R² | 0.75–0.95 | Below 0.5: something systematic is missing. Above 0.98: suspect leakage |
| MAPE | < 15% | Sensitive to small target values |
| wMAPE | < 15% | More robust; prefer it |

**R² above 0.98 is a warning, not a triumph.** Look for a control that is a proxy for the
target — lagged sales, year-on-year sales, a "baseline forecast" derived from the target.
A near-perfect MMM is almost always leaking.

---

## 3. Generalisation

```python
overfit_gap = in_sample_r2 - cv_r2
```

| Gap | Read as |
|---|---|
| < 0.05 | Good generalisation |
| 0.05–0.20 | Acceptable |
| > 0.20 | Memorised. Tighten priors, cut Fourier order, merge collinear channels |

**No CV means generalisation is unknown.** A model that has never been asked to predict an
unseen week has not been validated. Parameter stability across folds
(`mmm.plot.param_stability()`) matters as much as the R² level — a model can predict well
while its attribution swings between folds, and attribution is what you are selling. See
`mmm-validation`.

---

## 4. Baseline health

```python
from agent_mmm.diagnostics import decompose, check_baseline
dec = decompose(mmm.idata)
print(dec["shares"])                       # intercept / trend / seasonality / controls / media
print(check_baseline(mmm.idata, decomposition=dec))
```

| Check | Healthy | Failure means |
|---|---|---|
| Negative periods | 0 | The model claims the business would sell less than nothing without marketing. Every channel number is inflated |
| Baseline share | 50–90% | Below 30%: an omitted driver is being credited to media. Above 95%: the baseline is eating media |
| Drift start→end | within ±50% | A missing structural variable |

**Check this before any ROAS.** It is the fastest way to find a broken model, and the check
most often skipped. See `mmm-baseline-and-trend` for what to do about each failure.

---

## 5. Did the data teach us anything?

```python
from agent_mmm.diagnostics import check_prior_contraction
check_prior_contraction(mmm.idata)     # needs the prior group
```

Contraction = `1 - sd(posterior)/sd(prior)`.

| Contraction | Read as |
|---|---|
| > 0.5 | The data determined this parameter |
| 0.2–0.5 | Partially informed |
| < 0.2 | **Prior-dominated** — the posterior restates your assumption |

Run `sample_prior_predictive` before fitting or this check is unavailable. It is the honest
version of "prior pull": a boundary heuristic guesses, contraction measures.

A prior-dominated parameter is not a bug — some channels genuinely cannot be identified
from flat, always-on spend. It is a reporting obligation. Presenting a prior-dominated
ROAS as a finding is how an assumption becomes a budget decision.

---

## 6. Plausibility

| Check | Threshold | Failure usually means |
|---|---|---|
| Media share of target | < 60% | Starved baseline; missing price or distribution |
| Single channel share | < 70% (≥3 channels) | Collinearity — check VIF and the spend correlation matrix |
| Negative contributions | none | Spend dated by invoice not delivery, or collinearity with an opposite-signed variable |
| Implied ROAS/CPA | matches business knowledge by order of magnitude | A scale error, usually un-rescaled contributions |
| Contributions sum to target | exact (identity link) | Normalisation, un-broadcast intercept, or a log link |

```python
from agent_mmm.diagnostics import check_attribution_plausibility
check_attribution_plausibility(mmm.idata, spend_totals={"tv": 1_200_000, "sem": 800_000})
```

`spend_totals` must be in the target's own units — contributions are rescaled by
`target_scale` first so both sides of the ratio agree.

---

## 7. Model comparison

```python
import arviz as az
az.compare({"baseline": idata_a, "with_trend": idata_b})
```

`az.waic` was removed in ArviZ 1.x. LOO-PSIS is better anyway because Pareto-k tells you
when the approximation itself is unreliable.

Two cautions. LOO on a time series is optimistic — it leaves out single points from an
autocorrelated series, so a model that memorises local structure scores well. And a better
predictive score does not mean better attribution: a flexible baseline usually wins on LOO
while attributing less to media, so selecting on LOO alone systematically biases you
towards models that say marketing does nothing. Use it as a tiebreak between similar
models, never as a substitute for time-slice CV.

---

## 8. Plots worth actually looking at

```python
mmm.plot.posterior_predictive()             # does it track the series, and where does it miss?
mmm.plot.residuals_over_time()              # structure in residuals = a missing variable
mmm.plot.prior_vs_posterior(var="adstock_alpha")
mmm.plot.saturation_curves_scatter()        # where does the observed data stop?
mmm.plot.waterfall_components_decomposition()
mmm.plot.param_stability()                  # across CV folds
az.plot_trace(mmm.idata, var_names=["saturation_beta"])
az.plot_energy(mmm.idata)
```

`residuals_over_time` is the highest-yield plot. Structure in the residuals — a run of
positive errors, a seasonal wave, a step — is a variable you have not included, and it tells
you *when* to look for it.

`saturation_curves_scatter` is the second: it shows where the observed spend stops, and
therefore where the curve becomes extrapolation.

---

## 9. Quick triage

```
Won't converge
  divergences -> target_accept, then the specification (identifiability)
  r-hat high  -> more tune, then check for multimodality
  low ESS     -> more draws, then check prior scaling

Converges but fits badly
  R² < 0.5    -> missing control, structural break, wrong target
  residuals show structure -> the plot tells you when; go find what happened
  CV gap > 0.2 -> too many parameters for the data

Fits but the answer is wrong
  baseline negative or tiny -> missing driver (price, distribution, trend)
  one channel dominates     -> collinearity
  channel negative          -> date alignment, then collinearity
  prior-dominated           -> not identified; report as an assumption, and test it
  media share > 60%         -> baseline starved
```

---

## 10. What to report

The DS report should say, in plain terms:

* Convergence status and any parameter that failed.
* In-sample and out-of-sample fit, and the gap.
* Baseline share and whether it ever went negative.
* Which parameters were prior-dominated.
* Which channels are calibrated to an experiment and which are inferred.
* What would change the answer.

That list is what tells a reader how far to trust each number, which is the only thing a
diagnostics section is for.

---

## Library

```python
from agent_mmm.diagnostics import (
    run_diagnostics, decompose, check_convergence, check_baseline,
    check_prior_contraction, check_attribution_plausibility, compare_models,
)
```

Thresholds: `RHAT_TARGET=1.01`, `RHAT_FAIL=1.05`, `ESS_THRESHOLD=400`,
`OVERFIT_GAP_THRESHOLD=0.20`, `CONTRACTION_MIN=0.20`, `BFMI_THRESHOLD=0.2`.

Slash command: `/mmm-diagnose`.
