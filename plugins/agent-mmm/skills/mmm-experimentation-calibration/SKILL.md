---
name: mmm-experimentation-calibration
description: |
  Designing incrementality experiments and folding their results into an MMM. Use when planning a geo holdout, matched-market test, ghost-ads or conversion-lift study, sizing an experiment, converting a lift result into a model constraint, calibrating with pymc-marketing add_lift_test_measurements, Meridian ROI priors, or Robyn calibration_input, or when reconciling an MMM with platform-reported or experimental numbers.
---

# Experimentation and Calibration

An MMM without an experiment is an argument from correlation. Several very different
attributions fit the same historical data equally well; the one the model reports is
whichever the priors preferred. An experiment breaks that tie with a measurement, and it
is the difference between a model you can defend and a model you can only display.

This is the highest-leverage work in MMM. One good geo test moves a channel's credible
interval more than any amount of model tuning.

---

## 1. Why observational MMM cannot settle incrementality

Three problems, none of which more data fixes:

* **Confounding.** Budgets rise when demand is expected to rise. The model sees spend and
  sales move together and cannot tell which caused which.
* **Collinearity.** Channels are planned together and flighted together. When two
  channels correlate at 0.9, the data does not contain the information needed to split
  their effects; the prior does the splitting.
* **Targeting selection.** Retargeting and brand search reach people who were already
  going to convert. Their observational contribution is real correlation and mostly not
  incremental.

An experiment creates variation that is *unrelated to demand* — that is the whole trick.
Everything else in this skill is bookkeeping.

---

## 2. Designs

| Design | How it works | Best for | Main limitation |
|---|---|---|---|
| **Geo holdout** | Switch a channel off in a random subset of geos | Any channel with geo-targetable buying | Needs enough comparable geos; spillover across borders |
| **Matched-market** | Pair similar markets, treat one of each pair | Few geos, or non-random assignment | Matching quality drives everything |
| **Synthetic control** | Build a weighted combination of control geos to mimic the treated one | A single large treated market | Needs a long pre-period; assumes the weights keep holding |
| **Ghost ads / PSA** | Log the users who *would* have seen the ad; compare against them | Auction-based platforms (Meta, Google, YouTube) | Platform-run; you cannot audit the counterfactual |
| **Conversion lift** | Platform-run randomised holdout | Same | Same, plus the platform grades its own homework |
| **Switchback** | Alternate on/off over time in the same market | Fast-acting channels only | Carryover contaminates adjacent periods — unusable for TV or OOH |
| **Budget-split / scaled** | Different spend levels across geos | Estimating the *curve*, not just one point | Needs several arms; more expensive |

**Prefer geo tests for MMM calibration.** They measure the same quantity the MMM
estimates (total incremental effect on the business outcome), at the same aggregation.
Platform lift studies measure a platform-defined conversion for platform-attributed users
— a different quantity, often on a different outcome, and not directly substitutable.

**Switchbacks and carryover do not mix.** If a channel's half-life is two weeks, an
on-week is contaminated by the previous off-week. Restrict switchbacks to channels whose
effect is essentially within-period, and even then use periods several half-lives long.

---

## 3. Designing a test that will actually resolve something

**Start from the decision, not the design.** "Should we keep spending £2m a year on
this?" needs a test that can distinguish a ROAS of 0.8 from 1.5. "How much should we
spend?" needs several spend levels, not an on/off test.

**Power.** The detectable lift scales with the noise in the outcome and the number of
geos. Roughly, for a two-arm geo test:

```
detectable_lift ≈ z * sd(geo-level outcome) * sqrt(2 / n_geos_per_arm)
```

The practical consequences:

* Halving the detectable lift needs four times the geos.
* Longer tests help through both more periods and more accumulated effect — but a test
  shorter than ~3 half-lives measures a truncated effect and understates the channel.
* Test a channel where you expect a *large* effect first. A null result on an
  underpowered test of a small channel tells you nothing and costs the same.

**Assignment.** Randomise geos, stratified by size and by baseline outcome. Do not let
the media team choose which markets to hold out — they will pick the ones they care about
least, which correlates with response.

**Pre-period.** Collect at least 8–12 weeks of pre-period data to verify parallel trends
and to fit a synthetic control if you need one.

**Contamination to plan for**: spillover between adjacent DMAs; national buys that cannot
be geo-suppressed; the treated market's sales team compensating; a concurrent promotion.

---

## 4. What to record

For an experiment to become a model constraint you need **all** of:

| Field | Meaning | Why it is required |
|---|---|---|
| `channel` | Which model channel it tested | Maps the constraint to a curve |
| `x` (`baseline_spend`) | Spend level the test started from | Locates the point on the saturation curve |
| `delta_x` (`spend_during_test`) | Incremental spend evaluated | The curve is pinned between `x` and `x + delta_x` |
| `delta_y` (`lift_absolute`) | Measured incremental target units | The measurement itself |
| `sigma` (`lift_se`) | Standard error of the lift | How hard the model should be pulled |
| dates | Test window | Scoping, and checking against spend data |
| design | How it was run | Credibility weighting |

**`sigma` is not optional and must not be invented.** It determines whether the
constraint is a nudge or a command. A test reported as "+8% lift, 90% CI [2%, 14%]"
converts to `sigma ≈ (upper − lower) / (2 × 1.645)` in absolute units. A test with no
uncertainty is not a measurement.

A result with a point estimate but no interval can still inform a **prior** (see §6) — it
just cannot be a likelihood term.

Common mistakes: recording percentage lift without the base (unusable); recording revenue
when the model's target is conversions (unit mismatch); recording the *total* spend rather
than the *incremental* spend the test evaluated.

---

## 5. Calibrating pymc-marketing

The model must be built first — the constraint attaches to the graph.

```python
mmm.build_model(X, y)

df_lift = pd.DataFrame([
    {"channel": "tv",     "x": 0.0,      "delta_x": 60_000.0, "delta_y": 1_500.0, "sigma": 400.0},
    {"channel": "social", "x": 40_000.0, "delta_x": 20_000.0, "delta_y":   380.0, "sigma": 150.0},
])
mmm.add_lift_test_measurements(df_lift)      # dist=pmd.Gamma by default
mmm.fit(X, y, draws=2000, tune=3000, chains=4, target_accept=0.95)
```

What it does:

```
model_estimated_lift = saturation(x + delta_x) - saturation(x)
observed:  dist(|model_estimated_lift|, sigma) ~ |delta_y|
```

It conditions the *shape of the response curve* on a measured point. Two consequences
worth understanding:

* It constrains saturation, not adstock. A test that ran for less than the carryover
  window measured a truncated effect, and the constraint will pull the curve down. Either
  run the test long enough, or adjust `delta_y` for the tail you did not observe.
* Units must match the model's scale. If your channel column is impressions, `x` and
  `delta_x` are impressions, not pounds. `pymc_marketing.mmm.lift_test.scale_lift_measurements`
  helps.

Add several tests with distinct `name=` arguments. Calibrate on ROAS or cost-per-target
instead with `mmm.add_cost_per_target_calibration(...)`.

One panel dimension column per entry in `mmm.dims` must appear in `df_lift` — a geo test
constrains the geos it ran in.

## Calibrating Meridian

Meridian parameterises media effects by ROI, so calibration means setting the prior:

```python
import numpy as np, tensorflow_probability as tfp
from meridian.model import prior_distribution, spec as model_spec
tfd = tfp.distributions

# 90% of the mass inside the test's confidence interval, per channel.
prior = prior_distribution.PriorDistribution(
    roi_m=tfd.LogNormal(roi_mu, roi_sigma, name="roi_m"),
)
spec = model_spec.ModelSpec(
    prior=prior,
    media_prior_type="roi",
    roi_calibration_period=calibration_mask,   # (n_media_times, n_media_channels) bool
)
```

`roi_calibration_period` restricts the ROI prior to the window the test actually covered,
which matters when a channel was only tested in one quarter. Meridian's default `roi_m`
is `LogNormal(0.2, 0.9)` — a median ROI near 1.2 with a 90% range of roughly 0.3–5.3.
Leaving that default in place on a channel you have measured throws the measurement away.

Helpers: `prior_distribution.lognormal_dist_from_range(low, high, mass_percent=0.9)` and
`lognormal_dist_from_mean_std(mean, std)`.

## Calibrating Robyn

```r
calibration_input <- data.frame(
  channel = c("tv_spend"),
  liftStartDate = as.Date("2025-03-03"),
  liftEndDate   = as.Date("2025-04-14"),
  liftAbs = 1500, spend = 60000, confidence = 0.9,
  metric = "revenue", calibration_scope = "immediate"
)
InputCollect <- robyn_inputs(InputCollect = InputCollect,
                             calibration_input = calibration_input)
```

Robyn adds MAPE-against-lift as a **third optimisation objective** alongside NRMSE and
DECOMP.RSSD. It therefore *prefers* models consistent with your test rather than
*requiring* them, which is a genuinely weaker constraint than a Bayesian likelihood term.
`confidence < 0.8` triggers a low-confidence warning.

---

## 6. When you have a number but not a usable measurement

Not every result can be a likelihood term. Fall back in this order:

1. **Point estimate + interval** → an informative ROI prior (Meridian directly;
   pymc-marketing via a tighter `saturation_beta` implied by the ROI).
2. **Point estimate only** → a prior centred there with a deliberately wide sigma, and a
   note that the width is a guess.
3. **A directional finding** ("clearly positive", "no detectable effect") → a sign
   constraint or a prior with most mass below 1.0.
4. **A platform-reported number** → treat as an upper bound, not a measurement. Platform
   attribution counts conversions it can see and claims users it targeted; both inflate.

Never convert a platform's last-click ROAS into a calibration constraint. It measures a
different quantity, and hard-coding it into an MMM launders an attribution artefact into
a causal claim.

---

## 7. Reconciling an MMM with an experiment

They will disagree. That is information, not a failure.

| Pattern | Likely explanation |
|---|---|
| MMM > experiment | Confounding: the model is crediting media with demand. Or the test window was shorter than the carryover, truncating the measured effect |
| MMM < experiment | The channel is under-identified (flat spend, collinearity) and the prior is pulling it down. Or the test period was unusually effective (creative, seasonality) |
| Both wide, overlapping | Consistent. Calibrate and move on |
| Both narrow, disjoint | Something is wrong with one of them. Check units, dates, and which outcome each measured, before touching the model |

**Check the boring things first.** In practice, most disagreements come from a unit
mismatch (revenue vs conversions), a date misalignment (invoice vs delivery), or the test
covering only part of the channel that the model treats as a whole.

---

## 8. Building a measurement programme

A single test calibrates one channel at one spend level at one point in time. What makes
a measurement practice is a cadence:

* **Prioritise by uncertainty × spend.** Test the channel where a wrong answer costs most,
  which is usually large-spend channels with wide posteriors, or always-on channels the
  model cannot identify at all.
* **Rotate.** Effects decay in relevance as creative, platforms and competitors change. A
  test from three years ago is weak evidence about today.
* **Test the curve, not just the point.** Multiple spend levels across geos identify
  saturation, which is what a budget optimiser actually needs. A single on/off test pins
  one point and leaves the shape to the prior.
* **Pre-register the analysis.** Decide the outcome, window and comparison before seeing
  results, or you are measuring your own flexibility.
* **Keep a register.** The spec's `experiments:` block is that register — channel, design,
  dates, lift, standard error, source. It feeds calibration automatically and survives
  staff turnover.

```yaml
experiments:
  - channel: tv_grps
    design: geo_holdout
    start_date: 2025-03-03
    end_date: 2025-04-14
    baseline_spend: 0
    spend_during_test: 60000
    lift_absolute: 1500
    lift_se: 400
    confidence: 0.9
    source: "Nielsen geo holdout, 20 DMAs, 10 treated"
    notes: "Ran 6 weeks vs a 3.5-week half-life; ~2 half-lives of tail unobserved."
```

`agent_mmm.fit_runner.run_fit` attaches every experiment with a complete measurement
(`lift_absolute`, `lift_se`, `spend_during_test`) as a lift-test constraint and records in
`metrics.json` which ones were used — and which were dropped, and why.

---

## 9. Reporting a calibrated model

Say plainly which channels are measured and which are inferred. They deserve different
confidence in the deck:

* *"TV and Meta are calibrated to geo holdouts run this year. Their ROAS ranges reflect
  both model and experimental uncertainty."*
* *"OOH and affiliate have never been tested. Their ranges are what the model can infer
  from historical variation, which for OOH is limited because spend is nearly constant.
  Treat them as directional."*

And state what would change your mind: the test that would most reduce uncertainty, what
it would cost, and what decision it would unblock.

---

## Related skills

`mmm-causal-design` for why observational identification fails; `mmm-channel-semantics`
for which channels most need testing; `mmm-validation` for refutation tests when no
experiment exists; `mmm-api-reference` §6 for the exact calibration API.
