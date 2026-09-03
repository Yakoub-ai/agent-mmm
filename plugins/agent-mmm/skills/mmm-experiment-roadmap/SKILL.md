---
name: mmm-experiment-roadmap
description: |
  Producing the experiment roadmap deliverable — which incrementality tests to run, in what order, what result each is expected to produce, how to execute it, and how it feeds back into the model. Use when asked what experiments to run, how to prioritise testing, whether a proposed test is adequately powered, how many geos or weeks are needed, or how much revenue a test puts at risk.
---

# The Experiment Roadmap

An MMM without an experiment is an argument from correlation. The roadmap is how that gets
fixed: a ranked, powered, costed sequence of tests, each with a pre-registered expectation
and a path back into the model.

---

## 1. The failure this prevents

Most media tests that get run cannot detect the effect they are looking for. The test
returns a null, the null is read as "the channel does nothing", and budget moves on
evidence that was never there.

This is knowable in advance, in an afternoon, from the number of geos and the noise in the
outcome. **An underpowered test is worse than no test**, because it manufactures false
confidence in the wrong direction. Say so, and say what would fix it.

---

## 2. Measure the noise before designing anything

```python
from agent_mmm.experiments import (
    estimate_geo_cv, estimate_pre_period_correlation, estimate_period_cv,
)

cv  = estimate_geo_cv(panel, geo_column="geo", target_column="revenue")
rho = estimate_pre_period_correlation(panel, geo_column="geo",
                                      target_column="revenue", date_column="date")
```

`rho` — the correlation between each geo's pre-period and test-period outcome — is the
single most important input to geo-test power, and it is measurable from the panel you
already have. Market sizes are persistent, so it is usually above 0.9.

**This is why the pre-period is not optional.** Adjusting for it (CUPED, or a
difference-in-differences against a matched pre-period) shrinks the residual noise by
`sqrt(1 - rho²)`. At rho = 0.9 that more than halves the detectable effect, for free, and
it is usually worth far more than adding geos you do not have. It also cannot be
reconstructed after the fact: the pre-period has to be recorded before the buy changes.

---

## 3. Power

```python
from agent_mmm.experiments import geo_lift_mde, required_geos_for_mde, holdout_mde

mde = geo_lift_mde(n_treatment=20, n_control=20, cv=cv,
                   pre_period_correlation=rho, duration_periods=8)
```

**Geo designs.** The detectable effect scales with `sqrt(1/n_t + 1/n_c)`, so doubling the
geos buys about a 30% reduction — real, but expensive, and usually the least available
lever.

**Duration helps, but less than `1/sqrt(n)`.** Residuals are autocorrelated: a market
running hot in week one tends to still be running hot in week two. Under AR(1)-like
correlation the effective sample size is `n / (1 + (n-1)φ)`. At φ = 0.3, eight weeks buys
a 38% noise reduction rather than the 65% an independence assumption would promise.
Designing on independence produces a test quietly about half as powerful as its plan said.

**One-sided is the default.** Media tests ask "did it help". Insisting on two-sided
inflates the required effect by about 15% for no gain in what the business decides.

**Time-based holdouts need a pre-period.** Without a comparison window nothing is estimable
at any sample size — that is a design error, not an underpowered design. If the comparison
is meant to come from unaffected markets or products, that is a synthetic control, which is
a different and better design.

---

## 4. The expected effect, and the carryover discount

```python
from agent_mmm.experiments import expected_lift_from_share

expected = expected_lift_from_share(contribution_share=0.12, holdout_fraction=1.0)
```

If a channel is believed to drive 12% of sales and all of it is switched off, a perfect
test would show a 12% drop — except that **carryover from before the test keeps working
during it**, so a short window measures less than the steady-state effect. Ignoring this is
a standard way to arrive at a test designed 15-30% underpowered.

Compare the expected effect to the detectable effect. If expected < detectable, the test
is blocked and the report must say what would unblock it: more geos, a longer window, a
larger holdout share, or a better pre-period covariate.

---

## 5. Prioritisation

**Priority = money at risk × how little we know × how cleanly it can be tested.**

* **Money at risk** — the channel's share of spend. Testing a 2% channel is rarely worth a
  quarter of measurement capacity.
* **How little we know** — the width of the posterior interval relative to the point
  estimate. A channel already measured precisely is not worth testing however large it is;
  a channel whose interval spans 0.5 to 5.0 has not been measured at all. Before a model
  exists, always-on channels are the least identified by default.
* **How cleanly it can be tested** — a channel that cannot be bought by geo is worth less
  because only a much weaker design is available. A channel whose switch-off carries brand
  risk gets blocked organisationally more often than it fails statistically.
* **A prior experiment lowers priority.** A second test on an anchored channel buys less
  than a first test on an unanchored one.

This ranking often puts **brand search first**: large, always-on, almost never varied, and
the channel where the model and reality most often disagree.

---

## 6. Building the roadmap

```python
from agent_mmm.experiments import ChannelTestCandidate, build_roadmap, render_roadmap

roadmap = build_roadmap(candidates, n_geos_available=40, geo_cv=cv,
                        pre_period_correlation=rho, duration_periods=8)
```

**Sequence, do not parallelise.** Two geo tests running at once in overlapping markets
contaminate each other's control arm. Raise `max_concurrent` only when the tests use
disjoint geo sets.

---

## 7. What every plan must contain

**A pre-registered expectation.** What the current model predicts, written down *before*
the test runs. A test with no prediction cannot be surprising, and therefore teaches
nothing. If no model has been fitted, say so rather than inventing an expectation.

**What would change our mind, in both directions.** A result near zero says the channel is
less incremental than the model believes. A result well above the expectation says the
model is crediting some of this channel's effect elsewhere — usually to a channel it
correlates with, or to the baseline.

**Execution detail specific enough to hand over.**

1. Stratify on outcome level *and* on the channel's own spend, then randomise within
   strata. Hand-picking markets is how a test gets an answer somebody already wanted.
2. Record the pre-period before touching the buy.
3. Suppress the channel and **change nothing else** — no creative refresh, no promotion, no
   budget shifted into other channels to "compensate". Compensating spend is the single
   most common way a media test is destroyed.
4. Verify suppression in flight: pull delivery by geo in week one. Partial suppression
   looks exactly like a weak effect.
5. Hold for the full window even if early numbers look bad. Stopping early on a noisy read
   is how a test becomes a story.
6. Analyse as difference-in-differences or CUPED against the pre-period. **Report the lift
   with its standard error**, not a p-value alone — the MMM needs the uncertainty to
   calibrate against, and a point estimate with no error bar cannot be used.

**Risks named.** Spillover into control markets from adjacent geography and national media;
carryover inside the window; concurrent promotions or competitor action in one arm.

**The calibration path.** `add_lift_test_measurements` in pymc-marketing, ROI priors in
Meridian, `calibration_input` in Robyn. After refitting, check both that the channel moved
towards the test *and* that the other channels did not quietly absorb the difference — the
contributions have to add up, so constraining one channel pushes on all of them.

---

## 8. Questions that shape the roadmap more than the statistics do

* **What is the maximum revenue the business will accept putting at risk, and who signs it
  off?** Every design trades revenue for information, and that trade is a business
  decision.
* **Which calendar periods are off-limits?** Peak trading, a launch, a seasonal spike. This
  usually constrains the roadmap more than power does.
* **Who owns execution, and can they actually suppress by geo?** Some buying platforms and
  some agency contracts make it impossible.
* **What decision will the result change?** A test whose result changes nothing should not
  run, however clean the design.

---

## 9. When there is no geo panel

* **Matched market** — pick pairs of similar markets and treat one of each. Weaker, but
  workable with as few as ten pairs.
* **Switchback** — alternate on and off in time. Only for fast-response channels; useless
  where carryover is long.
* **Ghost ads / PSA control** — platform-native, clean, but only available on some
  platforms and only for that platform's inventory.
* **Synthetic control** — build a counterfactual from unaffected products, regions or
  categories. Better than naive pre/post, and the right fallback for a national channel.
* **A natural experiment already in the data** — an unplanned dark period, a budget freeze,
  a stockout, a platform outage. These are free and routinely overlooked. Ask what went
  wrong last year.

---

## Related skills

`mmm-experimentation-calibration` for design theory and per-framework calibration mechanics.
`mmm-causal-design` for what a test actually identifies.
`mmm-multi-geo-panel` for the panel structure geo designs depend on.
`mmm-validation` for where experiments sit in the hierarchy of evidence.
