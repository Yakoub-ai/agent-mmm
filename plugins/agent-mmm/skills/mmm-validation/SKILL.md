---
name: mmm-validation
description: |
  Validating that an MMM is trustworthy, not merely well-fitting — time-series cross-validation, holdouts, refutation tests, parameter recovery on simulated data, stability across refreshes, and sensitivity to specification. Use when deciding whether a model is fit to inform budget decisions, designing a validation plan, running placebo or subset refutation tests, checking whether conclusions survive reasonable alternative specifications, or explaining why R-squared is not validation.
---

# Validating an MMM

Fit quality is not validation. A model with R² 0.95 can be wrong about every channel,
because MMM's decisive parameters — the split of credit between correlated channels, the
baseline/media boundary, the shape of the saturation curve outside observed spend — are
weakly constrained by fit. Two models with identical predictive accuracy can recommend
opposite budgets.

Validation asks a different question: **would this model's recommendation still be right
if the world were slightly different from the sample we happened to see?**

---

## 1. Five levels, weakest to strongest

| Level | What it tests | What it cannot tell you |
|---|---|---|
| 1. In-sample fit | Can the model reproduce its training data | Nothing about generalisation or causality |
| 2. Out-of-sample prediction | Does it forecast unseen periods | Whether the *decomposition* is right — a model can predict well with wrong attribution |
| 3. Stability | Do conclusions survive resampling, refreshes, reasonable respecification | Whether they are right, only whether they are stable |
| 4. Refutation | Does it correctly fail tests it should fail | Whether the surviving estimate is the true one |
| 5. Experiment | Does the estimate match a randomised measurement | Only for the channel, spend level and period tested |

Levels 1–2 are table stakes. **Level 3 is where most MMM reviews should spend their
effort and usually do not.** Level 5 is the only one that establishes causality, which is
why the experimentation skill exists.

---

## 2. Time-series cross-validation

Random k-fold is invalid here: it trains on the future to predict the past, and adstock
leaks across fold boundaries. Use expanding-window slices.

```python
from pymc_marketing.mmm import TimeSliceCrossValidator

cv = TimeSliceCrossValidator(
    n_init=80, forecast_horizon=13, step_size=13, date_column="date",
    sampler_config={"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.95},
)
class Builder:
    def build_model(self, X, y): return make_mmm()
combined = cv.run(X, y, mmm=Builder())
```

```
Fold 1: train[0:80]   test[80:93]
Fold 2: train[0:93]   test[93:106]
Fold 3: train[0:106]  test[106:119]
```

Read three things, not one:

* **Level.** CV R² above ~0.6 is a reasonable bar for weekly data.
* **The gap.** `in_sample_R² − CV_R²` above 0.20 means memorisation. Tighten priors,
  reduce Fourier order, merge collinear channels.
* **Parameter stability across folds** (`mmm.plot.param_stability()`). This is the part
  people skip and the part that matters: if a channel's adstock or beta swings wildly
  between folds, its estimate is a function of which weeks you happened to include. The
  point forecast can still look fine while the attribution is unusable.

Also look at **CRPS** (`mmm.plot.cv_crps()`), which scores the whole predictive
distribution rather than the mean, and rewards honest uncertainty.

**Costs**: n folds means n fits. Use a lighter sampler for CV and the full one for the
final model.

---

## 3. Holdout

A single held-out tail (the last 13–26 weeks) is cheaper than full CV and answers a
narrower question. Useful as a gate before an expensive CV run.

Meridian supports this natively via `holdout_id` — a boolean mask over (geo, time) that
excludes KPI from training while **keeping media in**, so adstock still flows correctly
into the held-out period. That detail matters: naively truncating the data zeroes the
carryover entering the holdout and makes the model look worse than it is.

Robyn's `ts_validation = TRUE` holds out a tail and searches `train_size` as a
hyperparameter.

---

## 4. Refutation tests

Borrowed from causal inference: deliberately break something and check the model responds
correctly. A model that passes a test it should fail is not validated, it is unfalsifiable.

| Test | Procedure | Pass condition |
|---|---|---|
| **Placebo treatment** | Replace a channel's spend with random noise of the same distribution | Its effect collapses to ~0 |
| **Random common cause** | Add an irrelevant random control | Channel estimates barely move |
| **Data subset** | Refit on random 80% subsets (respecting time order) | Estimates stay within their credible intervals |
| **Unobserved confounder** | Simulate a confounder correlated with spend and target | Quantifies how much bias a plausible omitted variable could cause |
| **Add-noise** | Add measurement noise to a channel | Its effect attenuates smoothly, not erratically |
| **Time shift** | Shift one channel's spend by a few periods | Its effect drops; if it does not, the model is fitting something other than that channel |

The time-shift test is the most informative and least used. If shifting TV spend by three
weeks leaves its estimate unchanged, the model is not identifying TV — it is identifying
something TV correlates with, most often seasonality.

**Placebo test sketch:**

```python
X_placebo = X.copy()
rng = np.random.default_rng(0)
X_placebo["tv_spend"] = rng.permutation(X["tv_spend"].values)   # keeps the marginal, breaks the timing
mmm_placebo = make_mmm(); mmm_placebo.fit(X_placebo, y, ...)
# The placebo channel's contribution should be indistinguishable from zero.
```

---

## 5. Parameter recovery on simulated data

The only way to check the *machinery* rather than the *model*: generate data from known
parameters, fit, and see whether the true values fall inside the posterior.

```
1. Choose true alpha, lam, beta per channel, plus a baseline and seasonality.
2. Simulate spend with realistic flighting and realistic correlation between channels.
3. Generate the target through the same adstock/saturation the model will use.
4. Fit the model.
5. Check the true values against the posterior intervals.
```

What this catches that nothing else does:

* **Sign and scale errors** in your own pipeline.
* **How much collinearity your setup can survive.** Simulate two channels correlated at
  0.9 and see whether recovery fails. It usually does — which tells you what your real
  model can and cannot resolve.
* **Whether your priors are strong enough to matter**, by simulating from a truth that
  disagrees with them.
* **Whether an experiment would help**, by adding a simulated lift-test constraint and
  watching the interval shrink.

Simulation-based calibration is the rigorous version: repeat over many prior draws and
check that the true value's rank within the posterior is uniform. Expensive, but the only
way to prove a model implementation is correct.

---

## 6. Stability

**Across refreshes.** Fit the quarterly refresh and compare channel contributions and
ROAS to last quarter. Large swings with only 13 weeks of new data mean the model is not
identified — small perturbations are moving a knife-edge split. Stakeholders will notice
this before you do, and it destroys trust faster than a wide interval.

**Across specifications.** Fit a small set of reasonable alternatives and compare the
*conclusions*, not the fit:

* Fourier order 4 vs 8
* Fixed intercept vs time-varying
* Channels grouped vs split
* Logistic vs Hill saturation
* Normal vs StudentT likelihood

If "increase TV, cut affiliate" survives all of them, it is a robust recommendation. If
TV's ROAS ranges from 1.2 to 4.0 across specifications, the honest deliverable is that
range plus a test proposal — not the number from your favourite specification.

This is the single most useful thing to do before a readout, and the thing most likely to
be skipped because it produces an uncomfortable answer.

---

## 7. Model comparison

```python
import arviz as az
az.compare({"baseline": idata_a, "with_trend": idata_b})   # az.waic was removed in ArviZ 1.x
```

LOO on a time series is optimistic — it leaves out single points from an autocorrelated
series, so a model that memorises local structure scores well. Use it as a tiebreak
between similar models, never as a substitute for time-slice CV. Check Pareto-k values;
high k means the approximation itself is unreliable.

And remember that better predictive score does not mean better attribution. A model with
a flexible baseline will usually win on LOO while attributing less to media. Choosing on
LOO alone systematically biases you towards models that say marketing does nothing.

---

## 8. Sanity checks that are not statistics

Cheap, fast, and they catch more real errors than any of the above:

* **Do contributions reconstruct the target?** Under an identity link they must sum
  exactly. Under a log link they do not, for a known reason.
* **Is the baseline ever negative?** Blocking failure. See the baseline skill.
* **Is any channel negative?** Media effects should not be.
* **Does one channel take >70% of media effect with three or more channels?** Collinearity
  signature.
* **Are the implied CPAs plausible against what the business knows?** An implied CPA a
  tenth of the known blended CPA is a scale error somewhere.
* **Does the marketing team recognise the story?** Not decisive — the whole point may be
  to overturn a belief — but a model that contradicts everything everyone knows is more
  often broken than revolutionary.

---

## 9. A validation plan proportional to the decision

| Decision at stake | Minimum validation |
|---|---|
| Exploratory, directional | Convergence + in-sample fit + baseline sanity |
| Tactical reallocation within a channel | + time-slice CV, parameter stability |
| Annual budget setting | + specification sensitivity, refutation tests, refresh stability |
| Board-level or contractual | + at least one calibrating experiment per major channel |

Validation cost should scale with the size of the decision. A quarterly reallocation of
5% of budget does not need simulation-based calibration; a three-year investment case
does not deserve less than an experiment.

---

## 10. Reporting validation honestly

State what was validated and what was not:

> *Time-slice CV over 4 folds: mean out-of-sample R² 0.71 against 0.84 in-sample (gap
> 0.13). Adstock and saturation parameters were stable across folds for TV, Meta and
> search; OOH's estimate moved by more than its interval between folds and should be read
> as directional. Conclusions were checked against six alternative specifications: the
> recommendation to shift budget from affiliate to prospecting held in all six; TV's ROAS
> ranged from 1.6 to 2.4 across them. TV and Meta are calibrated to geo holdouts; no other
> channel has been tested.*

That paragraph is worth more than three pages of charts. It tells a decision-maker exactly
how far to trust each number, which is the only thing they actually need from a validation
section.

---

## Related skills

`mmm-diagnostics` for convergence and the automated checks;
`mmm-experimentation-calibration` for level 5; `mmm-causal-design` for the assumptions
refutation tests probe; `mmm-iterative-improvement` for the tournament that should be
scored on CV, not fit.
