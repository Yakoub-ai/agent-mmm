---
name: mmm-data-quality
description: |
  Assessing whether a dataset can support an MMM, and interpreting the audit that says so. Use when evaluating data readiness, checking minimum requirements, reading audit findings, diagnosing collinearity, sparse or flat channels, insufficient history, panel problems, or deciding whether to model at all with the data available.
---

# Data Quality for MMM

The audit's job is to answer one question: **can this data identify the effects we are
going to claim to have measured?** Not "is it clean" — clean data can be completely
uninformative, and messy data can be perfectly adequate.

```bash
/mmm-analyze-data
```
```python
from agent_mmm.data_audit import run_audit
findings = run_audit(spec)      # -> mmm-workspace/audit/audit_report.md
```

The audit runs in six groups, ordered so a failure early makes later checks moot:
**contract → shape → integrity → identifiability → signal → semantics.**

---

## 1. Minimum requirements

| Requirement | Hard floor | Comfortable | Why |
|---|---|---|---|
| Periods | 52 | 104+ | One cycle lets you fit seasonality; two let you validate it |
| Observations per parameter | 5 | 10+ | Below this the priors are answering, not the data |
| Channels | 1 | 3–8 | More channels, more parameters, no more data |
| Granularity | monthly | **weekly** | Daily is day-of-week noise; monthly cannot identify carryover |
| Geos (panel) | 5 | 20+ | Pooling needs enough units to estimate between-geo variance |

**The parameter budget is the check people skip.** Each channel costs roughly three
parameters (decay, saturation steepness, scale) and brings no new observations. 104 weekly
observations with 10 channels and 8 Fourier modes is about 3 observations per parameter.
The model will still fit, converge, and produce confident-looking numbers — mostly
reflecting the priors.

A geo panel is the cheapest way out: 20 geos × 104 weeks is 2,080 observations, and
cross-sectional variation in media weight is real information.

---

## 2. Contract

Every column the spec names must exist, and be numeric.

Non-numeric columns are usually currency symbols, thousands separators, or `"NULL"`
strings. Fix them at source. `pd.to_numeric(errors="coerce")` turns unparseable rows into
NaN, which then becomes zero in a careless fill — inventing periods of no spend.

The audit also lists **numeric columns present in the file but absent from the spec**.
Check each one: an omitted driver does not stay omitted, it gets attributed to whatever
correlates with it.

---

## 3. Shape

* **Dense calendar.** Adstock convolves over consecutive positions, so a missing week
  silently shortens carryover for everything after it.
* **Declared granularity matches observed spacing.** Every half-life, `l_max` and carryover
  prior is expressed in periods; a wrong granularity rescales all of them at once.
* **No duplicate keys.** Duplicated periods double-count both spend and target.
* **Rectangular panel.** Every (geo, date) cell must exist. Neither pymc-marketing nor
  Meridian will impute one.

`agent_mmm.data_prep.reindex_complete()` inserts missing cells as **NaN, not zero** — what
they mean is a decision, not a default.

---

## 4. Integrity

| Finding | What it means | Response |
|---|---|---|
| Missing spend | Usually genuinely zero | Fill with 0 |
| Missing target | Period unusable | Drop the row; never fill a target |
| Missing price/distribution | State persisted | Forward-fill |
| Negative spend | A credit posted to the wrong period | Reallocate to the original period, do not clip |
| Target outliers (|z| > 4) | A promotion, a stockout, an error, or a real spike | Explain each one before deciding |
| Structural break | A relaunch, pricing change, pandemic, tracking migration | Indicator, split, or time-varying intercept |
| Skewed target | Heavy tails | StudentT likelihood or a log link — not deletion |

**Never blanket-`fillna(0)`.** Zeroing a missing target invents a period of no sales;
zeroing a missing price invents a giveaway. `suggest_imputation(spec, df)` proposes a rule
per column from the declared roles; review it before applying.

**Flag outliers, do not delete them.** A spike is a data point with an explanation
attached. Deleting it discards both, and biases the model towards the ordinary — the
regime a budget optimiser will push away from.

---

## 5. Identifiability — the part that decides whether the answer means anything

### Collinearity

| Signal | Threshold | Consequence |
|---|---|---|
| Pairwise correlation | \|r\| > 0.8 | The model cannot separate the two; the split is the prior's opinion |
| VIF | > 5 warn, > 10 critical | Coefficient unstable; small data changes swing the ROAS |

**Collinearity is not a nuisance to be regularised away.** It means the information you
need is not in the data. Options, in order:

1. **Group** the channels that always move together and report one honest number.
2. **Break the correlation** with deliberately different flighting, or geo variation.
3. **Experiment** — the only way to create variation independent of the media plan.
4. Widen the priors and say the split is an assumption.

### Flat and constant channels

A constant column is perfectly collinear with the intercept and carries no information at
all. A near-constant one (CV < 0.15) is barely better: its coefficient is set by the prior,
and the saturation curve near zero — which is exactly where a budget optimiser will look
when it considers cutting the channel — is pure extrapolation.

Always-on channels (brand search, long-running affiliate) sit here permanently. They need
an experiment, not a better model.

### Parameter budget

The audit reports observations per parameter directly. Below 10, expect the priors to be
doing most of the work; below 5, say so explicitly in the report.

---

## 6. Signal

**Does the target move?** CV below 0.05 means there is nothing for media to explain, and
the model will return priors.

**Lagged correlation.** The audit reports each channel's correlation with the target at lag
0 and at its best lag. Two useful patterns:

* Best correlation at a positive lag (spend leads the target) → make sure `l_max` reaches
  that far. An upper-funnel channel looks uncorrelated at lag 0 and is easy to dismiss.
* Best correlation at a *negative* lag (the target leads spend) → reverse causality.
  Budgets set as a percentage of revenue, or bidding to a ROAS target, produce this. No
  model setting fixes it; see `mmm-causal-design`.

A negative contemporaneous correlation is usually a data problem — spend dated by invoice
rather than delivery — before it is a media problem.

**Seasonality.** Strength below 0.05 means Fourier terms will fit noise; reduce the order.
Above 0.5 means seasonality and media compete hard for the same variance whenever campaigns
are flighted seasonally, and named event flags do a better job than more Fourier modes.

---

## 7. Semantics — do the numbers mean what they claim?

**Cost per exposure unit.** Divide spend by impressions per period. Real auction prices
move; they do not move tenfold between the 5th and 95th percentile. A wild range means
spend and delivery were joined on mismatched dates or a partially populated feed — a data
bug the model will faithfully convert into a channel effect.

Periods with spend but no exposure teach the model that the channel does nothing. Periods
with exposure but no spend are free delivery, a make-good, or a broken join; if genuinely
free, it is organic, and ROAS is undefined at zero cost.

**Confounder coverage.** The audit checks for price/promotion and distribution.

> Omitting price is the single most common cause of overstated media ROAS in retail and
> CPG. Promotions and media are planned together, so the promotion's effect goes to
> whatever ran alongside it.

**Experiment coverage.** Channels with no incrementality test rest on observational
identification alone. That is not a blocking failure, but those channels deserve wider
ranges in the report than the tested ones.

---

## 8. Reading the tier

| Tier | Meaning | Do |
|---|---|---|
| **PASS** | No blocking errors, few warnings | Proceed; keep the warnings in the DS report |
| **WARN** | No blockers, several warnings | Proceed with each warning decided and documented |
| **FAIL** | Blocking errors | Fix them. A model on FAIL data is not a weaker model, it is a wrong one |

Blocking errors: missing required columns; non-numeric modelling columns; fewer than 52
periods; duplicate keys; a non-rectangular panel; a constant channel; VIF > 10; negative
spend; a granularity mismatch; a target that does not vary.

---

## 9. When the data cannot support an MMM

Say so. The alternatives are better than a model nobody should act on:

| Problem | Alternative |
|---|---|
| < 52 periods | Wait, or build a geo panel, or run experiments meanwhile |
| One channel dominates spend entirely | Experiments on that channel; MMM adds little |
| All channels perfectly correlated | Change the flighting deliberately for a quarter, then model |
| No spend variation at all | Experiments only |
| Target is not measured consistently | Fix measurement first; the model will fit the measurement change |

Delivering "the data cannot answer this, here is what would" in week two is a better
outcome than delivering a confident wrong answer in week ten.

---

## Related skills

`mmm-data-engineering` for building and fixing the dataset;
`mmm-channel-semantics` for what each column should be;
`mmm-causal-design` for which variables belong at all;
`mmm-validation` for testing the model the data supports.
