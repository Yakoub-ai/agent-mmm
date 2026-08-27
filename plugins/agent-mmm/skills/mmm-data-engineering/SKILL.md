---
name: mmm-data-engineering
description: |
  Building the MMM dataset: sourcing, joining, aggregating, handling missing data, currency and inflation, fiscal calendars, taxonomy mapping, outliers and reproducible pipelines. Use when assembling data from platform exports and finance systems, deciding how to fill gaps, choosing daily vs weekly, aligning spend to delivery dates, mapping campaign names to channels, or when the audit reports impossible costs, gaps, duplicates or a non-rectangular panel.
---

# Data Engineering for MMM

The dataset is the model's entire view of the world. Sixty to eighty percent of MMM
project time goes here, and the failures are quiet: nothing errors, the model fits, and
the answer is wrong in a way no diagnostic catches.

The governing principle: **every transformation is a claim about what happened.**
`fillna(0)` claims no spend occurred. Averaging when aggregating claims a variable is a
rate. Deleting an outlier claims it never happened. Make each claim explicitly, record it,
and be able to defend it.

---

## 1. Target shape

One row per period, per panel unit:

| date | geo | target | ch1_spend | ch1_impressions | ... | price | distribution | population |
|---|---|---|---|---|---|---|---|---|

Requirements every framework shares:

* **Dense.** Every period present, no gaps. Adstock convolves over consecutive positions,
  so a missing week silently shortens carryover for everything after it.
* **Rectangular** (panel). Every (geo, period) cell exists. Meridian and pymc-marketing
  panel models both require it and neither will impute for you.
* **One row per key.** Duplicates double-count spend and target.
* **Numeric.** Currency symbols, thousand separators and `"NULL"` strings turn columns
  into objects, and a careless `to_numeric(errors="coerce")` turns them into zeros.

---

## 2. Granularity

| | Daily | **Weekly** | Monthly |
|---|---|---|---|
| Observations in 2y | 730 | 104 | 24 |
| Day-of-week noise | dominates | averaged out | averaged out |
| Carryover resolution | high | adequate | poor |
| Matches media planning | rarely | usually | sometimes |
| Data availability | patchy | good | good |

**Weekly is right for most MMM.** Daily data is dominated by day-of-week effects and by
the fact that most media is bought and reported weekly; you pay for the extra rows in
noise and in day-of-week parameters. Monthly gives 24 observations for a model with 30+
parameters — it cannot identify carryover at all.

Use daily only when the response is genuinely fast (app installs, flash sales) *and* the
data is genuinely daily at source. Aggregating monthly data to weekly by dividing by 4.3
invents variation and is worse than modelling monthly.

**Week definition must be consistent across every source.** A Monday-start week in the
media export and a Sunday-start week in the sales extract shifts spend one day relative
to sales, which shows up as a spurious lag.

---

## 3. Aggregation semantics

When collapsing periods, the rule depends on what the variable *is*, not its dtype:

| Kind | Rule | Examples |
|---|---|---|
| **Flow** | sum | spend, impressions, clicks, sales, conversions |
| **State** | mean or last | price, distribution, store count, ACV |
| **Rate** | mean, ideally weighted | frequency, conversion rate, CPM |
| **Population** | mean or last | geo population |

Summing a price index produces a number with no meaning, and nothing downstream will
notice. `agent_mmm.data_prep.infer_aggregation()` derives the rule from the spec's
declared roles, and `aggregate_to_period()` reports which columns it summed and which it
averaged so you can check.

Weighted rates are worth the effort where they matter: average frequency across a period
should be weighted by reach, not by period.

---

## 4. Joining sources

A typical build joins: platform exports (Meta, Google, TikTok, DSPs), agency plans (TV,
OOH, radio, print), finance actuals, sales or CRM data, price and distribution feeds,
weather, macro series, competitive tracking.

**Spend should come from finance, not from platforms.** Platform-reported spend excludes
agency fees, ad-serving and production, and is reported net of credits at inconsistent
times. Finance is the number the business actually spent; use platforms for delivery
metrics and finance for cost.

**Align on delivery, not on invoicing.** The three dating problems, in order of frequency:

* *Invoice date* — a TV invoice raised on the 30th for a campaign that ran across the
  month puts a month of exposure into one week.
* *Booking date* — OOH is contracted months ahead of exposure.
* *Accrual* — sponsorship spread evenly across a contract year, unrelated to when anyone
  saw anything.

For anything accrued or contracted, model the *exposure* (broadcast dates, event dates,
delivered impressions), not the money.

**Timezones.** Platform data is often reported in the account's timezone, sales in the
market's. At weekly granularity this rarely matters; at daily it shifts everything by a
day.

---

## 5. Taxonomy mapping

Campaign names are the join key nobody designed. Map them deterministically and version
the mapping:

```
campaign_name -> platform -> channel -> funnel_stage -> role
"UK_BRAND_YT_Q1_AwarenessMasthead" -> YouTube -> ctv -> upper -> paid_media
"UK_PERF_GAds_Brand_Exact"         -> Google  -> sem_brand -> lower -> paid_media
```

Rules that save rework:

* Keep the mapping in a versioned file, not in a notebook. Every refresh must be able to
  reproduce last quarter's split.
* Map to `role` (paid / organic / non-media treatment) at the same time as channel — the
  role decides whether a ROAS is even defined.
* Send unmapped spend to an explicit `other` channel rather than dropping it. Dropped
  spend becomes unexplained target variation attributed to whatever remains.
* Reconcile totals against finance every refresh. A 3% gap is a mapping change nobody
  told you about.

---

## 6. Missing data

**Diagnose before filling.** Why the value is missing determines what to do:

| Situation | What it means | Fill |
|---|---|---|
| No spend that period | Genuinely zero | `0` |
| Feed not received | Value exists, we lack it | Backfill from source; otherwise `interpolate` or leave NaN |
| Channel started mid-series | Did not exist | `0` before launch; consider starting the series later |
| Channel wound down | Stopped | `0` after; check for a structural break |
| Price/distribution not reported | State persisted | `ffill` |
| Target missing | Period unusable | `drop` — never fill a target |

Never blanket-`fillna(0)`. Zeroing a missing target invents a period of no sales; zeroing
a missing price invents a giveaway.

```python
from agent_mmm.data_prep import suggest_imputation, impute, prepare_dataset

strategies = suggest_imputation(spec, df)   # proposal from declared roles
strategies["competitor_spend"] = "interpolate"
df, report = prepare_dataset(spec, df=df, impute_strategies=strategies)
print(report.to_markdown())                  # every change, and every judgement call
```

**Median imputation is a last resort.** It shrinks the variance the model needs and
biases that variable's coefficient towards zero. If a channel needs median imputation for
20% of its history, it is not ready to be modelled.

**A channel with a short history** (launched 6 months into a 2-year window) has only
those months of information. Zero-filling the rest is correct but does not create
information — expect a wide posterior and say so.

---

## 7. Money: currency and inflation

**Currency.** Convert with the rate in force when the money was spent, not today's. A
constant retrospective rate removes real variation in local purchasing power; a
per-period rate applied to a target measured in a different currency introduces variation
no channel caused. Convert both spend and a monetary target, or neither.

**Inflation.** Over three years, inflation alone can make late spend look 15% larger at
identical delivery. The model reads that as more media. Deflate spend and a monetary
target with the same index, to the same base — deflating one side manufactures a trend.

Media cost inflation is not general inflation. CPMs in a competitive category can rise
much faster than CPI, which is the strongest argument for modelling on impressions rather
than spend where you can.

```python
from agent_mmm.data_prep import deflate, convert_currency
df = deflate(df, ["tv_spend", "revenue"], cpi_series, base_period=df.index[-1])
```

---

## 8. Calendars

**Fiscal vs calendar.** If the business reports on a 4-4-5 fiscal calendar, the target may
be aggregated to fiscal weeks while media reports on ISO weeks. Pick one and convert
everything to it.

**ISO weeks** have a 53-week year roughly every 5–6 years. A yearly Fourier term assumes
52; the extra week creates a small phase drift that accumulates over a long series.

**Holidays move.** Easter shifts by up to five weeks; Ramadan moves ~11 days a year;
Chinese New Year moves within Jan–Feb. Year-on-year comparison at fixed week numbers is
wrong for all three. Use explicit event flags anchored to the actual date.

```python
from agent_mmm.data_prep import add_calendar_features, add_event_flags
df = add_calendar_features(df, "date", country="GB")
df = add_event_flags(df, "date", {
    "black_friday": ("2024-11-25", "2024-12-02"),
    "site_migration": ("2025-02-10", "2025-02-24"),
})
```

---

## 9. Outliers and structural breaks

**Flag, do not delete.** A spike is a data point with an explanation attached; deleting it
throws away both the observation and the explanation. Deleting outliers also biases the
model towards the ordinary, which is exactly the regime a budget optimiser will push away
from.

| Cause | Treatment |
|---|---|
| Data error (duplicate load, wrong units) | Fix at source |
| One-off event (competitor exit, viral moment) | Event flag |
| Promotion | Price/promo variable |
| Stockout | Availability control |
| Genuine heavy tail | StudentT likelihood, `nu` ≈ 3–5 |

**Structural breaks** — a relaunch, a pricing overhaul, a pandemic, a measurement change
(consent-mode rollout, iOS 14, a tracking migration). Options: an indicator for the
regime, splitting the series, or a time-varying intercept. An unmodelled break is
attributed to whatever moved at the same time.

Measurement changes are the sneaky ones: a change in how conversions are counted looks
exactly like a change in how well marketing works.

---

## 10. Geo panels

The strongest argument for a geo panel is arithmetic: 50 geos × 104 weeks is 5,200
observations against 104 national ones, and cross-sectional variation in media weight is
information that national aggregation destroys.

Requirements: a rectangular grid; a population column (Meridian requires it, and it
enables per-capita scaling); geos small enough to differ in media weight but large enough
for a stable target; a documented view on spillover between adjacent geos.

Watch for: national buys that cannot be geo-allocated (allocate by population and mark
the assumption); geos whose target is mostly zeros; and one dominant geo that carries the
whole estimate.

---

## 11. Reproducibility

Anyone should be able to rebuild the exact dataset a model was fitted to.

* **Versioned raw extracts.** Keep them; platforms restate history.
* **Deterministic pipeline.** Same inputs, same output, every time. No manual steps.
* **Fingerprint the modelling frame.** `agent_mmm.fit_runner.data_fingerprint()` hashes it
  and stores the hash in `metrics.json`, so a run can always be tied to its input.
* **Record library versions.** `environment_metadata()` captures pymc-marketing, PyMC,
  ArviZ, NumPy, pandas — enough to explain why a rerun differs.
* **Version the taxonomy mapping** alongside the code.
* **Refresh cadence**: re-run the audit every refresh, not just the first time. New data
  brings new gaps, new campaigns and new mapping misses.

---

## 12. Pre-flight checklist

```
Shape
  [ ] One row per (date[, geo]); no duplicates
  [ ] Every period present between first and last
  [ ] Panel rectangular
  [ ] Declared granularity matches observed row spacing

Integrity
  [ ] No negative spend
  [ ] Missing values diagnosed, not blanket-filled
  [ ] Target has no invented values
  [ ] Numeric columns are actually numeric

Semantics
  [ ] Spend from finance; delivery from platforms
  [ ] Dates align to delivery, not invoice or booking
  [ ] Cost per exposure unit is stable within an order of magnitude
  [ ] Currency and inflation handled consistently on both sides
  [ ] Aggregation rule matches each variable's kind

Coverage
  [ ] Price / promotion present
  [ ] Distribution present (retail/CPG)
  [ ] Competitor activity present if it moves
  [ ] Named events and known breaks flagged
  [ ] Unmapped spend routed to `other`, not dropped

Identifiability
  [ ] >= 104 periods, or a geo panel making up for it
  [ ] Channels vary; nothing is constant
  [ ] Pairwise correlations below ~0.8, VIF below ~10
  [ ] Enough observations per parameter (aim for >= 10)

Reproducibility
  [ ] Pipeline is deterministic and versioned
  [ ] Data fingerprint recorded with the run
  [ ] Taxonomy mapping versioned
```

`agent_mmm.data_audit.run_audit(spec)` mechanises most of this and writes
`mmm-workspace/audit/audit_report.md`. Run it before every fit, and again at every
refresh.

---

## Related skills

`mmm-data-quality` for interpreting audit findings; `mmm-channel-semantics` for what each
column should be; `mmm-causal-design` for which variables belong at all;
`mmm-baseline-and-trend` for controls and events.
