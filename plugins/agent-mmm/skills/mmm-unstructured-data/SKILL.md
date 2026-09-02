---
name: mmm-unstructured-data
description: |
  Making sense of raw, messy, unstructured marketing data before it becomes a modelling dataset. Use when handed a folder of exports, a shared drive, or files nobody can fully explain — platform exports at different grains, finance workbooks, CRM extracts, hand-maintained promo calendars. Covers profiling unknown files, inferring date formats and grain, classifying columns, mapping campaign names to channels, reconciling against finance, and the question bank that has to be answered before any of it can be trusted.
---

# Understanding Unstructured Marketing Data

Real MMM projects do not begin with a tidy CSV. They begin with a folder containing a Meta
export, three Google Ads reports at different grains, a GA4 extract, a finance workbook
with merged header cells, a promo calendar somebody maintained by hand, and two files
nobody can explain.

The temptation is to start joining. Resist it. **The most valuable output of this stage is
a list of questions**, and the second most valuable is a written record of what each file
actually is. Joins made before those exist are joins that get silently redone three weeks
later, after a model has been built on them.

---

## 1. The governing principle

**Every transformation is a claim about what happened.**

| Transformation | Claim |
|---|---|
| `fillna(0)` | No activity occurred in that period |
| `fillna(method="ffill")` | The last known state persisted |
| `mean()` on aggregation | The variable is a rate |
| `sum()` on aggregation | The variable is a quantity |
| dropping a row | That period never happened |
| coercing text to numeric | Anything unparseable was zero |

None of these is wrong. All of them are wrong *sometimes*, and which one applies is a fact
about the business, not about the data. That is why this stage is mostly conversation.

---

## 2. Profile before you plan

```python
from agent_mmm.discovery import discover, render_discovery_report

report = discover("data/raw")        # read-only: reads everything, changes nothing
print(render_discovery_report(report))
```

For each file this establishes:

* **The date column and the single format it parses under.** Applying one format to the
  whole column matters more than it sounds: parsing element by element turns `05/06/2022`
  into May 6th and `19/06/2022` into June 19th *in the same column*, scrambling the
  calendar with no error and no warning. Where every day-of-month is 12 or less, the order
  genuinely cannot be recovered from the file and becomes a question.
* **The grain** — daily, weekly, monthly — from the modal gap between distinct dates, with
  a consistency score. A low score means missing periods.
* **The shape** — one row per period (wide), one row per campaign per day (long), or an
  undated lookup table. This decides whether the file needs aggregating and pivoting before
  it can join.
* **A semantic role per column**, with a confidence and a reason. Anything below 50%
  confidence becomes a question rather than an assumption.
* **Currency, duplicates, constants, negatives, and text-that-is-really-numbers.**

Across files it establishes the modelling grain, the window where sources overlap, and the
join candidates with their overlap counts.

---

## 3. What the profiler catches that people miss

**Money stored as text.** `"£1,234.50"` and `"(2,000)"` are extremely common in finance
exports. A careless `to_numeric(errors="coerce")` turns them into zeros, which the model
reads as "the channel was dark" rather than "we failed to parse this".

**Ratios that look like money.** CPC, CPM, CPA, ROAS, CTR. A ratio cannot carry adstock or
saturation and must never be a channel input. If the ratio is all you have, ask for the
numerator and denominator separately.

**The coarsest source dragging down the grain.** One monthly TV plan among weekly sources
does not make this a monthly model — it makes the TV file a problem to solve. Splitting a
monthly total evenly across weeks invents variation the model will read as real; modelling
everything monthly throws away three quarters of the observations. Ask for a broadcast
calendar, a spot log, or flight dates.

**Overlap, not coverage.** A media file covering three years joined to a sales file
covering eight months is an eight-month model. People are consistently surprised by this at
the wrong moment.

**Geo naming that does not match.** `GB-LON` in one file and `London` in another needs a
crosswalk. A join across mismatched keys drops rows quietly and biases whatever survives.

---

## 4. The question bank

These have to be answered by a human. They are not inferable, and guessing at any of them
produces a model that is wrong in a way no diagnostic catches.

### Blocking — nothing proceeds without these

1. **What decision will this inform, by when, and what would change if the answer came
   back differently?** This sets the required precision, the channel granularity, and
   whether an MMM is even the right instrument.
2. **Which column is the target, and is it gross or net** of returns, tax and discounts?
   Modelling gross revenue and reporting net ROAS overstates every channel by the return
   rate.
3. **Which of these are paid media, which are organic, and which are business levers**
   (price, promotion, distribution, launches)? Roles decide structure, and an organic
   channel given a ROAS is a category error no diagnostic catches.
4. **Has any channel ever been tested?** Even informally. Even with a disappointing result.
   One usable measurement anchors the whole decomposition.
5. **For long-format files: which column identifies the channel, and which rows must be
   excluded?** Test campaigns, internal traffic, cancelled orders, employee purchases.
6. **What do duplicate rows mean** — genuine repeats to sum, restatements where the latest
   wins, or an export defect?

### Structural

7. **Does the date mark delivery, or booking, or billing?** Adstock is a claim about when
   an impression landed. Billing dates can sit weeks from delivery.
8. **Which day does the week start on, in each file?** A Monday-start media week against a
   Sunday-start sales week shifts spend one day and shows up as a spurious lag.
9. **For each channel: is it always on, and has spend varied materially?** A channel that
   never varies cannot be measured observationally, whatever the model reports.
10. **Where spend and exposure both exist, which should drive the transformation?** For
    auction channels bought on efficiency, exposure is the better input and spend stays for
    ROAS.
11. **What is this column?** For anything unclassified or weakly classified. An
    unexplained column is either a missing control or a leak, and you cannot tell which.

### Confounders and history

12. **Does price move over the window, and do we have it?** If price moves and is not
    controlled, media takes credit for promotions. This is the most common source of
    inflated ROAS in retail and CPG.
13. **Is there a distribution, store-count or availability measure?** Growth from opening
    stores looks exactly like growth from advertising.
14. **What happened in this window that is not in the data?** A rebrand, a site migration,
    a tracking change, a stockout, a competitor entering, a pandemic. Structural breaks
    explain more variance than most channels and will otherwise be absorbed into whichever
    channel happened to move at the time.
15. **Are there multiple currencies, and at what rate should they convert, as of when?**
    Converting at today's rate rewrites history; converting at the transaction rate mixes
    FX movement into media response.

---

## 5. Campaign names to channels

```python
from agent_mmm.taxonomy import suggest_rules, build_taxonomy, render_taxonomy_report

rules  = suggest_rules(df["campaign_name"].unique())   # a draft to argue with
result = build_taxonomy(df, name_column="campaign_name", spend_column="spend", rules=rules)
```

This mapping is where a surprising share of MMM error lives, because it is done once, by
hand, in a spreadsheet nobody keeps, and never checked again.

**Unmapped spend above 1% is an error, not a rounding difference.** Every unmapped
currency unit is media the model cannot see but whose effect is still in the target, so it
lands in the baseline or in whichever channel correlates with it. Work the unmapped list in
spend order, not alphabetically.

**Splits that must survive the mapping:**

* **Brand vs generic search.** Brand search is usually the least incremental line in the
  plan and generic is usually among the most. Collapsed together they produce a number that
  describes neither.
* **Prospecting vs retargeting.** Retargeting reaches people who already intended to buy;
  its measured ROAS is the most commonly overstated number in any media plan.
* **PMax stays its own channel** with the caveat that its coefficient mixes search,
  shopping, display and video inventory, unless a breakdown is available.

**Check the conflicts.** Where two rules claim the same name, priority decided it, and
priority is a number someone should have chosen deliberately rather than inherited.

---

## 6. Reconcile against finance

```python
from agent_mmm.reconciliation import reconcile_spend, render_reconciliation_report

rec = reconcile_spend(modelled_df, finance_df, finance_spend_column="media_cost")
```

Platform exports and the general ledger always disagree. The **shape** of the disagreement
is the diagnosis:

| Pattern | Usual cause | What to do |
|---|---|---|
| `constant_shortfall` | Agency fees, ad serving, VAT in the ledger but not the export | Decide gross or net deliberately, and use the same basis in the reported ROAS |
| `constant_excess` | Double counting, or a currency/tax basis difference | Find the campaigns counted twice |
| `drifting` | A channel added mid-window and never mapped, or a renamed campaign dropping out | Find the period the drift starts, look at what launched |
| `timing_shift` | Platform reports delivery, ledger reports invoice | Reconcile on delivery date — this one corrupts carryover directly |
| `isolated_spikes` | Rebates, credits, make-goods, restatements | Identify each; do not smooth them away |
| `unstructured` | The two sources are not measuring the same thing | Re-check the join keys before anything else |

A model whose media numbers finance does not recognise will not survive its first review
with the CFO, and it should not: if the model cannot account for the money, its ROAS is
measuring something other than the money.

**Get sign-off before the model is built, not when the results are presented.**

---

## 7. Order of work

1. **Profile everything.** Change nothing.
2. **Report the blocking questions.** Stop. This is a conversation, not a step.
3. **Map the taxonomy** once the roles and exclusions are agreed.
4. **Reconcile against finance** and get the numbers signed off.
5. **Join**, on an agreed grain and an agreed window, with a crosswalk for any key that
   needed one.
6. **Prepare** — dense calendar, one explicit imputation rule per column, shown before it
   is applied.
7. **Audit** the joined dataset.
8. **Write down every claim** the pipeline makes, so it can be defended and re-run.

Steps 1-4 are usually most of the elapsed time on an MMM project. That is not a sign the
project is going badly.

---

## Related skills

`mmm-data-engineering` for joins, currency, calendars and imputation mechanics.
`mmm-data-quality` for reading the audit and judging readiness.
`mmm-channel-semantics` for deciding what a column *is*.
`mmm-intake-questionnaire` for the full project spec conversation.
