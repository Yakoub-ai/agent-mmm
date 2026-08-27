---
name: mmm-intake-questionnaire
description: |
  Intake for an MMM project — the questions that must be answered before any modelling, and the spec.yaml they produce. Use when starting a new MMM, resuming an incomplete intake, updating a spec, choosing a framework, declaring channel roles and experiments, or reviewing whether a project is properly scoped.
---

# MMM Intake

Intake produces `./mmm-workspace/spec.yaml` — the framework-agnostic definition of the
project. Everything downstream (audit, priors, model, reports) reads from it, and the same
spec compiles to pymc-marketing, Meridian or Robyn.

Intake is not paperwork. Three of the questions below — the decision, the channel roles,
and the confounders — determine whether the finished model is *right*. Everything else
determines how precisely it is estimated.

```bash
/mmm-intake-quick    # 6 questions, enough to run the audit
/mmm-intake          # full intake
```

Re-running `/mmm-intake` on an existing spec asks only about missing or stale fields.

---

## Section 1 — The decision

Ask first, because it changes the answers to everything else.

1. **What decision will this model inform?** ("How do we split next year's £12m?")
2. **Who signs it off, and when is the budget set?**
3. **What would you do differently depending on the answer?**
4. **Who will disagree with the result, and what would change their mind?**

If nobody can answer 3, the project is a reporting exercise. Say so now rather than in
week ten. Question 4 is worth asking out loud: the most common way a good MMM dies is that
its first appearance is a surprise to the person whose budget it questions.

Also capture: company, industry, region, business model, prior MMM experience
(greenfield vs brownfield — see that skill).

---

## Section 2 — Target

5. **What is being modelled?** Revenue, conversions, policies, signups, installs.
6. **Which system is the source of truth,** and has its definition changed in the window?
   A measurement change looks exactly like a change in how well marketing works.
7. **Monetary, acquisition, or volume?** Determines ROAS vs CPA framing throughout.
8. **Currency** (if monetary) or **value per unit** (if not, and known).
9. **Is value per unit time-varying?** If so, name the column — Meridian uses it as
   `revenue_per_kpi`.

```yaml
target_column: policies_sold
target_unit:
  kind: acquisition        # monetary | acquisition | volume
  label: policy
  value_per_unit: 250.0    # optional; enables ROAS alongside CPA
```

---

## Section 3 — Data

10. **Path, format, date column, granularity.** Weekly unless the response is genuinely
    fast and the data is genuinely daily at source.
11. **Date range and number of periods.** Below 104, expect wide intervals; below 52, an
    MMM is not appropriate without a geo panel.
12. **Geo panel?** Column name, list of geos, and a population column (Meridian requires
    it). A geo panel is the cheapest way to buy statistical power.
13. **Known gaps, outliers, structural breaks.** Relaunches, pandemics, tracking
    migrations, system changes.

```yaml
data_path: ./data/weekly.csv
date_column: date
granularity: weekly
geo:
  is_panel: true
  geo_column: dma
  population_column: population
```

---

## Section 4 — Channels and roles

This is where most MMM errors are introduced, and it takes longer than people expect.

**For each column, ask what it *is*:**

14. **Did we pay a media owner?** No → organic or non-media treatment.
15. **Does it represent exposure to advertising?** No → non-media treatment or control.
16. **Can we choose how much to buy next quarter?** No → control; keep it out of the
    optimiser.
17. **Is it caused by marketing?** (Site visits, brand search, installs, leads.) Yes → it
    is a mediator, not a channel. See `mmm-causal-design`.

**Then, for each media channel:**

18. **Spend column, and exposure column if measured.** Prefer exposure as the model input
    with spend alongside — spend alone confounds cost inflation with delivery.
19. **Reach and frequency available?** If so, and video/CTV matters, Meridian is the only
    framework that models it.
20. **Always-on, or flighted?** Always-on channels cannot be identified from observational
    data; flag them now so the report can say so.
21. **Should any channels be split or combined?** Brand vs generic search is the
    highest-value split. Channels that always move together must be combined — the model
    cannot separate them.

```yaml
channels:
  - column: brand_search_spend
    channel_type: sem_brand
    role: paid_media
    funnel_stage: lower
    always_on: true
  - column: tv_grps
    channel_type: tv
    role: paid_media
    spend_column: tv_spend
    exposure_column: tv_grps
    adstock: delayed
  - column: email_sends
    role: organic_media          # no spend => reports will not quote a ROAS
  - column: price_index
    role: non_media_treatment
    expected_sign: negative
```

`classify_channel(column_name)` proposes a type from the name. Confirm every one — column
names lie.

---

## Section 5 — Confounders

A confounder must move both spend and the target. Omitting one transfers its effect to
media.

22. **Does price or promotion move?** *If nothing else is captured, capture this.* It is
    the single most common cause of overstated retail and CPG ROAS.
23. **Does distribution change?** Store count, ACV, stock availability.
24. **Competitor activity?** Spend, share of voice, promotions.
25. **Macro or category conditions?** Consumer confidence, category volume, rates.
26. **Operational events?** Outages, delivery problems, capacity limits.
27. **Named calendar events?** Christmas, Black Friday, Ramadan, Golden Week,
    back-to-school, tax deadlines. These are step changes and belong as flags, not as
    Fourier order.
28. **Seasonality shape and strength?** Which weeks peak, which dip.

```yaml
controls:
  - column: price_index
    category: pricing
    expected_sign: negative
  - column: competitor_sov
    category: competitive
seasonality:
  yearly_fourier_modes: 6
  holiday_country: GB
```

---

## Section 6 — Experiments

29. **Has any channel ever been tested?** Geo holdout, matched market, ghost ads,
    conversion lift.
30. **For each: dates, channel, incremental spend, measured lift, and its standard error.**
31. **Can a test run during this project?**

An experiment with a lift value and a standard error becomes a likelihood constraint and
does more for the model than any amount of tuning. One with only a point estimate can
still inform a prior.

```yaml
experiments:
  - channel: tv_grps
    design: geo_holdout
    start_date: 2025-03-03
    end_date: 2025-04-14
    spend_during_test: 60000
    lift_absolute: 1500
    lift_se: 400
    source: "20 DMAs, 10 treated"
```

If the answer to 29 is "no", record that explicitly. It is the top limitation of the
model, and it belongs in the report.

---

## Section 7 — Framework and architecture

32. **Which framework?** See `mmm-framework-selection`. Defaults: Meridian for geo panels
    with RF; pymc-marketing for custom structure or a log link; Robyn for national models
    in R.
33. **Additive or multiplicative?** Log link when media and price interact
    multiplicatively (pymc-marketing only).
34. **Baseline flexibility.** Fixed intercept, changepoints, or time-varying. This is the
    largest single lever on the media share — decide it, do not default it.
35. **Expected media contribution.** A rough prior belief (e.g. 20–30%) makes the intercept
    prior meaningful instead of arbitrary.

```yaml
framework: pymc-marketing
architecture:
  link: identity
  likelihood: StudentT
  expected_media_contribution: 0.25
trend:
  kind: linear_changepoints
  n_changepoints: 8
```

---

## Section 8 — Validation and delivery

36. **Holdout period?** Which dates to reserve.
37. **How much validation does this decision deserve?** See `mmm-validation` §9.
38. **Which stakeholder reports are needed?**
39. **Refresh cadence?**

---

## Quick intake

`/mmm-intake-quick` asks six questions and produces a spec good enough to run the audit:

1. Company and industry
2. Data path
3. Date and target columns
4. Channel columns
5. Target unit kind
6. Greenfield or brownfield

Everything else takes a default. Run the audit immediately — its findings usually generate
the rest of the intake conversation for you.

---

## Validating the spec

```python
from agent_mmm.spec import load_spec
spec = load_spec("mmm-workspace/spec.yaml")
spec.channel_columns()        # what enters the media transformation
spec.control_columns()        # controls + non-media treatments
spec.spend_columns()          # only channels that have spend
spec.required_columns()       # everything the data must contain
```

The schema rejects: a monetary target without a currency; a panel without a geo column; an
experiment referencing an unknown channel; an experiment with neither a lift nor an ROI.

---

## What good intake looks like

* Every column has a declared **role**, confirmed with the client rather than inferred.
* Price and distribution are present, or their absence is a recorded decision.
* Known experiments are in the spec with their standard errors.
* The baseline specification is a choice, not a default.
* The decision the model informs is written down, with a name against it.

Intake that takes a morning and prevents a wrong model is the best-value time in the
project.
