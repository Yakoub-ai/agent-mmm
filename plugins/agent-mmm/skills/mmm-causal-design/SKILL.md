---
name: mmm-causal-design
description: |
  Causal identification for MMM — what to control for, what never to control for, and why. Use when deciding which variables to include, when a channel's estimate looks structurally wrong, when handling mediators (search volume, site visits, brand metrics), colliders, funnel effects or reverse causality, when building a DAG for pymc-marketing's dag/treatment_nodes/outcome_node, or when explaining why an MMM coefficient is not automatically a causal effect.
---

# Causal Design for MMM

An MMM is a regression that everyone treats as a causal claim. That is only legitimate
if the model is specified so the coefficient *is* the causal effect. Fit quality has no
bearing on this: a model with R² 0.95 and the wrong control set gives a confidently wrong
answer, and nothing in the diagnostics will flag it.

The question a good MMM answers is a counterfactual: *if we had spent differently, what
would have happened?* Getting a regression to answer that requires deciding, in advance,
what to adjust for — and adjusting for the wrong thing does more damage than adjusting
for nothing.

---

## 1. The four variable types

Draw the arrows before you write the model.

```
CONFOUNDER            MEDIATOR                 COLLIDER              INSTRUMENT
   Z                  X -> M -> Y             X -> C <- Y            Z -> X -> Y
  / \                                                                (Z ⊥ Y | X)
 X   Y
```

| Type | Definition | Action | Consequence of getting it wrong |
|---|---|---|---|
| **Confounder** | Causes both spend and the target | **Control for it** | Omitted → the confounder's effect is credited to media |
| **Mediator** | Caused by spend, causes the target | **Do NOT control** (unless you want only the direct effect) | Controlled → media's effect through it disappears |
| **Collider** | Caused by both spend and the target | **NEVER control** | Controlled → creates a spurious association out of nothing |
| **Instrument** | Moves spend but affects the target only through spend | Use for identification | Rare in MMM; an experiment is the usable case |

The asymmetry matters: an omitted confounder biases the estimate, but a controlled
mediator *deletes* the effect you are trying to measure. The second mistake is more
common and less visible.

---

## 2. Confounders: the ones that actually bite

A confounder must move both spend and the target. In MMM that is a short, predictable
list.

| Confounder | Why it moves spend | Why it moves the target |
|---|---|---|
| **Seasonality** | Budgets are flighted into peak season | Demand is seasonal |
| **Price / promotion** | Media supports promotions | Price moves sales directly |
| **Distribution** | Launch media accompanies expansion | More availability, more sales |
| **Trend / brand equity** | Budgets grow with the business | The business is growing |
| **Competitor activity** | Budgets respond to competitors | Competitors take share |
| **Macro conditions** | Budgets are cut in downturns | Demand falls in downturns |
| **Stock availability** | Media is pulled when out of stock | Cannot sell what you do not have |

**Expectation-driven budgeting is the deepest problem.** Planners spend more when they
expect demand. That makes *expected demand* a confounder — and it is unobserved. Proxies
help (last year's sales in the same week, category volume, a demand index) but never fully
resolve it. This is the structural reason MMM needs experiments: an experiment creates
spend variation that expectation did not cause.

**Never proxy expected demand with anything derived from the current target.** Lagged
sales, year-on-year sales, or a "baseline forecast" built from the target will explain the
target beautifully and leave media with nothing. A near-perfect MMM is almost always
leaking the target into the right-hand side.

---

## 3. Mediators: the mistake that deletes your effect

A mediator sits on the causal path from media to the target.

```
TV -> brand awareness -> brand search -> conversions
Meta -> site visits -> conversions
Radio -> store footfall -> sales
```

Control for brand search in a conversion model and TV's effect *through* brand search
vanishes. You are left with TV's direct effect only, which for an upper-funnel channel is
close to zero. The model then reports, with high confidence, that TV does nothing.

**How to tell a mediator from a confounder:** ask which came first, and what causes what.
Search volume rises *after* the TV flight → mediator. Interest rates move independently of
your media plan → confounder. If marketing can move it, it is not a confounder.

**Common variables that are mediators, not controls:**
website sessions, app installs, brand search volume, brand awareness / consideration
scores, email list size, social followers, store footfall, quote starts, leads.

Putting any of these on the right-hand side of a sales model is the fastest way to make an
MMM say marketing does not work.

**What to do instead:**

1. **Leave it out.** The channel's coefficient then captures its total effect, direct and
   indirect. This is usually what you want.
2. **Model the funnel explicitly.** Fit media → mediator, then mediator + media →
   conversions. Total effect = direct + (effect on mediator × mediator's effect on the
   target). pymc-marketing 1.x supports funnel-aware structures for exactly this.
3. **Control deliberately, and say so.** Sometimes the direct effect is the question — for
   example, isolating what brand search adds *beyond* the demand upstream media created.
   That is a legitimate analysis as long as the report says the number is a direct effect.

**Circularity.** Never put SEM spend in a model whose target is search volume: search
volume drives SEM impressions, not the reverse. The multi-model funnel design (sales
model, search-volume model, SEM-clicks model) exists to keep each stage's inputs upstream
of its output.

---

## 4. Colliders: adjusting for one manufactures a correlation

A collider is caused by both the treatment and the outcome. Conditioning on it induces a
dependence between two things that were independent.

Concrete MMM cases:

* **Conditioning on converters.** Analysing only weeks with high conversions, or only
  users who converted, conditions on the outcome. Any channel that also causes conversions
  now looks correlated with everything else that does.
* **Filtering to "campaigns that ran".** If a campaign runs when both budget is available
  *and* the outlook is good, the "ran" flag is a collider.
* **Platform-attributed conversions as a control.** These are caused by both the media and
  the conversion. Adjusting for them is conditioning on a collider *and* on a mediator at
  once.
* **Any post-treatment filter.** Excluding stockout weeks, excluding "anomalous" weeks,
  excluding low-spend weeks — each conditions on something downstream of both.

The tell is that a collider is measured *after* the treatment. If a variable could not
have existed before the spend decision, be suspicious.

---

## 5. Reverse causality

Spend responds to sales as often as sales respond to spend:

* Budgets are set as a percentage of revenue.
* Performance channels bid to a target ROAS, so spend rises when conversion rates rise.
* Media is cut when a product goes out of stock.
* Budgets are pulled forward when a quarter is going badly.

Symptoms: a channel with a negative fitted effect; a channel whose correlation with the
target is strongest at *negative* lag (spend follows sales); implausibly high estimates
for automated-bidding channels.

Treatments: control for whatever drives the feedback (revenue targets, stock, conversion
rate — carefully, since some of these are mediators); model at a granularity coarser than
the feedback loop (monthly budget-setting is not visible in monthly data); or run an
experiment, which severs the loop by construction.

---

## 6. Writing the DAG down

The discipline is worth more than the diagram. Writing arrows forces you to state what
you believe causes what, and disagreements surface before the model is fitted rather than
during the readout.

```python
dag = """
digraph {
    season -> demand;   season -> tv_spend;   season -> search_spend;
    price -> sales;     price -> promo_spend;
    distribution -> sales;
    tv_spend -> brand_search;   brand_search -> sales;   tv_spend -> sales;
    search_spend -> sales;
    demand -> sales;    demand -> search_spend;
}
"""
mmm = MMM(
    ..., dag=dag,
    treatment_nodes=["tv_spend", "search_spend", "promo_spend"],
    outcome_node="sales",
)
```

pymc-marketing uses the DAG to identify a valid adjustment set (needs the `dag` extra:
`dowhy`, `networkx`, `pygraphviz`). `pymc_marketing.mmm.causal` also provides
`BuildModelFromDAG`, `CausalGraphModel`, and `TBFPC` — a Bayes-factor causal-discovery
routine oriented on a target variable. Treat discovery output as a hypothesis to argue
about with the marketing team, never as a specification to adopt.

**A minimal DAG protocol for an intake workshop:**

1. List every variable, including unmeasured ones (write "expected demand" even though
   you cannot observe it).
2. Ask, for each pair: could A cause B? Could B cause A? Could something cause both?
3. Mark which arrows the marketing team disagrees about — those are the assumptions the
   report must disclose.
4. Read off the adjustment set. Anything downstream of a treatment is not in it.

---

## 7. From coefficient to causal claim

A media coefficient is a causal effect only if all of these hold:

- [ ] Every confounder is measured and in the model (price, distribution, seasonality, competition, macro).
- [ ] No mediator is controlled for (no site visits, brand search, or awareness on the right-hand side).
- [ ] No collider is conditioned on, and no post-treatment filtering was applied.
- [ ] Spend variation is not purely demand-driven (there is flighting, geo variation, or an experiment).
- [ ] The channel is identified — spend actually varies, and is not collinear with another channel.
- [ ] The functional form can express the true response (carryover long enough, saturation shape plausible).
- [ ] The estimate is consistent with, or calibrated to, an experiment.

Every unchecked box is a caveat that belongs in the report. Two or three unchecked and the
right output is a range with a recommendation to test, not a point estimate with a
recommended budget.

---

## 8. Diagnosing a structurally wrong result

| Symptom | Causal cause | Fix |
|---|---|---|
| Brand search has the highest ROAS | Mediator effect: it harvests demand upstream media created | Model the funnel, or control for demand, or run a brand-search holdout |
| TV shows no effect | Its mediator (brand search, site visits) is in the model as a control | Remove the mediator, or compute the total effect through it |
| A channel is negative | Reverse causality, or spend dated by invoice rather than delivery | Check the data dating first; then look for the feedback loop |
| Media explains >60% of sales | Missing confounder — usually price or distribution | Add the driver |
| Every channel looks great | Seasonality or trend is uncontrolled and everything correlates with it | Check the baseline share and the seasonal term |
| Retargeting beats prospecting | Selection on people already likely to convert | Holdout test; expect the model to be wrong here |
| Results swing between refreshes | The model is not identified; small data changes move a knife-edge split | Group collinear channels, tighten priors, calibrate |

---

## 9. What the frameworks give you

| | pymc-marketing | Meridian | Robyn |
|---|---|---|---|
| DAG-based adjustment | `dag`, `treatment_nodes`, `outcome_node` (dowhy) | no | no |
| Causal discovery | `causal.TBFPC` | no | no |
| Funnel / mediation | funnel-aware structures | no built-in | no |
| Separate non-media treatments | via `control_columns` + role discipline | `non_media_treatments` slot | `context_vars` + `context_signs` |
| Experiment as likelihood | `add_lift_test_measurements` | ROI priors (`roi_calibration_period`) | `calibration_input` (third objective) |
| Sign constraints | prior support | prior type | `paid_media_signs`, `context_signs` |

None of them checks your causal assumptions. They give you the vocabulary to state them;
stating them correctly is your job.

---

## Related skills

`mmm-channel-semantics` for which variables are mediators by nature;
`mmm-baseline-and-trend` for confounders that live in the baseline;
`mmm-experimentation-calibration` for resolving what observation cannot;
`mmm-validation` for refutation tests that probe these assumptions.
