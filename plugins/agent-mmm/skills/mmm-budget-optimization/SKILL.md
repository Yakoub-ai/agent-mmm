---
name: mmm-budget-optimization
description: |
  Budget allocation and scenario analysis from a fitted MMM. Use when optimising spend across channels, setting bounds and constraints, running sensitivity or what-if scenarios, comparing current versus optimal allocation, choosing between fixed-budget and target-return scenarios, or judging whether an optimiser's recommendation is safe to act on. Covers pymc-marketing 1.x mmm.budget_optimizer, Meridian's optimizer and Robyn's allocator.
---

# Budget Optimisation

The optimiser is the sharpest instrument in the toolkit and the easiest to misuse. It will
answer whatever question you pose, including questions the model has no information about
— and it is most confident exactly where it knows least, because the extrapolated part of
a saturation curve is smooth and well-behaved.

Everything below is about posing the question so the answer is usable.

---

## 1. The principle

Optimal allocation **equalises marginal return across channels**, subject to constraints.
It does not rank channels by average ROAS. On a concave curve, moving budget into a
high-average-ROAS channel that is already saturated destroys value, and average ROAS
cannot tell you that.

Two questions, two scenarios:

| Question | Scenario |
|---|---|
| "Reallocate our existing £10m" | Fixed budget — maximise response |
| "How much *should* we spend?" | Flexible budget to a target return |

The second is the more valuable question and the one most teams never ask. If marginal
ROAS exceeds 1.0 at current spend across several channels, the answer is not "reallocate",
it is "spend more".

---

## 2. pymc-marketing 1.x

```python
opt = mmm.budget_optimizer("2026-01-05", "2026-03-30")

result = opt.allocate_budget(
    total_budget=1_000_000,
    budget_bounds={
        "tv_grps":      (150_000, 400_000),
        "sem_spend":    ( 80_000, 250_000),
        "social_spend": ( 50_000, 300_000),
    },
)   # -> BudgetOptimizationResult   (a tuple in earlier versions; changed in 1.1)
```

`mmm.budget_optimizer(start, end)` is the recommended entry point: it builds the
optimisation model, computes `num_periods`, and pulls `adstock_periods` from the fitted
adstock so carry-in and carry-over are handled. Constructing `BudgetOptimizer` directly
means getting those right yourself.

**The window must be at least as long as the carryover.** Optimising a 13-week window for
a channel with a 16-week effective carryover counts only the response landing inside the
window, which systematically under-funds long-carryover media. `mmm.effective_carryover_lags()`
returns `l_max` plus any additional carryover declared by registered effects.

For non-spend channels (impressions, GRPs), pass `cost_per_unit` so the optimiser works in
money. Custom objectives go through `utility_function` and `set_constraints`.

```python
mmm.plot.budget_allocation()
mmm.plot.allocated_contribution_by_channel_over_time()
```

## Meridian

```python
from meridian.analysis import optimizer
opt = optimizer.BudgetOptimizer(mmm)

fixed = opt.optimize(fixed_budget=True, budget=1_000_000,
                     spend_constraint_lower=0.3, spend_constraint_upper=0.3,
                     use_optimal_frequency=True)

flexible = opt.optimize(fixed_budget=False, target_roi=1.5)

fixed.output_optimization_summary("optimization.html", ".")
```

Meridian's constraints are relative (±30% of current spend by default), which is a
sensible default for a plan people have to execute. `use_optimal_frequency=True` also
optimises frequency for RF channels — a lever no other framework exposes.

## Robyn

```r
AllocatorCollect <- robyn_allocator(
  InputCollect = InputCollect, OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response",        # or "target_efficiency"
  total_budget = 1000000, date_range = "last_12",
  channel_constr_low = 0.7, channel_constr_up = 1.5,
  constr_mode = "eq"                # "ineq" allows underspending
)
```

Robyn's allocator is fast and its plots are the best of the three for a marketing
audience. `constr_mode = "ineq"` is worth using when the honest answer might be "spend
less".

---

## 3. Bounds are the most important input

An unbounded optimiser will recommend zero for some channels and a tenfold increase for
others, because nothing in the mathematics prevents it. Bounds encode what is executable
and where the model has evidence.

**Set bounds from three considerations, and take the tightest:**

1. **Evidence.** Do not let the optimiser go beyond the observed spend range by more than
   ~50%. Outside that range the curve is the prior's shape, not a finding.
2. **Executability.** Media cannot triple overnight. Inventory is finite, contracts exist,
   teams have capacity. TV upfronts are committed months ahead; OOH is bought in fixed
   cycles.
3. **Risk.** Cutting a channel to zero loses the option to learn about it, and often loses
   rate cards and partnerships that are expensive to rebuild.

A practical default is ±30% of current spend, widened for channels with strong evidence
and narrowed for channels with wide intervals.

**Never allow zero unless zero is genuinely on the table.** The saturation curve near the
origin is almost always extrapolation, especially for always-on channels, and a
recommendation to switch a channel off is the one most likely to be executed
irreversibly.

---

## 4. Constraints beyond bounds

Real plans carry structure the optimiser needs to be told about:

* **Minimum viable spend.** Below a threshold a channel cannot buy meaningful reach.
* **Committed spend.** Upfronts, sponsorships, contracted OOH.
* **Portfolio rules.** "At least 40% upper-funnel", "brand cannot fall below x".
* **Group budgets.** Total digital, total offline.
* **Frequency caps.** Meridian only.

pymc-marketing: `opt.set_constraints([...])`. Meridian: per-channel spend constraints.
Robyn: `channel_constr_low` / `channel_constr_up` per channel.

---

## 5. Optimising with uncertainty

The optimiser typically works on the posterior mean response surface, which throws away
exactly the information that should make you cautious. Two ways to put it back:

**Optimise per draw.** Run the allocation for a sample of posterior draws and look at the
distribution of recommended allocations. A channel whose recommended budget ranges from
£50k to £400k across draws has not been measured well enough to optimise.

**Evaluate the recommendation under uncertainty.** Take the optimal allocation and compute
the posterior distribution of its response. Compare against the current allocation's
distribution. If the intervals overlap substantially, the "12% uplift" headline is not
distinguishable from noise, and the honest recommendation is a smaller move plus a test.

Neither is built into the frameworks. Both are worth the effort before a plan that moves
real money.

---

## 6. Sensitivity and scenarios

```python
mmm.sensitivity.run_sweep(
    "channel_data", sweep_values=np.linspace(0.5, 1.5, 11),
    var_names="channel_contribution", sweep_type="multiplicative",
)
mmm.plot.sensitivity_analysis()
```

Scenarios that earn their place in a readout:

* **Current plan** — the baseline everything is compared to.
* **Optimal, unconstrained within evidence** — the ceiling.
* **Optimal, constrained to executable moves** — the realistic recommendation.
* **±20% total budget** — answers "what if finance cuts us?" before it is asked.
* **Channel off** — for any channel under scrutiny.
* **Marginal ROAS by channel at plan** — shows *why* the recommendation is what it is.

The last one is the most persuasive artefact in a budget conversation: a chart of marginal
return at current spend explains the whole recommendation without anyone needing to trust
the optimiser.

---

## 7. Before acting on an optimiser's output

- [ ] The model passed its diagnostics — convergence, baseline, CV gap.
- [ ] The baseline is plausible and never negative.
- [ ] Bounds keep every channel inside, or near, its observed spend range.
- [ ] The optimisation window is at least as long as the effective carryover.
- [ ] The recommendation was checked under posterior uncertainty, not just at the mean.
- [ ] Channels with wide intervals are not receiving large moves.
- [ ] The recommendation is executable — inventory, contracts, lead times.
- [ ] Someone who knows the media market has sanity-checked it.
- [ ] The plan is staged, with a measurement point before the next stage.

**Stage the change.** Moving 30% of a budget on a model's say-so is a bet on a model.
Moving 10%, measuring, and then moving again is a measurement programme — and the first
move creates exactly the spend variation the next model needs.

---

## 8. Failure modes

| Symptom | Cause |
|---|---|
| Recommends zero for a channel | Unbounded optimiser extrapolating below observed spend |
| Recommends a tenfold increase | Same, in the other direction; the curve is flat there and the model does not know it |
| Long-carryover channels are under-funded | Window shorter than `effective_carryover_lags()` |
| Uplift looks implausibly large | Extrapolation, or a saturated channel modelled as linear |
| Allocation swings between refreshes | The underlying model is not identified |
| Optimiser fails to converge | Infeasible constraints, or bounds that do not admit the total budget |
| Recommendation contradicts the media team | Sometimes the finding; more often a missing constraint they know about and the model does not |

---

## 9. Presenting the result

Three things, in this order:

1. **The marginal return chart.** Why the recommendation exists.
2. **The move.** Current vs recommended per channel, in money, with the constraint that
   bound each one.
3. **The expected gain, with its interval**, and what would falsify it.

And one sentence that keeps the conversation honest:

> *This is the best allocation given what the model can see. The channels with the widest
> ranges are the ones we have never tested; a geo holdout on [channel] would narrow the
> single largest source of uncertainty in this plan.*

---

## Related skills

`mmm-attribution` for marginal vs average return; `mmm-validation` for whether the model
is fit to optimise on; `mmm-experimentation-calibration` for closing the loop;
`mmm-api-reference` §9 for exact signatures.
