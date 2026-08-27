---
name: mmm-attribution
description: |
  Turning a fitted MMM into channel contributions, ROAS, CPA, marginal returns and response curves — correctly. Use when extracting contributions, computing return metrics with uncertainty, reading saturation curves, decomposing the target, comparing efficiency across channels, or diagnosing an implausible attribution result. Covers pymc-marketing 1.x counterfactual incrementality and the difference between average and marginal return.
---

# Attribution and Interpretation

Attribution is where a fitted model becomes a decision, and where most of the errors that
reach a boardroom are introduced. Four of them account for nearly all the damage:

1. Dividing raw contributions by spend instead of computing a counterfactual.
2. Forgetting that posterior contributions are normalised.
3. Reporting average ROAS when the decision needs marginal ROAS.
4. Reporting point estimates for quantities whose intervals span the decision.

---

## 1. Counterfactual incrementality — the right way

```python
incr = mmm.incrementality

roas  = incr.contribution_over_spend(frequency="quarterly")
cpa   = incr.spend_over_contribution(frequency="monthly")
mroas = incr.marginal_contribution_over_spend(frequency="quarterly")
inc   = incr.compute_incremental_contribution(frequency="monthly", include_carryover=True)
```

Each returns an xarray object carrying the full posterior — dims `(chain, draw, date,
channel)` — so you get intervals, not a point.

It works by re-running the model with spend perturbed and taking the difference. That
matters for three reasons:

* **It respects the link function.** Under a log link, components combine
  multiplicatively and `contribution / spend` is simply the wrong arithmetic.
* **It counts carryover that lands outside the window.** A quarter's TV spend is still
  working in the next quarter; `include_carryover=True` counts it.
* **It answers the question actually asked** — "what would have happened without this
  spend" — rather than "what number did the model assign to this term".

Use `start_date` / `end_date` to scope a period, and `num_samples` to trade precision for
speed.

## 2. Direct contributions, when you need them

```python
scale = float(mmm.get_scales_as_xarray()["target_scale"])
contrib = mmm.idata["posterior"]["channel_contribution"] * scale   # identity link only

df  = mmm.compute_mean_contributions_over_time(central_tendency="median")
ds  = mmm.compute_counterfactual_contributions_dataset()           # link-aware, any link
```

**`channel_contribution` in the posterior is normalised.** Multiply by `target_scale`
before comparing to anything real. A contribution of 0.13 is not 13% and not £0.13 — it is
0.13 max-abs-scaled units.

**`intercept_contribution` has no `date` dimension** when the intercept is constant. It
applies in every period, so it must be broadcast across dates before summing. Forgetting
that understates the baseline by a factor of *n periods* — which is exactly how a model
ends up "showing" that media drives 90% of sales.

```python
from agent_mmm.diagnostics import decompose
dec = decompose(mmm.idata)      # handles scaling and broadcasting
print(dec["shares"])            # intercept / trend / seasonality / controls / media
```

Under a **log link**, do not use `add_original_scale_contribution_variable` on a component:
it gives `exp(component) * target_scale`, a multiplicative factor rather than an additive
contribution. Use `compute_counterfactual_contributions_dataset()`.

## 3. Summary factory

`mmm.summary` is a ready `MMMSummaryFactory`:

```python
mmm.summary.contributions()      mmm.summary.roas()
mmm.summary.channel_share_hdi()  mmm.summary.total_contribution()
mmm.summary.channel_spend()      mmm.summary.waterfall()
mmm.summary.saturation_curves()  mmm.summary.change_over_time()
mmm.table()
```

---

## 4. Average vs marginal return — not the same question

| Metric | Question | Use for |
|---|---|---|
| **Average ROAS** | What did this channel return overall? | Reporting, historical accountability |
| **Marginal ROAS** | What would the *next* pound return? | **Budget decisions** |

On any concave saturation curve, marginal ROAS is below average ROAS, and the gap widens
the more saturated the channel. A channel with average ROAS 4.0 and marginal ROAS 0.9 is a
past success and a present mistake — more money into it destroys value.

**Budget reallocation is a marginal question.** Optimal allocation equalises marginal
return across channels, not average return. Ranking channels by average ROAS and shifting
budget to the top of that list systematically over-funds saturated channels.

```python
mroas = mmm.incrementality.marginal_contribution_over_spend(frequency="quarterly")
```

For non-monetary targets, the same distinction holds for CPA: average CPA vs the cost of
the next acquisition.

---

## 5. Uncertainty is the deliverable

```python
import arviz as az
roas = mmm.incrementality.contribution_over_spend(frequency="quarterly")
mean = roas.mean(dim=("chain", "draw"))
ci   = az.hdi(roas, hdi_prob=0.89)

# The question a decision-maker is actually asking:
p_above_1 = float((roas > 1.0).mean())
```

`P(ROAS > 1)` and `P(channel A's ROAS > channel B's)` are more useful than a point
estimate, because they answer the decision directly. "TV's ROAS is 2.1" invites a
challenge; "there is an 85% chance TV returns more than it costs, and a 60% chance it
beats display" invites a decision.

A channel whose 89% interval spans 0.5 to 5.0 has not been measured. Say so, and propose
the experiment that would narrow it, rather than quoting the midpoint.

---

## 6. Response curves

```python
curve = mmm.sample_saturation_curve(
    max_value=1.0,          # SCALED space: original_max / channel_scale.mean()
    num_points=100, num_samples=500, original_scale=True,
)
mmm.plot.saturation_curves(curve=curve)
mmm.plot.saturation_curves_scatter()   # with the observed spend points overlaid
mmm.plot.marginal_curve()
```

Read three things:

* **Where current spend sits.** On the steep part means headroom; on the flat part means
  saturation.
* **How wide the band is at that point.** A wide band at current spend means the curve is
  not identified where it matters.
* **Where the observed data stops.** Everything beyond the rightmost scatter point is
  extrapolation from the prior's functional form, not evidence.

The scatter overlay is the most useful of the three plots precisely because it shows the
last point. A recommendation to double spend on a channel whose data stops at current
levels is a recommendation about your prior.

---

## 7. Decomposition

```python
mmm.plot.waterfall_components_decomposition()
mmm.plot.contributions_over_time()
mmm.plot.channel_contribution_share_hdi()
```

Under an identity link the components sum exactly to the fitted target. Under a log link
they do not, for the reason in §2.

**Read the baseline before any channel number.** If it is negative, or below ~30%, or above
~95% of the target, the channel numbers are downstream of a broken decomposition and none
of them is usable. See `mmm-baseline-and-trend`.

---

## 8. Efficiency framing

For each channel, four numbers together:

| Number | Meaning |
|---|---|
| Contribution share | % of media effect |
| Spend share | % of budget |
| Efficiency ratio | contribution share ÷ spend share; > 1 is over-performing its budget |
| Marginal ROAS | what the next pound returns |

| Pattern | Read as | Action |
|---|---|---|
| High mROAS, low spend share, steep curve | Under-invested | Increase, in steps you can measure |
| Low mROAS, high spend share, flat curve | Saturated | Reduce and redeploy |
| High average ROAS, low mROAS | Past success, present saturation | Hold; do not scale |
| Wide interval, any mean | Not measured | Test before acting |

The last row is the one most often ignored. A channel with a wide interval should trigger
an experiment, not a budget move in either direction.

---

## 9. Target units

For non-monetary targets, ROAS is undefined without a value per unit.

| Target | Primary metric | Secondary |
|---|---|---|
| Revenue | ROAS | Incremental revenue |
| Conversions / signups | **CPA** (cost per incremental acquisition) | ROAS if `value_per_unit` is known |
| Volume | Cost per incremental unit | Same |

```python
cpa = mmm.incrementality.spend_over_contribution(frequency="quarterly")
```

If a value per unit exists, report both and state which one is the assumption. Meridian
calls this CPIK (cost per incremental KPI). See `mmm-target-units`.

**Organic and owned channels have no spend, so they have no ROAS.** Reporting "email ROAS
= 40" is a division by a number that does not exist. Report incremental units per thousand
sends.

---

## 10. Diagnosing an implausible result

| Result | Usual cause | Check |
|---|---|---|
| Brand search has the highest ROAS | Harvesting demand upstream media created | Funnel structure; run a brand-search holdout |
| A channel is negative | Spend dated by invoice not delivery, or collinearity | Date alignment first, then the correlation matrix |
| One channel takes >70% (with ≥3 channels) | Collinearity | VIF, pairwise correlation |
| Media explains >60% of the target | Starved baseline — usually missing price or distribution | `decompose()`, then the control set |
| Contributions do not sum to the target | Normalised scale, un-broadcast intercept, or a log link | `decompose()` handles all three |
| ROAS is orders of magnitude off | Contributions not rescaled by `target_scale` | Compare against total observed target |
| Everything looks excellent | Uncontrolled seasonality or trend correlating with everything | Baseline share |
| Retargeting beats prospecting | Selection, not effect | Holdout test |

---

## 11. Communicating attribution

**Say what kind of number it is.** MMM ROAS is *incremental* — what would not have
happened otherwise. It is not comparable to a platform's last-click ROAS, which counts
conversions the platform can see and claims users it targeted. The two differ by a factor
of several for lower-funnel channels, and presenting them side by side without that caveat
starts an argument nobody wins.

**Lead with the range.** "TV returns between £1.60 and £2.40 per pound, most likely £2.00"
survives scrutiny. "TV ROAS is 2.0" invites someone to find a week where it was not.

**Separate measured from inferred.** Channels calibrated to an experiment deserve more
confidence in the deck than channels the model inferred from historical variation. Say
which is which.

**Do not compare across models.** ROAS from a revenue model and ROAS from a
search-volume model measure different outcomes and are not on the same scale.

---

## Related skills

`mmm-baseline-and-trend` — read the baseline first.
`mmm-budget-optimization` — turning marginal returns into an allocation.
`mmm-causal-design` — why a coefficient may not be a causal effect.
`mmm-stakeholder-reporting` — audience-specific framing.
`mmm-api-reference` §7 — exact signatures.
