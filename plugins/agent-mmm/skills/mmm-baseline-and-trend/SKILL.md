---
name: mmm-baseline-and-trend
description: |
  The baseline of an MMM — intercept, trend, seasonality and controls — and how it competes with media for variance. Use when the baseline is negative, implausibly large or small, when deciding between a fixed intercept, a linear trend, changepoints, knots or a time-varying intercept, when seasonality is absorbing media's effect, when reconciling the decomposition, or when explaining "what would have happened anyway" to a stakeholder.
---

# Baseline and Trend: the part of the model nobody checks

Every MMM answers one question — *what would have happened anyway?* — and everything
else is arithmetic on the remainder. The baseline is that answer. Almost every
implausible ROAS in a finished model is a baseline problem wearing a channel's name.

An MMM decomposes the target as:

```
target = baseline + media + noise
baseline = intercept + trend + seasonality + controls + non-media treatments
```

The baseline and media compete for exactly the same variance. Anything you let the
baseline absorb is unavailable to media, and anything you leave out of the baseline gets
attributed to whichever channel happened to move at the same time. There is no neutral
choice here; every baseline specification is a prior about how much marketing does.

---

## 1. Check the baseline first, always

Before reading a single ROAS:

| Check | Healthy | What a failure means |
|---|---|---|
| Baseline never negative | 0 negative periods | The model claims the business would sell less than nothing without marketing. Structurally broken. |
| Baseline share of target | ~50–90% for an established brand | Below 30%: media is being credited with the business. Above 95%: media has no room left. |
| Baseline drift start→end | within ±50% | Large unexplained drift means a missing structural driver, and makes every forward-looking recommendation fragile. |
| Baseline shape | smooth, or steps you can name | Wiggles that track campaign flighting mean the baseline is eating media. |
| Seasonality amplitude | matches known seasonality | An implausibly large seasonal term is absorbing seasonal campaigns. |

```python
from agent_mmm.diagnostics import decompose, check_baseline
dec = decompose(mmm.idata)          # totals and shares in target units
print(dec["shares"])                # intercept / trend / seasonality / controls / media
print(check_baseline(mmm.idata, decomposition=dec))
```

In Meridian: `analyzer.baseline_summary_metrics()` and
`analyzer.negative_baseline_probability()` — the latter exists precisely because the
failure is common enough to warrant its own diagnostic.

---

## 2. Negative baseline

**What it means.** The fitted intercept-plus-controls path drops below zero in some
periods. Since the target cannot be negative, media contributions must be making up the
difference — the model is borrowing from an impossible counterfactual to pay media.

**Consequences.** Every channel's contribution is inflated by whatever the baseline went
short. ROAS, contribution shares and the budget optimiser are all downstream of this, so
none of them is usable.

**Causes, in the order to check them:**

1. **A missing trend.** The business grew or shrank and the model has only a constant
   intercept. Media that also grew absorbs the trend, and the intercept is pushed down
   to compensate.
2. **A missing structural driver.** Distribution expanded, a competitor left, price
   changed, a pandemic happened.
3. **Over-flexible seasonality.** Many Fourier modes can swing far below zero between
   knots.
4. **An unconstrained control with a large negative coefficient.**
5. **Priors that force media too high.** A calibration constraint or a strong ROI prior
   can make the media term larger than the data supports, and the intercept absorbs the
   residual.

**Fixes.** Add the missing driver. If none exists, add a trend (see §3) — but note that a
trend is an admission that something real is missing, not an explanation. Some
practitioners force the baseline non-negative with a link function or a positive-support
prior on the intercept; that removes the symptom without removing the cause, and should
be a last resort with a note in the report.

---

## 3. Choosing a trend specification

| Option | What it says | Cost |
|---|---|---|
| **No trend** | The business level is constant | Any real drift is attributed to media |
| **Linear term** | Steady growth or decline | Cannot bend; extrapolates badly |
| **`LinearTrend(n_changepoints=k)`** | Piecewise-linear, bending at k points | k more parameters; changepoints near campaign shifts steal media's signal |
| **Knots (Meridian)** | Piecewise-linear time effects at chosen knots | `knots=None` on a geo model means *one knot per period* — a baseline that can fit anything |
| **`time_varying_intercept=True` (HSGP)** | Smoothly varying baseline | Very flexible; needs 104+ periods, and lengthscale choice is a modelling decision, not a default |

**The trade-off is the whole game.** A flexible baseline explains the data better and
attributes less to media. A rigid baseline attributes more to media, some of which media
did not do. Neither is "correct" — the specification encodes your belief about how much
of the variation marketing is responsible for.

Practical guidance:

* Prefer **explaining** the drift to **absorbing** it. A distribution variable is worth
  more than any trend term, because it is a driver you can act on and a trend is not.
* Set the flexibility **coarser than your media flighting**. If campaigns turn over
  monthly, a baseline that can bend monthly will absorb them. Lengthscales of a quarter
  or more (`ls_mu` ≈ 13 weekly periods) keep the two separable.
* In Meridian, **never leave `knots=None` on a geo model** without deciding it. That
  gives one coefficient per time period, which is the most flexible baseline available.
  Set an explicit number, or `enable_aks=True` to let the automatic knot selection
  choose.
* **Test the sensitivity.** Fit with and without the flexible baseline and compare the
  media share. If it moves by a third, the trend specification — not the data — is
  driving your conclusion, and the report has to say so.

### Long-term brand effects

A time-varying intercept is also how brand-building shows up: sustained upper-funnel
investment raises the base level rather than producing week-by-week spikes. That means a
rigid baseline *under*-credits brand media, and a flexible one absorbs it entirely. Neither
extreme is right, and no purely observational model resolves it. If long-term effects are
the question, you need a longer series (3+ years), a brand-tracking variable, or an
explicit two-stage structure — not a better trend term.

---

## 4. Seasonality

**Fourier terms** capture smooth annual shape. `yearly_seasonality=n` adds `2n`
parameters. More modes fit sharper shapes and absorb more variance.

* Weekly data: 4–8 modes is usually right. Start at 4.
* Strong, spiky seasonality: do **not** just raise the order. Fourier terms fit sharp
  peaks by oscillating, which produces ringing either side of the peak and can push the
  baseline negative.
* Very weak seasonality: 2–4 modes, or none. Extra modes will fit noise.

**Named events** — Christmas, Black Friday, Ramadan, Golden Week, back-to-school, tax
deadlines — should be explicit 0/1 indicators, not left to Fourier. They are step
changes, and they are exactly when campaigns run. Without a flag, the model gives Black
Friday's sales to whatever media ran during Black Friday.

**Daily data** additionally needs day-of-week terms (`WeeklyFourier`), and paydays or
month-end effects in some categories.

```python
from agent_mmm.data_prep import add_calendar_features, add_event_flags
df = add_calendar_features(df, "date", country="GB")
df = add_event_flags(df, "date", {"black_friday_2024": ("2024-11-25", "2024-12-02")})
```

**The diagnostic**: if seasonal strength is high *and* campaigns are flighted seasonally,
seasonality and media are partly confounded. Check the correlation between each channel's
spend and the fitted seasonal component. Where it is high, the split between them is
determined by your priors, and an experiment during the season is the only way to settle it.

---

## 5. Controls: what belongs in the baseline

A control earns its place if it (a) moves the target and (b) correlates with media.
Something that fails (a) adds noise; something that fails (b) is harmless but useless.

| Category | Examples | Why it matters |
|---|---|---|
| Pricing | price index, discount depth, promo flag | Planned with media. **The most common omission.** |
| Distribution | ACV, store count, stock availability | Moves sales independently; correlates with launch media |
| Competitive | competitor spend, share of voice, competitor promotions | Response to competition looks like media wear-out |
| Macro | consumer confidence, unemployment, category volume, interest rates | Slow-moving; can be collinear with trend |
| Operational | delivery times, site outages, call-centre capacity | Explains dips media gets blamed for |
| Calendar | holidays, events, fiscal periods | Sharp, and campaign-adjacent |
| Weather | temperature, rainfall anomaly | Large in food, drink, apparel, garden, DIY |

Two cautions:

* **Controls are not scaled** by pymc-marketing (target and channels are). A control on a
  wildly different scale makes the default `gamma_control` prior mean something you did
  not intend. Standardise controls yourself.
* **Do not control for a mediator.** A variable that marketing *causes* — website
  sessions, brand search volume, app installs — will absorb marketing's effect if you put
  it in as a control. That is not adjustment, it is deletion. See the causal-design skill.

---

## 6. Reconciling the decomposition

The components should reconstruct the fitted target. Under an identity link they sum
exactly:

```
intercept + trend + seasonality + controls + media == fitted target
```

Two things break this legitimately:

* **Log link.** Components combine multiplicatively, so per-component counterfactuals sum
  to *more* than the total — interactions are counted by every component that
  participates. Use `compute_counterfactual_contributions_dataset()` and say so in the
  report.
* **Normalised scale.** `channel_contribution` in the posterior is in max-abs-scaled
  units. Multiply by `target_scale` before comparing to anything real. A time-constant
  intercept has no `date` dimension and must be broadcast across periods before it is
  summed — forgetting that understates the baseline by a factor of *n periods*, which is
  how a model ends up "showing" that media drives 90% of sales.

`agent_mmm.diagnostics.decompose()` handles both and is the single source the baseline
and attribution checks share, so they cannot report contradictory shares.

---

## 7. Explaining the baseline to stakeholders

The baseline is the most misunderstood number in the deck. Two framings that work:

* *"If we switched off all measured marketing and everything else stayed the same, the
  model estimates the business would still do about 70% of current sales. That is brand
  equity, distribution, repeat purchase, and demand we did not create this quarter."*
* *"The baseline is not 'free' sales. It is the accumulated result of everything we have
  done before, plus everything outside marketing. It is also not fixed — if brand
  investment stops, it erodes on a timescale longer than this model can see."*

What to avoid: calling the baseline "organic", which implies it is costless and permanent.
It is neither, and treating it as a floor is how brand budgets get cut.

---

## 8. Diagnostic checklist

```
Baseline negative anywhere?
  -> Add the missing driver (distribution, price, competitor, regime change)
  -> Failing that, add a trend and note the assumption
  -> Reduce Fourier order if the negative periods sit between seasonal peaks

Baseline < 30% of target?
  -> An omitted driver is being credited to media. Which one moves with the media plan?
  -> Check the intercept prior: centred near 1.0 in scaled space starves media,
     centred near 0 starves the baseline

Baseline > 95% of target?
  -> Too-flexible trend or seasonality. Coarsen it and refit
  -> Or media genuinely does very little; an experiment will tell you which

Baseline tracks campaign flighting?
  -> The trend is absorbing media. Coarsen the lengthscale / reduce changepoints

Baseline drifts > 50% with no explanation?
  -> A structural variable is missing. Forecasts from this model are not trustworthy

Media share moves a lot when the trend spec changes?
  -> Your conclusion is a modelling choice. Report the range, not a point,
     and prioritise an experiment
```

---

## Related skills

`mmm-causal-design` for mediators and controls; `mmm-channel-semantics` for what belongs
in the baseline versus in media; `mmm-diagnostics` for the automated checks;
`mmm-experimentation-calibration` for resolving a baseline/media split the data cannot.
