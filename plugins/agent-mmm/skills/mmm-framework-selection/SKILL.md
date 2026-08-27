---
name: mmm-framework-selection
description: |
  Choosing between pymc-marketing, Google Meridian, Meta Robyn and other MMM frameworks, and migrating between them. Use when starting a project and deciding which framework to build in, when a client already has a model in another framework, when reconciling results across frameworks, when asked "should we use Meridian or Robyn", or when a spec feature cannot be expressed in the chosen framework.
---

# Choosing an MMM Framework

The framework matters less than the data, the causal specification, and whether you have
an experiment — those decide whether the answer is right. The framework decides what you
*can express*, how uncertainty is reported, and how much work a given question costs.

Versions verified August 2026: **pymc-marketing 1.1.0**, **google-meridian 1.8.0**,
**Robyn 3.12.1**. LightweightMMM is archived and should not be used for new work.

---

## 1. Decision, in order

```
Do you have a geo panel with population data?
  └─ yes, and reach/frequency matters ......................... Meridian
  └─ yes, and you need custom structure or a log link ......... pymc-marketing
  └─ yes, otherwise .......................................... Meridian or pymc-marketing

National only?
  └─ R team, need fast exploration + a strong allocator ....... Robyn
  └─ need real uncertainty, custom priors, mediation .......... pymc-marketing
  └─ want ROI priors and a guided workflow ................... Meridian (national mode)

Non-negotiable requirements that force the choice:
  Reach & frequency optimisation ............................. Meridian (only option)
  Multiplicative / log-link model ............................ pymc-marketing (only option)
  DAG-based causal identification ............................ pymc-marketing (only option)
  Funnel / mediation structure ............................... pymc-marketing (only option)
  Arbitrary custom priors or likelihood ...................... pymc-marketing
  Hierarchical geo pooling ................................... Meridian or pymc-marketing
  Must run in R .............................................. Robyn
```

---

## 2. Comparison

| | **pymc-marketing 1.1** | **Meridian 1.8** | **Robyn 3.12** |
|---|---|---|---|
| Language / backend | Python / PyMC | Python / TF Probability | R / glmnet + Nevergrad |
| Paradigm | Bayesian | Bayesian | Frequentist (ridge) |
| Geo hierarchy | `dims=("geo",)` | native, population-weighted | **none** |
| Media prior | coefficient | **ROI / mROI / contribution / coefficient** | bounds + signs |
| Uncertainty | posterior | posterior | Pareto spread + bootstrap CIs |
| Adstock | geometric, delayed, Weibull CDF/PDF, binomial, none | geometric, binomial | geometric, Weibull CDF/PDF |
| Per-channel adstock | yes (`MediaConfigList`) | yes (`adstock_decay_spec` map) | **no** |
| Saturation | 9 forms | Hill or none | Hill only |
| Time effects | Fourier, LinearTrend, HSGP | knots (+ auto selection) | Prophet |
| Link function | identity or **log** | identity | identity |
| Reach & frequency | no | **native + optimal frequency** | no |
| Organic media | via role discipline | **own slot** | `organic_vars` |
| Non-media treatments | via `control_columns` | **own slot** | `context_vars` + signs |
| Experiment calibration | **likelihood term** (`add_lift_test_measurements`) | ROI prior + calibration period | third objective |
| DAG / causal ID | **yes** (dowhy) | no | no |
| Mediation / funnel | **yes** | no | no |
| Built-in data guardrail | audit is yours to run | **EDA guardrail blocks bad fits** | input checks |
| Budget optimiser | `mmm.budget_optimizer()` | `BudgetOptimizer` + scenarios | `robyn_allocator` (excellent) |
| Cross-validation | `TimeSliceCrossValidator` | `holdout_id` | `ts_validation` |
| Speed | moderate | moderate (GPU helps) | **fast** |
| Reporting artefacts | plot suite + summary factory | **HTML summary + optimisation reports** | one-pagers + allocator plots |
| Learning curve | steep (a toolkit) | moderate (an application) | moderate |

---

## 3. What each is genuinely best at

**pymc-marketing — the modelling toolkit.**
Choose it when the model needs to say something the others cannot express: a
multiplicative structure, a funnel where TV acts through brand search, a DAG-derived
adjustment set, a custom likelihood, an R2D2 prior that fixes the baseline/media variance
split directly, or a lift test entering as a hard likelihood constraint rather than a
preference. It is also the only one where "add a component" is a normal afternoon's work.

The cost is that it will let you build a bad model without complaint. There is no
guardrail; the discipline is yours.

**Meridian — the well-designed application.**
Choose it when you have a geo panel and want the structurally correct thing by default:
population weighting, ROI priors that make experiment calibration a one-liner, separate
slots for organic and non-media variables, native reach and frequency, and an EDA
guardrail that refuses to fit data with critical problems. Its HTML summary and
optimisation reports are the best out-of-the-box stakeholder artefacts of the three.

The cost is rigidity. No log link, no Fourier seasonality, no mediation, fixed parameter
families, and a heavy TensorFlow dependency.

**Robyn — the fast explorer.**
Choose it for national models in an R shop, when you want to sweep many specifications
quickly and the allocator's output is the deliverable. Its refresh workflow keeps
quarterly updates stable, and JSON model persistence makes results genuinely
reproducible.

The cost is that it has no geo hierarchy, no posterior, and a built-in prior
(DECOMP.RSSD) that pushes every channel's effect share towards its spend share.

**Do not use LightweightMMM.** Archived. Meridian is Google's successor to it.

---

## 4. Running two frameworks

Worth it when the decision is large and the frameworks disagree in ways you can learn
from. Not worth it as a routine.

**How to compare honestly:** compare *decisions*, not point estimates. "Both say to shift
budget from affiliate to prospecting" is a robust finding. "One says TV ROAS 2.1 and the
other 2.6" is noise from different structural assumptions, and averaging them produces a
number neither model supports.

**Expect disagreement from known causes:**

| Source | Effect |
|---|---|
| Baseline flexibility (knots vs Fourier vs HSGP) | Often the largest single driver of the media share |
| Saturation family (Hill vs logistic) | Changes marginal ROAS more than average ROAS |
| Adstock family | Shifts the timing of credit between channels |
| Prior parameterisation (ROI vs coefficient) | Different implied beliefs even when both feel "uninformative" |
| Geo pooling vs national aggregation | Pooling usually narrows intervals and can move the point |
| DECOMP.RSSD (Robyn only) | Pulls effect shares towards spend shares |

If the frameworks disagree about a decision, that is the finding: the data does not
identify the answer, and the next step is an experiment, not a third model.

---

## 5. Migration

`agent_mmm` keeps the spec framework-agnostic so a project can be recompiled rather than
rebuilt:

```python
from agent_mmm.model_factory import compile_spec
spec.framework = "meridian"
result = compile_spec(spec, priors=priors, build=False)
print(result.code)          # runnable Meridian script
print(result.unsupported)   # what Meridian cannot express from this spec
print(result.data_contract) # the shape it needs
```

`result.unsupported` is the important field. A spec feature that vanishes silently is how
a model ends up not meaning what its author thinks.

### Migration notes

**pymc-marketing → Meridian.** Fourier seasonality has no equivalent — decide a knot
count. A delayed adstock maps to `binomial` with a different shape. A log link cannot be
ported. Coefficient priors must be re-expressed as ROI priors. Gains: geo hierarchy, RF,
explicit organic/non-media slots.

**Meridian → pymc-marketing.** ROI priors must be converted to coefficient priors, or
replaced with lift-test constraints (usually better). Knots become `LinearTrend` or
`time_varying_intercept`; choose a lengthscale coarser than your media flighting. RF
channels lose the frequency dimension — model reach, and say frequency is out of scope.

**Robyn → Bayesian.** Hyperparameter bounds become priors: centre on the middle of the
bound, set sigma from its width, and note that this invents information the bounds did not
carry. `context_signs` become prior supports. Prophet terms become Fourier terms plus
explicit holiday flags. The Pareto spread does not convert into a posterior — do not try.

**Anything → Robyn.** Priors collapse to bounds. Geo structure is lost. Per-channel
adstock is lost. Expect the DECOMP.RSSD objective to move results towards spend shares.

---

## 6. Common questions

**"Meridian is from Google, so is it better?"**
It is better *at what it is designed for*: geo-hierarchical models with ROI priors and RF.
It is not a general Bayesian modelling toolkit and does not try to be.

**"Robyn is free and fast, why not just use it?"**
For a national model with lift tests and an allocation question, it is a reasonable
choice. It cannot do geo hierarchies and it does not produce a posterior; if either
matters, it is the wrong tool.

**"Can we just average the frameworks?"**
No. They embed different structural assumptions, so the average is a model nobody
specified. Compare decisions, report the range, and resolve the disagreement with an
experiment.

**"The client already has a Robyn model."**
That is a brownfield project. Extract its effect shares and hyperparameters as *priors*
for the new model, keep the channel taxonomy so results are comparable, and expect to
explain why the numbers moved. See `mmm-greenfield-vs-brownfield`.

**"Which is best for a first MMM?"**
Meridian if you have geo data — the guardrails and defaults prevent the most common
mistakes. pymc-marketing if you have a strong modelling background and need flexibility.
Either way, spend the first week on data and the causal specification, not on the
framework.
