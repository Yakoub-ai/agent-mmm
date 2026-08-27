---
name: mmm-robyn
description: |
  Meta Robyn reference and practice guide, verified against Robyn 3.12.1 (R). Use when building, reviewing or debugging a Robyn model, setting hyperparameter bounds, interpreting the Pareto front and DECOMP.RSSD, using calibration_input, running robyn_allocator or robyn_refresh, or translating a model between Robyn and a Bayesian framework.
---

# Meta Robyn (verified against v3.12.1, CRAN, May 2026)

Robyn is Meta's open-source MMM: R, ridge regression, hyperparameters searched with an
evolutionary algorithm (Nevergrad, via reticulate). It is **frequentist**, and that single
difference propagates through everything below.

Install: `install.packages("Robyn")` plus Python's `nevergrad` via reticulate.
Requires R ≥ 4.0.

---

## 1. What "frequentist" changes

| | Bayesian (pymc-marketing, Meridian) | Robyn |
|---|---|---|
| Prior knowledge | distributions | hyperparameter **bounds** + sign constraints |
| Output | a posterior | a **Pareto front of many models** |
| Uncertainty | credible intervals | spread across the Pareto set + bootstrapped decomposition CIs |
| Regularisation | priors | ridge penalty, lambda selected by CV |
| Experiment calibration | likelihood term / prior | a third **optimisation objective** |
| Model selection | model comparison | pick one solution from the front |

Two practical consequences:

* **Bounds lose information.** A tight belief and a vague one with the same range become
  the same object. There is no way to say "probably 0.6, but I am not sure".
* **The Pareto front is not a posterior.** Its spread reflects which hyperparameters the
  search happened to find that trade off the objectives, not a probability distribution
  over parameter values. Reporting it as if it were a credible interval is a category
  error, and a common one.

Robyn's real strength is speed of exploration and its allocator; it is a strong choice
when you need many national models quickly and have lift tests to anchor them.

---

## 2. The three objectives

Robyn optimises simultaneously:

* **NRMSE** — fit.
* **DECOMP.RSSD** — distance between each channel's *effect share* and its *spend share*.
* **MAPE (lift)** — only when `calibration_input` is supplied; distance from your measured
  lifts.

**DECOMP.RSSD deserves scrutiny.** It penalises solutions where a channel's contribution
share differs from its spend share, on the reasoning that a wildly different split is
usually a modelling artefact. That is a *prior* — a strong one, expressed as a penalty —
and it biases results towards "everything performs about as well as its budget implies".
If your hypothesis is that one channel is dramatically mis-invested, this objective works
against finding it. `rssd_zero_penalty = TRUE` additionally penalises zero-coefficient
channels.

---

## 3. Inputs

```r
InputCollect <- robyn_inputs(
  dt_input = dt_input,
  dep_var = "revenue",
  dep_var_type = "revenue",          # or "conversion"
  date_var = "date",
  paid_media_spends = c("tv_spend", "sem_spend", "social_spend"),
  paid_media_vars   = c("tv_grps",  "sem_clicks", "social_impressions"),
  paid_media_signs  = c("positive", "positive", "positive"),
  organic_vars  = c("email_sends", "organic_social"),
  organic_signs = c("positive", "positive"),
  context_vars  = c("price_index", "competitor_spend", "acv"),
  context_signs = c("negative", "negative", "positive"),
  factor_vars   = c("promo_flag"),
  prophet_country = "GB",
  prophet_vars  = c("trend", "season", "holiday"),
  prophet_signs = c("default", "default", "default"),
  adstock = "geometric",             # geometric | weibull_cdf | weibull_pdf
  window_start = "2023-01-02",
  window_end   = "2025-12-29"
)
```

* `paid_media_spends` and `paid_media_vars` are **parallel and aligned**. Vars may be
  exposure; spends are always cost.
* **Organic variables have no spend and therefore no ROAS.** Robyn enforces this
  structurally, which is more than most workflows manage.
* `context_vars` carries confounders *and* business levers. Use `context_signs` — a price
  variable with an unconstrained sign is a wasted constraint.
* **Prophet** supplies trend, seasonality and holidays. Do not also hand-build calendar
  dummies; you will double-count and starve media.
* `window_start` / `window_end` set the modelling window while keeping earlier rows
  available for adstock warm-up — the same idea as Meridian's longer media axis.

---

## 4. Hyperparameters

Bounds, not priors. `hyper_names(adstock, all_media)` generates the required names.

| Adstock | Parameters |
|---|---|
| `geometric` | `<channel>_thetas` — decay per period, [0, 1) |
| `weibull_cdf` / `weibull_pdf` | `<channel>_shapes`, `<channel>_scales` |

Saturation is always Hill: `<channel>_alphas` (exponent; > 1 gives an S-curve) and
`<channel>_gammas` (inflection as a fraction of the channel's range).

```r
hyperparameters <- list(
  tv_spend_thetas = c(0.6, 0.9),      # long carryover
  tv_spend_alphas = c(0.5, 3),
  tv_spend_gammas = c(0.3, 1),
  sem_spend_thetas = c(0.0, 0.3),     # short carryover
  sem_spend_alphas = c(0.5, 3),
  sem_spend_gammas = c(0.3, 1),
  train_size = c(0.5, 0.8)            # time-series validation split, searched
)
```

Theta bounds are where channel knowledge enters. `theta` is the geometric decay per
period, so a half-life of h periods implies `theta = 0.5^(1/h)`: 3.5 weeks → ≈ 0.82,
0.5 weeks → ≈ 0.25. Set bounds around that, wide enough to let the data speak.

Fixing a hyperparameter to a single value (`c(0.8, 0.8)`) is how you impose a hard belief
— the only mechanism available.

---

## 5. Calibration

```r
calibration_input <- data.frame(
  channel = c("tv_spend", "social_spend"),
  liftStartDate = as.Date(c("2025-03-03", "2025-06-02")),
  liftEndDate   = as.Date(c("2025-04-14", "2025-06-30")),
  liftAbs = c(1500, 380),
  spend   = c(60000, 20000),
  confidence = c(0.9, 0.85),
  metric = c("revenue", "revenue"),
  calibration_scope = c("immediate", "immediate")
)
InputCollect <- robyn_inputs(InputCollect = InputCollect,
                             calibration_input = calibration_input)
```

`confidence < 0.8` triggers a low-confidence warning. `calibration_scope` is `"immediate"`
(the channel alone) or `"total"` (a multi-channel test).

This adds MAPE-to-lift as a third objective: Robyn **prefers** models consistent with your
tests rather than **requiring** them. Weaker than a Bayesian likelihood term, but still the
single highest-value input you can give it.

`robyn_calibrate()` is a different function — it fits saturation curves to external
spend–response curve data, for seeding hyperparameter bounds.

---

## 6. Running

```r
OutputModels <- robyn_run(
  InputCollect = InputCollect,
  iterations = 2000, trials = 5,
  ts_validation = TRUE,          # hold out a tail; leave TRUE
  add_penalty_factor = FALSE,
  rssd_zero_penalty = TRUE,
  nevergrad_algo = "TwoPointsDE",
  intercept = TRUE, intercept_sign = "non_negative",
  seed = 123
)

OutputCollect <- robyn_outputs(
  InputCollect, OutputModels,
  pareto_fronts = "auto",
  clusters = TRUE,               # groups similar solutions
  csv_out = "pareto", export = TRUE,
  plot_folder = "./robyn/"
)
```

`intercept_sign = "non_negative"` is the direct guard against a negative baseline — a
better default than fixing the symptom afterwards.

**Convergence.** `robyn_converge()` checks whether the evolutionary search plateaued.
Failing to converge means the front is arbitrary; raise `iterations` and `trials`.

**Choosing a model.** `OutputCollect$allSolutions` lists the Pareto-optimal solutions.
`clusters = TRUE` groups them so you compare *families* rather than noise. Pick by:
one-pager plausibility, decomposition sanity, calibration MAPE, and stability across the
cluster — not by NRMSE alone. Then record which solution ID you chose, because "the
Robyn model" is not a well-defined object without it.

---

## 7. Allocation and refresh

```r
AllocatorCollect <- robyn_allocator(
  InputCollect = InputCollect, OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response",          # or "target_efficiency"
  total_budget = 1000000,
  date_range = "last_12",
  channel_constr_low = 0.7, channel_constr_up = 1.5,
  channel_constr_multiplier = 3,
  constr_mode = "eq",                 # "eq" spends the budget, "ineq" may spend less
  optim_algo = "SLSQP_AUGLAG"
)
```

Robyn's allocator is one of its strongest features: fast, well-visualised, and it reports
the response curve position per channel.

```r
RobynRefresh <- robyn_refresh(
  json_file = "RobynModel.json",
  dt_input = dt_input_new, dt_holidays = dt_prophet_holidays,
  refresh_steps = 13, refresh_iters = 1000, refresh_trials = 3
)
```

`robyn_refresh` re-fits on new data anchored to the previous model, which keeps quarterly
updates stable. `robyn_write` / `robyn_read` / `robyn_recreate` persist a model as JSON —
this is how you make a Robyn result reproducible, and it is worth doing every time.

---

## 8. Robyn vs the Bayesian frameworks

| Concern | Robyn | Bayesian |
|---|---|---|
| Geo panel | **not supported** (aggregate, or one model per geo) | native |
| Priors | bounds + signs | full distributions |
| Uncertainty | Pareto spread + bootstrapped CIs | posterior |
| Calibration | third objective | likelihood term / prior |
| Seasonality | Prophet | Fourier / knots / HSGP |
| Adstock | one type for all channels | per-channel possible |
| Saturation | Hill only | many |
| Reach & frequency | no | Meridian yes |
| Link | additive | identity or log (pymc-marketing) |
| Speed | fast, parallel | slower |
| Language | R | Python |

**When Robyn is the right tool:** national models, R-based teams, fast iteration across
many specifications, and situations where the allocator's output is the deliverable.

**When it is not:** geo hierarchies; genuine uncertainty quantification; reach-and-frequency
questions; mediation or funnel structures; anywhere the DECOMP.RSSD prior would fight the
hypothesis you are testing.

`agent_mmm.backends.get_backend("robyn").translate(spec, priors=...)` generates the R
script and lists what the spec loses in translation.

---

## 9. Common failures

| Symptom | Cause |
|---|---|
| Wildly different models on the Pareto front | Under-converged search, or genuinely unidentified data. Check `robyn_converge()` first |
| Every channel's effect share ≈ its spend share | DECOMP.RSSD dominating. Lower its weight via `objective_weights`, or accept it as a stated prior |
| A channel gets a zero coefficient | Ridge shrank it; check `rssd_zero_penalty` and the sign constraint |
| Negative baseline | Set `intercept_sign = "non_negative"`, and add the missing driver |
| Calibration ignored | `confidence` too low, or dates outside `window_start`/`window_end` |
| Results not reproducible | Seed not fixed, or the model JSON was never written. Use `robyn_write` |
| Hyperparameter names rejected | Use `hyper_names(adstock, all_media)` to generate them |
| Prophet and dummies both included | Double-counted seasonality; drop the hand-built dummies |
