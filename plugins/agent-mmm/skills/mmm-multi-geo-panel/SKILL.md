---
name: mmm-multi-geo-panel
description: |
  Hierarchical multi-geo panel MMM — when geo data helps, how pooling works, data requirements, and the pymc-marketing 1.x and Meridian implementations. Use when the dataset has DMAs, regions or countries, when deciding between a national and a panel model, when handling nationally-bought media in a geo model, or when diagnosing pooling, spillover or rectangularity problems.
---

# Multi-Geo Panel MMM

A geo panel is the single cheapest way to buy statistical power in MMM. Fifty geos over
104 weeks is 5,200 observations against 104 national ones — and, more importantly,
cross-sectional variation in media weight is information that national aggregation
destroys entirely.

---

## 1. When a panel earns its complexity

**Use one when:**
* Media weight genuinely differs across geos (either by targeting, or by how national buys
  land).
* You want to run or have run geo experiments — the panel is the natural home for
  calibrating them.
* National spend is nearly flat but geo-level spend is not. This is common and is the
  strongest argument: a channel that cannot be identified nationally often can be
  identified across geos.
* You need geo-level recommendations.

**Stay national when:**
* Media is bought nationally and lands uniformly. Allocating it to geos by population adds
  no information and invents a variable.
* Geo-level target data is noisy or partially imputed.
* Fewer than about 5 geos — pooling needs enough units to estimate the between-geo
  variance, and with a handful the model is barely different from a national one.

---

## 2. What pooling actually does

Each geo gets its own parameters, drawn from a shared distribution:

```
alpha_g ~ Normal(mu_alpha, sigma_alpha)     for each geo g
```

`sigma_alpha` is estimated, which is the point: the data decides how much geos differ.

* Small `sigma_alpha` → geos are similar → heavy shrinkage towards the shared mean → a
  small geo borrows strength from the rest.
* Large `sigma_alpha` → geos genuinely differ → little shrinkage.

This is strictly better than the two alternatives. Pooling everything (a national model)
assumes geos are identical; fitting each geo separately assumes they share nothing and
gives noisy estimates for small geos. The hierarchy interpolates, and lets the data choose
where.

**Population weighting.** Geo-level targets scale with population, which is a nuisance
dimension, not a media effect. Meridian handles this natively and requires a population
column. In pymc-marketing, model per-capita values or include population as a control.

---

## 3. Data requirements

* **Rectangular.** Every (geo, date) cell must exist. Neither framework will impute.
* **Consistent target definition** across geos. A different sales system in one market is a
  measurement difference the model will read as a media difference.
* **Geo-level spend** for geo-targetable channels. For national buys, allocate by
  population or by a delivery proxy — and record that it is an assumption. A
  population-allocated national buy carries no cross-geo information, so it cannot help
  identify that channel.
* **Population** per geo.
* **Enough geos**: 5 minimum, 20+ comfortable.

```python
from agent_mmm.data_prep import reindex_complete
df = reindex_complete(df, spec)     # inserts missing cells as NaN, then impute per role
```

**Spillover.** Adjacent DMAs share media, commuters and delivery areas. A geo holdout in
one DMA leaks into its neighbours, which attenuates the measured effect. Options: leave
buffer geos out of both arms; use larger, less-connected units; or model spillover
explicitly. At minimum, record the assumption.

---

## 4. pymc-marketing 1.x

In 1.x the multidimensional MMM **is** the MMM — geo is a first-class dimension.

```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

mmm = MMM(
    date_column="date",
    channel_columns=["tv_grps", "sem_spend"],
    target_column="revenue",
    adstock=GeometricAdstock(l_max=12),
    saturation=LogisticSaturation(),
    control_columns=["price_index"],
    yearly_seasonality=6,
    dims=("geo",),          # TUPLE — a bare string iterates into characters
)
mmm.fit(X, y, draws=1000, tune=1500, chains=4)
```

`X` must contain the geo column as a categorical. Every parameter — intercept, channel
betas, adstock, saturation, sigma — becomes geo-indexed with hierarchical priors.
`dims=("geo", "brand")` extends to any number of panel dimensions.

Cost: parameters scale with geos, so sampling is slower and memory larger. Start with a
subset of geos to validate the pipeline before the full run.

## Meridian

Meridian is built for this case.

```python
coord_to_columns = load.CoordToColumns(
    time="date", geo="dma", kpi="revenue", population="population", ...
)
spec = model_spec.ModelSpec(
    media_effects_dist="log_normal",   # geo random effects; or "normal"
    unique_sigma_for_each_geo=False,
    baseline_geo=None,                 # defaults to the largest-population geo
    knots=13,                          # NOT None — see below
    control_population_scaling_id=np.isin(data.control_variable, per_capita_controls),
)
```

Three settings that matter:

* **`knots`.** `knots=None` on a geo model means **one knot per time period** — the most
  flexible baseline available, able to absorb almost anything media might explain. Always
  set it explicitly, or use `enable_aks=True`.
* **`media_effects_dist`.** `"log_normal"` (default) keeps geo effects positive;
  `"normal"` allows negative geo-level effects, which is rarely what you want for media.
* **`baseline_geo`.** The reference for geo dummy encoding. The default (largest
  population) is usually right; a small or atypical geo as reference makes the other
  intercepts harder to interpret.

## Robyn

**No hierarchical geo model.** Either aggregate to national, or fit one model per geo and
lose all pooling. If geo structure matters, Robyn is the wrong tool.

---

## 5. Geo experiments and panels together

This is where the panel pays for itself. A geo holdout produces exactly what a hierarchical
model wants: a channel switched off in known geos over known dates, with comparable control
geos in the same model.

* pymc-marketing: `add_lift_test_measurements` accepts one column per model dimension, so
  the constraint applies to the geos that ran the test.
* Meridian: `roi_calibration_period` scopes the ROI prior to the tested window.

Running geo tests without a panel model throws away most of their value — you measure one
number and cannot generalise it across markets.

---

## 6. Diagnostics specific to panels

| Check | Problem it catches |
|---|---|
| Rows per geo | Non-rectangular panel; a geo with partial history |
| Target CV per geo | A geo with a near-constant or mostly-zero target contributing nothing |
| Spend share per geo | One dominant geo carrying the whole estimate |
| `sigma_alpha` posterior | Near zero means pooling collapsed to national — the panel added nothing |
| Per-geo residuals | A geo the model systematically misses, usually a local factor |
| Geo-level VIF | Collinearity that national aggregation was hiding |

If `sigma_alpha` is essentially zero for every channel, the geos are behaving identically
and the panel is buying precision but not new structure. That is still worthwhile, but say
so rather than claiming geo-level insight the model does not have.

---

## 7. Common failures

| Symptom | Cause |
|---|---|
| Constructor behaves oddly, coords look like single letters | `dims="geo"` instead of `dims=("geo",)` |
| Fit fails on shape | Panel is not rectangular |
| Baseline absorbs everything (Meridian) | `knots=None` on a geo model |
| One geo dominates | No population weighting, or one very large market |
| A nationally-bought channel is unidentified | Population-allocated spend carries no cross-geo variation |
| Sampling is very slow | Parameters scale with geos; reduce geos or channels to validate first |
| Geo effects are implausibly different | `media_effects_dist="normal"` allowing negatives, or too few periods per geo |

---

## Related skills

`mmm-framework-selection` for whether Meridian or pymc-marketing;
`mmm-experimentation-calibration` for geo tests; `mmm-data-engineering` for building a
rectangular panel; `mmm-baseline-and-trend` for the knots decision.
