---
name: mmm-multi-geo-panel
description: |
  Multi-geo panel MMM using pymc-marketing's multidimensional API. Use when the dataset has multiple geographies, DMAs, countries, or regions that should be modeled together. Covers geo column setup, panel data validation, and the multidimensional MMM constructor.
---

# Multi-Geo Panel MMM

## When to Use

Use multi-geo panel modeling when:
- You have weekly marketing data for multiple regions, countries, or DMAs
- Channels are bought nationally but results vary by geo
- You want partial-pooling of effects across geos (shrink toward national mean)

**Do not use** for daily data or monthly data in v1 (weekly only).

## Data Format

Panel data must be in long format:

```
date        | geo    | spend_sem | spend_tv | target
2024-01-01  | UK     | 10000     | 50000    | 1200
2024-01-01  | DE     | 8000      | 40000    | 900
2024-01-08  | UK     | 11000     | 52000    | 1250
...
```

- One row per (date × geo) combination
- Date column: weekly periods
- Geo column: string identifier (country code, DMA code, region name)
- All numeric columns in original units (not pre-normalized)

## Spec Configuration

```yaml
geo_column: "geo"  # name of the geo column; null for single-geo
```

When `geo_column` is set, the framework:
1. Validates panel completeness (every geo should have data for every week)
2. Checks for geo-specific missing patterns (some geos inactive for certain channels)
3. Applies VIF checks within each geo before pooling

## Multi-Geo MMM Constructor

```python
from pymc_marketing.mmm.multidimensional import MMM

# For panel data, pass the geo column — the multidimensional MMM handles pooling
model = MMM(
    date_column="date",
    channel_columns=channel_cols,
    target_column="y",
    adstock=GeometricAdstock(l_max=5),
    saturation=LogisticSaturation(),
    control_columns=control_cols,
    yearly_seasonality=8,
    model_config=model_config,
    # geo_column handled via X DataFrame structure — include geo as a coordinate
)
```

**Note**: The multidimensional API infers geo dimensions from the DataFrame coordinates. Ensure the DataFrame index includes (date, geo) if multi-geo.

## Panel Validation Checks

The data audit (`run_audit`) performs these extra checks for multi-geo data:

1. **Completeness**: All geos present for all dates? Missing (date, geo) combinations → FAIL
2. **Geo count**: At least 3 geos recommended for meaningful partial-pooling
3. **Balanced panel**: Geos should have similar number of active observations
4. **Geo-level VIF**: Collinearity checked per-geo, not just overall

## Prior Specification for Multi-Geo

Priors in `model_config` apply globally. If channel effectiveness varies substantially by geo (e.g., TV works in Germany but not UK), consider:
1. Running separate single-geo models (simpler, more interpretable)
2. Widening the prior for the affected channel to let the data determine geo-specific effects

## Data Audit Multi-Geo Output

`audit.json` includes a `multi_geo` section:
```json
{
  "multi_geo": {
    "geo_count": 4,
    "geos": ["UK", "DE", "FR", "NL"],
    "panel_complete": true,
    "missing_cells": 0,
    "per_geo_row_counts": {"UK": 104, "DE": 104, "FR": 104, "NL": 104}
  }
}
```
