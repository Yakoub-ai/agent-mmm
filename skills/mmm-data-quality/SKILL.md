---
name: mmm-data-quality
description: |
  Data quality assessment and preparation for Marketing Mix Models. Use when evaluating datasets for MMM readiness, checking data requirements, validating columns, assessing collinearity, handling missing values, or preparing marketing spend data for pymc-marketing modeling. Also activate when the user asks about minimum data requirements, data granularity, or feature engineering for MMM.
---

# MMM Data Quality & Preparation

## Minimum Data Requirements

| Requirement | Threshold | Rationale |
|-------------|-----------|-----------|
| Minimum observations | 52 weeks | One full year captures seasonality |
| Recommended observations | 104+ weeks | Enables meaningful cross-validation (4+ folds) |
| Minimum channels | 1 | But 3+ recommended for meaningful attribution |
| Data frequency | Weekly preferred | Daily is noisy, monthly loses signal |

## Data Validation Checklist

### 1. Completeness
- Check for missing values in target, channels, and date columns
- Drop rows with NaN in critical columns (target, date)
- For channel spend: missing = 0 (no spend that week) vs truly missing data -- handle differently
- Verify no gaps in date sequence (missing weeks)

### 2. Target Variable
- Must be continuous and positive (sales, revenue, leads)
- Check for zeros -- are they real or data issues?
- Look for extreme outliers (>3 SD) -- consider Student-t likelihood
- Verify target has sufficient variation (CV > 0.10)

### 3. Channel Spend Columns
- Must be non-negative (spend can't be negative)
- Check for sufficient variation per channel -- near-zero channels can't be modeled
- Naming convention: prefix with `spend_` for clarity (e.g., `spend_SEM`, `spend_TV`)
- Verify spend reflects actual media investment, not impressions or clicks

### 4. Collinearity Assessment
```python
import pandas as pd

# Correlation matrix for channel spend
corr = X[channel_columns].corr()
high_corr = corr[corr.abs() > 0.7]  # Flag pairs > 0.7

# VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["feature"] = channel_columns
vif["VIF"] = [variance_inflation_factor(X[channel_columns].values, i)
              for i in range(len(channel_columns))]
# VIF > 5: moderate collinearity, > 10: severe -- consider combining channels
```

**If collinearity is high:**
- Combine correlated channels (e.g., Facebook + Instagram -> Meta)
- Drop the lower-spend channel
- Use informative priors to regularize

### 5. Seasonality & Controls
- Derive quarterly indicators from date: `is_q1`, `is_q2`, `is_q3`, `is_q4`
- Flag holidays and special periods (Christmas dip, Black Friday, etc.)
- Include macroeconomic controls if available (interest rate, CPI, unemployment)
- Controls are NOT scaled by MaxAbsScaler -- they pass through raw

### 6. Temporal Patterns
- Check for structural breaks (COVID, major product launches)
- Look for trend (increasing/decreasing baseline) -- may need time-varying intercept
- Verify data covers sufficient seasonal cycles (at least 1 full year)

## Feature Engineering

### Quarterly Indicators
```python
df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
df["is_q1"] = (df["quarter"] == 1).astype(int)
df["is_q4"] = (df["quarter"] == 4).astype(int)
```

### Holiday/Special Period Flags
```python
# Christmas dip: weeks containing Dec 20-31 or Jan 1-7
date = pd.to_datetime(df["date"])
df["is_christmas_dip"] = (
    ((date.dt.month == 12) & (date.dt.day >= 20)) |
    ((date.dt.month == 1) & (date.dt.day <= 7))
).astype(int)
```

### Channel Name Extraction
```python
# Strip spend_ prefix for display
channel_names = [col.replace("spend_", "") for col in channel_columns]
```

## Data Split for Cross-Validation

pymc-marketing's `TimeSliceCrossValidator` uses expanding-window CV:
- `n_init`: minimum training observations (typically 80 weeks)
- `forecast_horizon`: test window size (typically 13 weeks / 1 quarter)
- `step_size`: step between folds (typically = forecast_horizon)

```
Fold 1: train[0:80]     test[80:93]
Fold 2: train[0:93]     test[93:106]
Fold 3: train[0:106]    test[106:119]
Fold 4: train[0:119]    test[119:132]
```

Need at least `n_init + n_folds * forecast_horizon` observations for meaningful CV.

## Red Flags

| Issue | Impact | Fix |
|-------|--------|-----|
| < 52 weeks of data | Can't capture full seasonality | Collect more data or simplify model |
| Channel with 80%+ zeros | Model can't learn its effect | Drop or combine with related channel |
| VIF > 10 between channels | Attribution is unreliable | Combine correlated channels |
| Sudden structural break | Model fit degrades | Split pre/post or add indicator variable |
| Target is count data | Violates continuous assumption | Use log transform or different likelihood |
