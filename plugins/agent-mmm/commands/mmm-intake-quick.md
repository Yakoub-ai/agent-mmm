---
description: Quick 5-question MMM project intake. Creates ./mmm-workspace/spec.yaml with sensible defaults. Run /mmm-intake for the full 25-question version.
---

# MMM Quick Intake

You are running the quick MMM project intake. This asks 5 essential questions and creates `./mmm-workspace/spec.yaml` with sensible defaults. Run `/mmm-intake` for the full production version.

## Step 1: Check for existing spec

Read `./mmm-workspace/spec.yaml` if it exists. If it does, warn the user:
> ⚠️ A spec.yaml already exists. Continuing will overwrite it. The existing spec will be shown below.

Show the first 30 lines of the existing spec. Then proceed regardless (the user invoked the command).

## Step 2: Create workspace directory

Run: `mkdir -p ./mmm-workspace`

## Step 3: Ask 5 questions

Ask the user the following questions **one at a time** (do not batch them):

**Q1**: "What is your company name?" — record as `company_name`

**Q2**: "What is your industry? (e.g. insurance, retail, SaaS, automotive, FMCG, telecoms, finance, other)" — record as `industry`

**Q3**: "What is the primary market/region? (e.g. Sweden, UK, US, Germany, global)" — record as `region`

**Q4**: "What is the full path to your marketing data file? (CSV or Parquet, weekly aggregated, must contain a date column and a target/sales column)" — record as `data_path`

**Q5**: "What does '1 sale' mean in your data? For example: '1 policy sold', '1 new subscriber', '$1 in revenue', '1 SEK revenue', '1 unit sold'. This determines how we report marketing efficiency." — From the answer, infer:
  - `target_unit.kind`: if the answer mentions '$', '€', '£', 'SEK', 'EUR', 'USD', 'GBP', 'revenue', 'dollars', 'kronor', 'euros' → `monetary`; if it mentions 'policy', 'subscription', 'signup', 'lead', 'install', 'registration', 'application' → `acquisition`; otherwise → `volume`
  - `target_unit.label`: the key noun (e.g. 'policy', 'SEK', 'signup', 'unit')
  - `target_unit.currency_code`: if monetary, try to infer ISO 4217 code (SEK, USD, EUR, GBP); otherwise null

## Step 4: Load and inspect the data file

Run the following Python snippet, substituting the actual `data_path` from Q4 for `PATH` and choosing `.read_csv` or `.read_parquet` based on the file extension:

```python
import pandas as pd
path = 'PATH'
df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
print('columns:', list(df.columns))
print('rows:', len(df))
print('sample date range:', df.iloc[0, 0], 'to', df.iloc[-1, 0])
```

If the file doesn't exist or errors, note it and continue with `data_path` as provided — the audit step will catch issues later.

From the column list, identify:
- Columns containing 'spend', 'cost', 'budget', 'investment', 'media' → probable channel columns
- A column that looks like a date (date, week, period, ds, time) → `date_column` (default: first date-like column, fallback: `'date'`)
- A column that matches 'sales', 'revenue', 'conversions', 'orders', 'y', 'target', 'kpi' → `target_column` (default: `'y'`)

## Step 5: Write spec.yaml

Write `./mmm-workspace/spec.yaml` with this exact structure, filling in all discovered values. Use `null` (unquoted) for absent optional values and `""` for empty strings.

```yaml
version: "1"
mmm_type: greenfield
company_name: <Q1 answer>
industry: <Q2 answer>
region: <Q3 answer>
data_path: <Q4 answer>
date_column: <detected or 'date'>
target_column: <detected or 'y'>
target_unit:
  kind: <inferred from Q5>
  label: <inferred label>
  currency_code: <if monetary, else null>
  value_per_unit: null
channels:
  - column: <channel_column_1>
    label: ""
    channel_type: null
    is_active: true
  # ... one entry per detected channel column
controls: []
granularity: weekly
seasonality:
  yearly_fourier_modes: 8
  explicit_holiday_column: null
  expected_peaks: []
geo:
  is_panel: false
  geo_column: null
  geos: []
brownfield: null
notes: "Created by /mmm-intake-quick"
```

If no channel columns were detected, write `channels: []`.

## Step 6: Confirm

Tell the user:

> ✅ spec.yaml written to `./mmm-workspace/spec.yaml`
>
> **Detected**: X channels, date column: `<col>`, target: `<col>` (`<label>`)
>
> **Next steps:**
> - Run `/mmm-analyze-data` to audit your dataset
> - Run `/mmm-recommend-controls` to get external-factor suggestions
> - Run `/mmm-recommend-priors` to set up Bayesian priors
> - Or run `/mmm-intake` to fill in additional details (seasonality, geo, brownfield context)
