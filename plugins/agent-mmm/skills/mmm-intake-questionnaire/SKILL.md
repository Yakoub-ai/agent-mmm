---
name: mmm-intake-questionnaire
description: |
  Intake questionnaire for MMM projects. Use when starting a new MMM project, resuming an incomplete intake, or updating the spec.yaml. Covers company context, target unit, channel inventory, controls, seasonality, and greenfield vs brownfield classification.
---

# MMM Intake Questionnaire

## Purpose

The intake questionnaire captures everything needed to build a well-specified MMM. It produces `./mmm-workspace/spec.yaml` — the single source of truth for the entire framework.

## Two Modes

### `/mmm-intake-quick` (5 questions, ~2 minutes)
For smoke tests and rapid prototyping. Collects:
1. Company name + industry
2. Data file path
3. Channel columns (comma-separated)
4. Target column and unit description
5. Greenfield or brownfield?

Outputs a minimal `spec.yaml` that gets you to `/mmm-analyze-data`.

### `/mmm-intake` (full intake, ~10 minutes)
For production MMM work. Covers all 25 questions across 5 sections:

**Section 1 — Company & Context**
- Company name, industry vertical, primary region/country
- Business model (D2C, B2B, insurance, automotive, SaaS, retail, etc.)
- Prior MMM experience (greenfield vs brownfield)
- Stakeholders who will consume the results

**Section 2 — Data**
- Path to data file (CSV or Parquet)
- Date column name, date format, granularity (weekly required for v1)
- Number of rows and date range
- Any known gaps, outliers, or data quality issues

**Section 3 — Target Variable**
- What are you modeling? (sales, revenue, policies, signups, app installs, website sessions)
- Unit of measurement: is 1 unit = 1 dollar/SEK/EUR, or 1 unit = 1 policy/signup?
- Currency code if monetary (USD, GBP, SEK, EUR, etc.)
- Value per unit conversion if non-monetary (optional: enables ROAS in CFO reports)

**Section 4 — Marketing Channels**
- List all spend columns available
- Confirm which channels to include (exclude internally correlated channels)
- Are any channels always co-occurring? (combine recommendation)
- Multi-geo panel? (geo column name if yes)

**Section 5 — Controls & Seasonality**
- Any known seasonality patterns? (Q4 peak, summer dip, etc.)
- Holidays relevant to your market?
- Any promotions, pricing events, or competitor activity to control for?
- For brownfield: path to existing `idata.nc` and/or existing `model_config.json`

## Spec.yaml Schema

```yaml
version: "1"
mmm_type: "greenfield"  # or "brownfield"
company_name: "Acme Corp"
industry: "insurance"
region: "UK"
data_path: "./data/weekly_marketing.csv"
date_column: "date"
target_column: "policies_sold"
target_unit:
  kind: "acquisition"  # "monetary" | "acquisition" | "volume"
  label: "policy"
  currency_code: null  # required if kind=monetary
  value_per_unit: 250.0  # optional: enables ROAS for non-monetary targets
channels:
  - column: "spend_sem"
    channel_type: "sem"
    is_active: true
  - column: "spend_tv"
    channel_type: "tv"
    is_active: true
controls:
  - column: "is_q4"
    control_type: "seasonal"
    is_active: true
yearly_seasonality: 8
adstock_first: true
geo_column: null  # set to column name for multi-geo panel
brownfield:  # only present if mmm_type=brownfield
  existing_idata_path: "./previous_model/idata.nc"
  existing_model_config_path: "./previous_model/model_config.json"
```

## Resume Support

If `./mmm-workspace/spec.yaml` already exists, `/mmm-intake` reads it and asks only about missing or outdated fields. The `version` field tracks schema changes.
