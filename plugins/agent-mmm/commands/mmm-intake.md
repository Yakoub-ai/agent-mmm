---
description: Full MMM project intake questionnaire (~25 questions). Creates or updates ./mmm-workspace/spec.yaml. Covers company, target, channels, seasonality, controls, greenfield/brownfield, and multi-geo. Run /mmm-intake-quick for the 5-question fast version.
---

# MMM Full Intake Questionnaire

You are running the full MMM project intake. This covers all aspects of the project context needed to configure a high-quality Marketing Mix Model. Ask questions **one section at a time**. After each section, summarize what was recorded before moving to the next.

## Pre-flight

1. Check if `./mmm-workspace/spec.yaml` exists. If so:
   - Load it and show a summary of what's already filled in.
   - Tell the user: "I'll update the existing spec with your new answers. Press Enter to continue or type 'start fresh' to overwrite."
   - If they say 'start fresh', proceed from scratch; otherwise merge new answers into the existing spec.

2. Create workspace: `mkdir -p ./mmm-workspace`

---

## Section A: Company & Business Context

Ask these questions one at a time:

**A1**: "What is your company name?"

**A2**: "What industry are you in? (Examples: insurance, retail, SaaS, automotive, FMCG/CPG, telecoms, financial services, utilities, healthcare, other)"

**A3**: "What is your primary market or region? (e.g. Sweden, Scandinavia, UK, US, Germany, global)"

**A4**: "Is this model for the entire business, a specific brand, a product line, or a market segment? Describe briefly."
→ Record as `scope` in the spec.

**A5 (context only — do not include verbatim in spec.yaml, but use for recommendations)**: "Does the business have strong seasonality? If yes, describe the pattern. (Examples: Q4 holiday spike, summer peak, school year cycle, no strong seasonality)"
→ Extract named peaks (e.g. 'Christmas', 'Black Friday', 'Summer') and record them in `seasonality.expected_peaks`.

After Section A, summarize:
> ✓ Recorded: company=[name], industry=[industry], region=[region], scope=[scope]

---

## Section B: Target Variable & Sales

**B1**: "What is the full path to your marketing data file? (CSV or Parquet)"

After receiving the path, try to read column names and row count by running:

```python
import pandas as pd, sys
path = 'PATH'
df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
print('Columns:', list(df.columns))
print('Rows:', len(df))
print('First date:', df.iloc[0, 0], '| Last date:', df.iloc[-1, 0])
```

Show the column list to the user. If the file cannot be read, continue anyway — the data audit will surface the error.

**B2**: "Which column is your target / dependent variable (what the model will predict)? [Show detected options from column list]"
→ `target_column`

**B3**: "What does 1 unit of your target represent? For example: '1 policy sold', '$1 in revenue', '1 SEK in revenue', '1 app install', '1 qualified lead'. Be specific — this affects how we calculate ROI."
→ Infer:
  - `target_unit.kind`: mentions '$', '€', '£', 'SEK', 'EUR', 'USD', 'GBP', 'revenue', 'dollars', 'kronor', 'euros' → `monetary`; mentions 'policy', 'subscription', 'signup', 'lead', 'install', 'registration', 'application' → `acquisition`; otherwise → `volume`
  - `target_unit.label`: the key noun (e.g. 'policy', 'SEK', 'signup', 'unit')
  - `target_unit.currency_code`: if monetary, infer ISO 4217 code (SEK, USD, EUR, GBP); otherwise null

**B4 (ask only if target_unit.kind is 'acquisition' or 'volume')**: "Optional: What is the approximate monetary value of 1 [label]? (e.g. 'average policy value is 2,500 SEK', 'average order value is $85'). Enter a number or press Enter to skip."
→ `target_unit.value_per_unit` — extract the numeric value; set to null if skipped.

**B5**: "What is your date column name? [show detected date-like columns]"
→ `date_column`

After Section B, summarize detected data properties (rows, date range, target column, target unit).

---

## Section C: Marketing Channels

Show the column list from B1 again, highlighting columns that contain 'spend', 'cost', 'budget', 'investment', or 'media' as likely channel candidates.

**C1**: "Which columns are your marketing spend / investment channels? You can list column names or say 'all spend columns'. I've highlighted likely candidates above."
→ Build the `channels` list from the confirmed columns.

**C2**: "For each channel, what's a short human-readable label? (e.g. 'spend_sem' → 'Paid Search / SEM', 'spend_fb' → 'Meta / Facebook'). List them briefly or say 'use column names' to keep column names as labels."
→ Set `channels[i].label` for each channel.

**C3**: "Are any of these channels correlated with each other or always activated together? (This affects model collinearity — e.g. Facebook + Instagram always run together.)"
→ Append a note to the `notes` field; the data audit will flag VIF automatically.

**C4**: "Any channels you want to EXCLUDE from this model run? (e.g. a channel that only started 2 months ago and has too few observations)"
→ Set `is_active: false` for excluded channels.

**C5**: "Does your data have multiple geographies (e.g. countries, regions) as separate rows? If yes, which column identifies the geography?"
→ Set `geo.is_panel` (true/false), `geo.geo_column` (column name or null), `geo.geos` (list of unique geo values if readable from data, otherwise []).

After Section C, summarize the confirmed channel list and geo setup.

---

## Section D: Seasonality & Events

**D1**: "Do you have explicit indicator columns for holidays, promotions, or events already in your data? If yes, list the column names. If no, say 'none'."
→ Set `seasonality.explicit_holiday_column` to the column name if exactly one is given; if multiple, record the first and add a note.

**D2**: "What are the biggest seasonal peaks or dips in your business? (e.g. 'Black Friday / Cyber Monday', 'Christmas', 'Summer vacation', 'January slowdown', 'Back to school')"
→ Add to `seasonality.expected_peaks` (merge with any peaks captured in A5).

**D3**: "Were there any major structural events that affected your business during the data period? (e.g. COVID lockdowns, a major competitor entering the market, a product relaunch, a large brand campaign unlike others)"
→ Append a descriptive note to the `notes` field. The controls engine will suggest dummy variables for these.

After Section D, summarize seasonality configuration.

---

## Section E: Greenfield vs Brownfield

**E1**: "Is this a brand-new MMM (you haven't built one before for this dataset) or are you improving / replacing an existing model?"
→ If new: `mmm_type: greenfield`, set `brownfield: null`.
→ If improving: `mmm_type: brownfield`, proceed to E2–E3.

If brownfield:

**E2**: "Do you have an existing fitted model artifact you'd like us to build on? (An InferenceData `.nc` file from a previous pymc-marketing run, or a model_config dict.) Enter the file path or say 'no'."
→ `brownfield.idata_path` (null if 'no')

**E3**: "Any known issues with the previous model? (e.g. 'ROAS for TV seems too high', 'model was overfitting', 'priors were too tight', 'attribution for digital was implausible')"
→ `brownfield.notes`

---

## Section F: Review & Write

Show the user a preview of the complete spec that will be written:

```
Here is your MMM spec summary:

  Company:   [company_name] | Industry: [industry] | Region: [region]
  Scope:     [scope]
  Data:      [data_path] ([rows] rows, [date_range])
  Target:    [target_column] (1 [label] = ...)
  Channels:  [channel count] — [label list]
  Controls:  [controls count] (to be populated by /mmm-recommend-controls)
  Seasonality peaks: [expected_peaks list]
  Geo:       Single market / Panel ([geo_col])
  Type:      Greenfield / Brownfield
```

Ask: "Does this look correct? Type 'yes' to save, or describe any corrections."

Apply any corrections the user describes, then write `./mmm-workspace/spec.yaml` with the structure below:

```yaml
version: "1"
mmm_type: <greenfield or brownfield>
company_name: <A1>
industry: <A2>
region: <A3>
scope: <A4>
data_path: <B1>
date_column: <B5>
target_column: <B2>
target_unit:
  kind: <inferred from B3>
  label: <inferred from B3>
  currency_code: <if monetary, else null>
  value_per_unit: <B4 or null>
channels:
  - column: <column_name>
    label: <human label>
    channel_type: null
    is_active: true
  # one entry per confirmed channel
controls: []
granularity: weekly
seasonality:
  yearly_fourier_modes: 8
  explicit_holiday_column: <D1 or null>
  expected_peaks: <list from A5 + D2>
geo:
  is_panel: <C5>
  geo_column: <C5 or null>
  geos: <list or []>
brownfield: <null, or object with idata_path and notes>
notes: "<any structural event notes from D3 and collinearity notes from C3>"
```

After writing:

> ✅ Full spec written to `./mmm-workspace/spec.yaml`
>
> **Recommended next steps in order:**
> 1. `/mmm-analyze-data` — automated data quality audit
> 2. `/mmm-recommend-controls` — external factors & control variable suggestions
> 3. `/mmm-recommend-priors` — Bayesian prior recommendations per channel
> 4. `/mmm-build` — generate the model configuration
> 5. `/mmm-fit` — run prior predictive check and fit the model
> 6. `/mmm-diagnose` — validate convergence, overfit, and attribution plausibility
> 7. `/mmm-improve` — run the iterative improvement tournament
> 8. `/mmm-report cmo|cfo|mops|ds` — generate stakeholder reports
