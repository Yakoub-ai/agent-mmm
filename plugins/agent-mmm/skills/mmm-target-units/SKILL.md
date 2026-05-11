---
name: mmm-target-units
description: |
  Target unit handling for MMM: monetary vs acquisition vs volume targets, CPA vs ROAS framing, and value-per-unit conversions. Use when the target variable is not a currency amount (e.g., policies sold, signups, app installs), when designing the spec.yaml target_unit field, or when a CFO report needs to show both CPA and revenue-equivalent ROAS.
---

# MMM Target Units

## The Core Problem

Standard MMM assumes the target is revenue in a currency. But many businesses:
- Model policies sold (1 policy ≠ £X — policies have different values)
- Model app installs, signups, leads, website sessions
- Model physical units (cars sold, claims filed)

When the target is non-monetary, "ROAS" is meaningless without a value-per-unit conversion. Instead, use **CPA (cost per acquired unit)**.

## Target Unit Classification

```yaml
target_unit:
  kind: "monetary"    # ROAS framing: incremental_revenue / spend
  kind: "acquisition" # CPA framing: spend / incremental_units (policies, signups, etc.)
  kind: "volume"      # CPA framing: spend / incremental_units (installs, sessions, etc.)
```

## Value-Per-Unit Bridge

Add `value_per_unit` to enable both CPA and revenue-equivalent ROAS in CFO reports:

```yaml
target_unit:
  kind: "acquisition"
  label: "policy"
  value_per_unit: 250.0  # average policy value in £
```

With `value_per_unit`, the CFO report shows:
- CPA: £42/policy (direct efficiency metric)
- Revenue-equivalent ROAS: 5.9x (£250 × policies / spend)

## Library Functions

```python
from agent_mmm.target_units import is_monetary, spend_to_return_ratio, roas_label, cpa_label

# Check if monetary
if is_monetary(spec.target_unit):
    label = roas_label(spec.target_unit)  # e.g., "ROAS (£)"
else:
    label = cpa_label(spec.target_unit)   # e.g., "CPA (£/policy)"

# Compute CPA/ROAS for a channel
result = spend_to_return_ratio(spend=10000, incremental_units=238, unit=spec.target_unit)
# result: {"cpa": 42.0, "roas": 5.95, "unit_label": "policy"}
# roas is None if not monetary and no value_per_unit
```

## Examples by Industry

| Industry | Target Column | kind | label | value_per_unit |
|----------|--------------|------|-------|---------------|
| Insurance | `policies_sold` | acquisition | policy | 250.0 (optional) |
| SaaS | `trial_signups` | acquisition | signup | 120.0 (optional LTV) |
| Automotive | `test_drives` | acquisition | test drive | null |
| E-commerce | `revenue` | monetary | £ | null |
| App | `installs` | volume | install | 4.99 (optional) |
| Financial | `mortgage_applications` | acquisition | application | null |

## Priors in Normalized Space

**Important**: All MMM priors operate on the normalized (MaxAbsScaler) target. The scaling is applied equally regardless of target unit. `sample_posterior_predictive()` returns [0,1] normalized values — multiply by `model.get_scales_as_xarray()["target_scale"]` to recover original scale, which is in the original target unit (policies, signups, etc.).

ROAS and CPA calculations must happen in original scale, not normalized scale.
