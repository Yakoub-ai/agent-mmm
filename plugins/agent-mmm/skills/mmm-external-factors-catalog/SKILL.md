---
name: mmm-external-factors-catalog
description: |
  Catalog of external factors and control columns for MMM. Use when recommending controls for a new model, evaluating which external factors to include, or understanding why certain controls matter for a given industry or region.
---

# External Factors & Controls Catalog

## Control Categories

### Calendar Controls (Universal)
| Control | Type | How to Create |
|---------|------|--------------|
| `is_q4` | Seasonal | `date.dt.quarter == 4` |
| `is_q1` | Seasonal | `date.dt.quarter == 1` |
| `week_of_year` | Seasonality | `date.dt.isocalendar().week` (use Fourier instead if 52+ weeks) |
| `is_holiday` | Event | `holidays` library for the target country |
| `days_in_month` | Seasonal | `date.dt.days_in_month` — useful for monthly-adjacent patterns |
| `black_friday_week` | Event | Flag the week containing the last Friday of November |
| `christmas_week` | Event | Flag Dec 22–31 |

Use `yearly_seasonality=8` (8 Fourier modes) for weekly data — these capture smooth annual patterns. Add explicit indicator dummies for sharp drops/spikes the Fourier terms miss.

### Macro Controls
| Control | When to Include | Source |
|---------|----------------|--------|
| Consumer confidence index | Insurance, automotive, high-ticket retail | National statistics office |
| Interest rates | Insurance, mortgages, automotive | Central bank data |
| Unemployment rate | Any consumer industry | Statistics office |
| Fuel/energy prices | Automotive, logistics | Energy agency data |
| Exchange rates | International campaigns | Central bank or FX API |
| GDP growth | B2B, enterprise | Statistics office |

### Search Interest (Google Trends)
| Control | When to Include |
|---------|----------------|
| Brand search volume | Always — controls for organic demand that isn't media-driven |
| Category search volume | When modeling SEM clicks (treat as proxy for addressable demand) |
| Competitor brand search | When competitive dynamics are visible in data |

**Key rule**: When modeling SEM clicks (Series C), include search volume as a control, not a channel. SEM responds to search demand — including SEM spend when modeling search volume would be circular.

### Industry-Specific Controls

**Insurance**
- `new_reg_plates` — UK: March and September new registrations spike auto insurance demand
- `annual_renewal_period` — month when policies renew (insurer-specific)
- `flood_event` / `storm_event` — severe weather events spike home insurance quotes
- `comparison_site_promotion` — aggregator promotions drive channel mix shifts

**Automotive**
- `model_launch` — new model release weeks
- `plate_change_period` — UK Mar/Sep, German Jan, French Jan-Feb
- `fuel_price_spike` — correlated with EV/hybrid consideration
- `scrappage_scheme` — government incentive windows

**Retail / D2C**
- `prime_day` — mid-July (Amazon), affects cross-retailer competition
- `back_to_school` — Aug–Sep
- `valentine_week`, `mothers_day_week`, `fathers_day_week` — gifting spikes

**SaaS / Tech**
- `fiscal_year_end` — enterprise buying spikes at customer FYE
- `product_launch_week` — own product releases
- `competitor_outage` — if visible in data (unusual acquisition spikes)

## Controls Engine

Use `agent_mmm.controls_engine.recommend_controls(spec, audit_findings, base)` to auto-generate control recommendations. The engine reads `references/controls_catalog.yaml` and filters by:
1. Industry from `spec.industry`
2. Region from `spec.region`
3. Detected structural breaks from audit findings (adds break-point dummies automatically)
4. Seasonal strength from STL decomposition (adds Fourier recommendation if seasonal_strength > 0.4)

Run via slash command: `/mmm-recommend-controls`

## Adding Custom Controls

After running `/mmm-recommend-controls`, edit `mmm-workspace/controls/recommended.json` to add/remove controls, then re-run `/mmm-intake` to write accepted controls back to `spec.yaml`.
