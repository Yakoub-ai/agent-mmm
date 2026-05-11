---
name: mmm-stakeholder-reporting
description: |
  Stakeholder-specific MMM reporting templates and content guidance. Use when generating or reviewing CMO, CFO, Marketing Ops, or Data Science reports, or when explaining what each report should contain and how to frame results for each audience.
---

# MMM Stakeholder Reporting

## Report Philosophy

Each audience needs different content at different abstraction levels. The same underlying model produces four different documents.

**Rule**: Never report posterior means alone. Always show credible intervals (80% or 90% HDI). Uncertainty is a feature, not a bug — it prevents overconfident budget decisions.

## CMO Report

**Audience**: Chief Marketing Officer, VP Marketing  
**Goal**: Answer "Is our marketing working and where should we focus?"  
**Tone**: Plain English, minimal statistics  

**Content**:
1. Executive summary (3 sentences: what worked, what didn't, top recommendation)
2. Channel contribution table (% of incremental target attributable to each channel)
3. Top 3 channels to increase investment (near saturation ceiling? efficiency above median?)
4. Bottom 3 channels to review (over-saturated? low ROAS/CPA?)
5. Confidence note (high/medium/low based on convergence + CV R²)

**Key metric framing**:
- Monetary target: "Channel X drove £Ym in incremental revenue (80% CI: £Xm–£Zm)"
- Acquisition target: "Channel X drove N incremental policies (80% CI: N1–N2)"

## CFO Report

**Audience**: Chief Financial Officer, Finance Director  
**Goal**: Answer "What's the financial return on our marketing investment?"  
**Tone**: Numbers-first, uncertainty quantified  

**Content**:
1. Total spend vs total incremental return (with 80% credible interval)
2. Per-channel CPA or ROAS table (varies by `target_unit.kind`)
3. Efficiency ranking (best to worst ROAS/CPA)
4. Sensitivity: "If we cut Channel X by 20%, incremental units would fall by ~N"

**Target-unit-aware framing**:
```
Monetary target (kind=monetary):
  → ROAS = incremental_revenue / spend
  → "Channel X: ROAS 4.2x (80% CI: 3.1x–5.6x)"

Acquisition target (kind=acquisition), no value_per_unit:
  → CPA only = spend / incremental_units
  → "Channel X: CPA £42/policy (80% CI: £31–£58)"

Acquisition target WITH value_per_unit=250:
  → CPA + revenue-equivalent ROAS
  → "Channel X: CPA £42/policy, implying ROAS 5.9x at £250/policy value"
```

## Marketing Ops Report

**Audience**: Marketing Operations Manager, Channel Leads  
**Goal**: Answer "Where exactly are we over/under-spending and by how much?"  
**Tone**: Actionable, channel-by-channel  

**Content**:
1. Per-channel saturation status (below/near/above saturation point)
2. Current spend vs estimated optimal spend per channel
3. Estimated units gained from reallocation
4. Response curve inflection points (where is the knee of the curve?)
5. Adstock profiles (how long does each channel's effect last?)

**Saturation interpretation**:
- Below 30% saturation → significant headroom, efficient to increase spend
- 30–70% saturation → approaching diminishing returns, moderate efficiency
- Above 70% saturation → strong diminishing returns, consider reallocation

## Data Science Report

**Audience**: Data Scientists, Analytics Engineers, Model Reviewers  
**Goal**: Answer "Is this model trustworthy and reproducible?"  
**Tone**: Technical, complete, no sugarcoating  

**Content**:
1. Model specification (adstock type + l_max, saturation type, Fourier modes, controls list)
2. Sampler configuration (draws, tune, chains, target_accept)
3. Convergence summary (rhat max, ESS min, divergences)
4. Fit metrics (in-sample R², MAPE, CV R², overfit gap)
5. Prior audit (were any parameters pulled to prior bounds? indicates too-tight priors)
6. Attribution plausibility flags (any channel > 60% contribution? any negative ROAS?)
7. Reproducibility block (run ID, timestamp, pymc-marketing version, random seed if set)

## Library Integration

```python
from agent_mmm.reports.cmo import generate_cmo_report
from agent_mmm.reports.cfo import generate_cfo_report
from agent_mmm.reports.mops import generate_mops_report
from agent_mmm.reports.ds import generate_ds_report

# All return Markdown string and write to mmm-workspace/reports/<persona>.md
report_md = generate_cmo_report(spec, run_id, metrics, diagnostics, channel_contributions, base=".")
```

Run via slash command: `/mmm-report` (generates all four reports)
