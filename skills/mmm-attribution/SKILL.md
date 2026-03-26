---
name: mmm-attribution
description: |
  Channel attribution, ROAS calculation, contribution decomposition, and results interpretation for Marketing Mix Models. Use when extracting channel contributions, calculating return on ad spend, generating response/saturation curves, interpreting model outputs, assessing channel effectiveness, decomposing the target variable into components, or presenting MMM results to stakeholders. Also activate when the user asks about contribution shares, waterfall charts, or which channels are over/under-invested.
---

# MMM Attribution & Results Interpretation

## Extracting Channel Contributions

### Original Scale Contributions
```python
# Step 1: Add contribution variable to posterior
model.add_original_scale_contribution_variable(["channel_contribution"])

# Step 2: Access from idata
contrib = model.idata.posterior["channel_contribution_original_scale"]
# dims: (chain, draw, date, channel)

# Mean contributions per channel over time
contrib_mean = contrib.mean(dim=("chain", "draw"))  # (date, channel)

# Total contribution per channel (sum across all dates)
total_contrib = contrib.mean(dim=("chain", "draw")).sum("date")

# Contribution shares (proportion of each channel)
shares = contrib / contrib.sum("channel")
shares_mean = shares.mean(dim=("chain", "draw", "date"))
```

### NEVER Use Legacy Method
```python
# WRONG (legacy, doesn't exist in multidimensional):
# model.compute_channel_contribution_original_scale()

# WRONG (legacy):
# model.get_channel_contribution_share_samples()
```

### Using MMMSummaryFactory
```python
from pymc_marketing.mmm.summary import MMMSummaryFactory, MMMIDataWrapper

wrapper = MMMIDataWrapper(model.idata)
factory = MMMSummaryFactory(data=wrapper, model=model)

contributions = factory.contributions()     # DataFrame with HDI
roas = factory.roas()                       # ROAS per channel
saturation_df = factory.saturation_curves() # Saturation curve data
adstock_df = factory.adstock_curves()       # Adstock curve data
total_contrib = factory.total_contribution()
spend = factory.channel_spend()
```

## ROAS Calculation

### Basic ROAS
```python
# ROAS = Total Contribution / Total Spend
for ch in channels:
    ch_name = ch.replace("spend_", "")
    contribution = contrib_mean.sel(channel=ch_name).sum().values
    spend = X[ch].sum()
    roas = contribution / spend
    print(f"{ch_name}: ROAS = {roas:.2f}")
```

### ROAS with Uncertainty
```python
# Per-sample ROAS for credible intervals
roas_samples = contrib.sum("date") / X[channels].sum().values  # (chain, draw, channel)
roas_mean = roas_samples.mean(dim=("chain", "draw"))
roas_hdi = az.hdi(roas_samples, hdi_prob=0.89)
```

### Interpreting ROAS
| ROAS | Interpretation |
|------|---------------|
| > 3.0 | Strong performer -- consider increasing spend |
| 1.0 - 3.0 | Moderate -- maintaining value but room for optimization |
| < 1.0 | Underperforming -- spend exceeds attributed return |
| Very high (>10) | Suspicious -- check for low-spend channels with noise |

**Caveat:** ROAS from MMM is incremental (marginal) ROAS, not total ROAS. It measures the additional effect of each unit of spend, accounting for diminishing returns.

## Response Curves (Saturation Analysis)

### Generating Saturation Curves
```python
# Sample saturation curves with posterior uncertainty
sat_curve = model.sample_saturation_curve(max_value=2.0, num_points=100)
# Returns xarray with dims (chain, draw, x, channel)

# Plot
model.plot.saturation_curves(curve=sat_curve)
model.plot.saturation_curves_scatter()  # With actual data points
model.plot.saturation_scatterplot()
```

### Interpreting Saturation Curves
- **Steep initial slope:** High marginal return at low spend (under-invested)
- **Flat at current spend:** Near saturation (diminishing returns)
- **Inflection point:** Where marginal returns start declining sharply

### Marginal ROAS
```python
# Marginal ROAS at current spend levels
model.plot.marginal_curve()
model.plot.uplift_curve()
```

## Decomposition

### Waterfall Decomposition
```python
# Shows: intercept + controls + seasonality + each channel = total
model.plot.waterfall_components_decomposition()
```

### Time Series Decomposition
```python
# Contributions over time per channel
model.plot.contributions_over_time()

# Channel contribution share with HDI
model.plot.channel_contribution_share_hdi()
```

### Component Breakdown
The model decomposes the target into:
1. **Intercept** -- baseline level (stored as `intercept_contribution`)
2. **Control effects** -- impact of control variables (seasonality, macro, holidays)
3. **Fourier seasonality** -- yearly seasonal patterns
4. **Channel contributions** -- attributed effect of each media channel
5. **Residual** -- unexplained variation

## Results Interpretation Framework

### Step 1: Sanity Check
- Do contribution shares sum to approximately 1.0?
- Are all contributions positive? (Negative = possible model misspecification)
- Does the intercept make sense as a baseline?
- Are seasonal patterns as expected?

### Step 2: Channel Performance Assessment
For each channel, assess:
1. **Contribution share** -- what % of total media effect?
2. **Spend share** -- what % of total budget?
3. **ROAS** -- return per unit of spend
4. **Efficiency ratio** -- contribution share / spend share
   - Ratio > 1: Efficient (contributing more than its budget share)
   - Ratio < 1: Inefficient (contributing less than its budget share)

### Step 3: Investment Signals
| Signal | Condition | Recommendation |
|--------|-----------|----------------|
| Under-invested | High ROAS + low spend share + steep saturation curve | Increase spend |
| Over-invested | Low ROAS + high spend share + flat saturation curve | Decrease spend |
| Efficient | ROAS > median + not saturated | Maintain or cautiously increase |
| Budget-neutral | ROAS ~1.0, contribution ~ spend share | Maintain current level |

### Step 4: Cross-Model Validation (Multi-Model)
If running multiple models (sales, search, SEM funnel):
- Compare channel rankings across models
- Check for indirect effects (e.g., Meta drives search which drives SEM which drives sales)
- True ROI of a channel = direct effect + indirect effects through funnel stages
- Channels appearing in multiple models' top contributors are robustly important

## Presenting Results to Stakeholders

### Executive Summary Format
1. **Key finding** (1 sentence): "Digital Display drives 43% of attributed sales with strong ROAS of 7.2"
2. **Channel ranking table**: Spend share, contribution share, ROAS, efficiency ratio
3. **Investment recommendations**: Which channels to increase/decrease/maintain
4. **Confidence**: R² metrics, convergence status, CV performance
5. **Caveats**: Model assumptions, data limitations, indirect effects not captured

### Visualization Priority
1. Waterfall chart (overall decomposition)
2. Channel contribution share bar chart (with HDI)
3. ROAS comparison chart
4. Saturation curves (showing diminishing returns)
5. Budget optimization comparison (current vs. optimal)

## Common Interpretation Pitfalls

1. **Confusing marginal vs. average ROAS** -- MMM gives marginal (incremental) ROAS
2. **Ignoring indirect effects** -- Meta's true ROI includes its effect on search/SEM
3. **Over-interpreting small channels** -- Channels with <2% spend have noisy estimates
4. **Ignoring uncertainty** -- Always show HDI/credible intervals, not just point estimates
5. **Comparing across different targets** -- ROAS from a sales model vs. GQV model are not comparable
