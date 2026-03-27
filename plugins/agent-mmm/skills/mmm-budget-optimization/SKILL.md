---
name: mmm-budget-optimization
description: |
  Budget allocation optimization and sensitivity analysis for Marketing Mix Models. Use when optimizing media budget allocation, setting up BudgetOptimizer with CustomModelWrapper, defining budget bounds and constraints, running sensitivity analysis, comparing current vs. optimal allocation, or advising on budget reallocation strategies. Also activate when the user asks about optimal spend, budget constraints, or allocation scenarios.
---

# MMM Budget Optimization

## Primary API: MultiDimensionalBudgetOptimizerWrapper (0.19.1+)

The multidimensional MMM does **NOT** have `model.optimize_budget()`. Use `MultiDimensionalBudgetOptimizerWrapper`:

```python
import xarray as xr
import pandas as pd
from pymc_marketing.mmm.multidimensional import MultiDimensionalBudgetOptimizerWrapper

# total_media_contribution_original_scale is added automatically during build_model
# No need to call add_original_scale_contribution_variable() first

start_date = pd.to_datetime(X["date"].min()).strftime("%Y-%m-%d")
end_date = pd.to_datetime(X["date"].max()).strftime("%Y-%m-%d")
wrapper = MultiDimensionalBudgetOptimizerWrapper(
    model=model,
    start_date=start_date,
    end_date=end_date,
)

# Budget bounds as xr.DataArray — dims=["channel", "bound"], bound coords = ["lower", "upper"]
current_alloc = {ch: float(X[ch].sum()) for ch in channel_columns}
total_budget = sum(current_alloc.values())
budget_bounds = xr.DataArray(
    data=[[v * 0.5, v * 1.5] for v in current_alloc.values()],
    dims=["channel", "bound"],
    coords={"channel": list(current_alloc.keys()), "bound": ["lower", "upper"]},
)

# Optimize
opt_alloc, opt_result = wrapper.optimize_budget(
    budget=total_budget,
    budget_bounds=budget_bounds,
    # response_variable defaults to "total_media_contribution_original_scale"
)

# Extract per-channel optimal spend
for ch in channel_columns:
    print(f"{ch}: {float(opt_alloc.sel(channel=ch).values):.0f}")
```

## Bayesian Posterior CIs via sample_response_distribution

```python
# Compare current vs optimal allocation with posterior uncertainty
current_alloc_da = xr.DataArray(
    [float(X[ch].mean()) for ch in channel_columns],  # weekly rates
    dims=["channel"],
    coords={"channel": channel_columns},
)
optimal_alloc_da = xr.DataArray(
    [float(opt_alloc.sel(channel=ch).values) / len(X) for ch in channel_columns],
    dims=["channel"],
    coords={"channel": channel_columns},
)

# Returns xr.Dataset with "total_media_contribution_original_scale" and "channel_contribution"
current_resp = wrapper.sample_response_distribution(current_alloc_da)
optimal_resp = wrapper.sample_response_distribution(optimal_alloc_da)

# total_media_contribution_original_scale has dim "sample" only (already summed over date)
current_total = current_resp["total_media_contribution_original_scale"].values.flatten()
optimal_total = optimal_resp["total_media_contribution_original_scale"].values.flatten()
uplift = optimal_total - current_total

print(f"Expected uplift: {uplift.mean():.0f} ({np.percentile(uplift, 5):.0f} to {np.percentile(uplift, 95):.0f})")
print(f"P(positive uplift): {(uplift > 0).mean() * 100:.1f}%")
```

## Legacy API: BudgetOptimizer (still available, dict bounds)

`BudgetOptimizer.allocate_budget()` still accepts dict bounds for backward compat:

```python
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer, CustomModelWrapper

wrapper = CustomModelWrapper(base_model=model.model, idata=model.idata, channels=channel_columns)
optimizer = BudgetOptimizer(model=wrapper, num_periods=len(X))
opt_alloc, opt_result = optimizer.allocate_budget(
    total_budget=total_budget,
    budget_bounds={ch: (v * 0.5, v * 1.5) for ch, v in current_alloc.items()},
)
```

### With Custom Constraints
```python
from pymc_marketing.mmm.budget_optimizer import Constraint

# Example: TV must be at least 20% of total budget
def tv_min_constraint(x):
    tv_idx = channel_columns.index("spend_TV")
    return x[tv_idx] - 0.20 * total_budget  # >= 0

# For MultiDimensionalBudgetOptimizerWrapper:
wrapper.optimize_budget(budget=total_budget, budget_bounds=budget_bounds,
                        constraints=[Constraint(type="ineq", fun=tv_min_constraint)])
```

## Budget Bounds Strategy

| Strategy | Bounds | When to Use |
|----------|--------|-------------|
| Conservative | +/- 20% | First optimization, stakeholder comfort |
| Moderate | +/- 50% | Standard optimization |
| Aggressive | +/- 80% | Exploratory, large budget flexibility |
| Asymmetric | Custom per channel | Channel-specific constraints (contracts, minimums) |

### Setting Realistic Bounds
```python
# Factor in practical constraints:
budget_bounds = {}
for ch, spend in current_alloc.items():
    ch_name = ch.replace("spend_", "")

    if ch_name == "TV":
        # TV has annual contracts -- limited flexibility
        budget_bounds[ch] = (spend * 0.85, spend * 1.15)
    elif ch_name == "SEM":
        # SEM is demand-driven -- can scale with demand
        budget_bounds[ch] = (spend * 0.50, spend * 2.00)
    else:
        # Default moderate bounds
        budget_bounds[ch] = (spend * 0.50, spend * 1.50)
```

## Sensitivity Analysis

### Budget Sensitivity Sweep
```python
# Test different total budget levels
budget_levels = [total_budget * f for f in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]

results = []
for budget in budget_levels:
    alloc, result = optimizer.allocate_budget(
        total_budget=budget,
        budget_bounds=budget_bounds,
    )
    results.append({"total_budget": budget, "allocation": alloc, "result": result})
```

### Per-Channel Sensitivity
```python
# How does outcome change when one channel's spend changes?
# Use saturation curves for this:
sat_curve = model.sample_saturation_curve(max_value=2.0, num_points=100)

# The marginal curve shows diminishing returns at different spend levels
model.plot.marginal_curve()
```

### Visualization
```python
model.plot.budget_allocation()                           # Allocation comparison
model.plot.allocated_contribution_by_channel_over_time() # Temporal allocation
model.plot.sensitivity_analysis()                        # Sensitivity sweep
```

## Interpreting Optimization Results

### Comparing Current vs. Optimal
```python
import pandas as pd

comparison = pd.DataFrame({
    "channel": [ch.replace("spend_", "") for ch in channel_columns],
    "current_spend": [current_alloc[ch] for ch in channel_columns],
    "optimal_spend": [opt_alloc[ch] for ch in channel_columns],
})
comparison["change_pct"] = (comparison["optimal_spend"] / comparison["current_spend"] - 1) * 100
comparison["current_share"] = comparison["current_spend"] / comparison["current_spend"].sum()
comparison["optimal_share"] = comparison["optimal_spend"] / comparison["optimal_spend"].sum()
```

### Decision Framework

| Optimization Signal | Meaning | Action |
|--------------------|---------|--------|
| Channel gets +50% | High marginal return, far from saturation | Increase investment |
| Channel gets -30% | Low marginal return, near saturation | Reduce or reallocate |
| Channel unchanged | Already near optimal | Maintain |
| Hits upper bound | Would allocate more if allowed | Consider widening bounds |
| Hits lower bound | Would allocate less if allowed | Review if channel is contractual |

### Validation Checks
1. **Total budget preserved:** `sum(optimal) == total_budget` (within tolerance)
2. **Bounds respected:** Each channel within specified min/max
3. **Predicted improvement is reasonable:** >20% improvement = verify, not magic
4. **Rerun with different starting points:** Check for local optima
5. **Business sense check:** Does the reallocation make marketing sense?

## Common Pitfalls

1. **Using model.optimize_budget()** -- Doesn't exist in multidimensional MMM. Use `BudgetOptimizer` + `CustomModelWrapper`.
2. **Too tight bounds** -- Over-constrained optimization may find no improvement
3. **Too loose bounds** -- Unrealistic reallocations that can't be executed in practice
4. **Ignoring saturation** -- If a channel is already saturated, more spend won't help
5. **Ignoring indirect effects** -- Budget optimization on sales model alone misses funnel effects
6. **Single-point estimate** -- Always consider uncertainty in optimization results
7. **Not accounting for lag** -- Budget changes take time to show effect (adstock)

## Advanced: Multi-Scenario Analysis

```python
# Compare scenarios
scenarios = {
    "current": current_alloc,
    "optimized_moderate": opt_alloc_moderate,   # +/- 20%
    "optimized_aggressive": opt_alloc_aggressive, # +/- 50%
    "digital_shift": digital_heavy_alloc,        # Manual scenario
}

for name, alloc in scenarios.items():
    # Use model to predict outcome under each scenario
    # (requires passing allocation through saturation + adstock transforms)
    print(f"{name}: expected outcome = ...")
```
