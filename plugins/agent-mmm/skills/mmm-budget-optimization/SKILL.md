---
name: mmm-budget-optimization
description: |
  Budget allocation optimization and sensitivity analysis for Marketing Mix Models. Use when optimizing media budget allocation, setting up BudgetOptimizer with CustomModelWrapper, defining budget bounds and constraints, running sensitivity analysis, comparing current vs. optimal allocation, or advising on budget reallocation strategies. Also activate when the user asks about optimal spend, budget constraints, or allocation scenarios.
---

# MMM Budget Optimization

## Setup: BudgetOptimizer + CustomModelWrapper

The multidimensional MMM does **NOT** have `model.optimize_budget()`. Use the standalone optimizer:

```python
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer, CustomModelWrapper

# Step 1: Wrap the fitted model
wrapper = CustomModelWrapper(
    base_model=model.model,    # The PyMC model object (not the MMM wrapper)
    idata=model.idata,         # InferenceData with posterior
    channels=channel_columns,  # List of channel column names
)

# Step 2: Create optimizer
optimizer = BudgetOptimizer(
    model=wrapper,
    num_periods=len(X),        # Number of time periods in the data
)
```

## Running Optimization

### Basic Optimization
```python
# Calculate current allocation
current_alloc = {ch: X[ch].sum() for ch in channel_columns}
total_budget = sum(current_alloc.values())

# Set bounds (e.g., +/- 50% per channel)
budget_bounds = {
    ch: (spend * 0.5, spend * 1.5)
    for ch, spend in current_alloc.items()
}

# Optimize
opt_alloc, opt_result = optimizer.allocate_budget(
    total_budget=total_budget,
    budget_bounds=budget_bounds,
)

# Results
print(f"Optimal allocation: {opt_alloc}")
print(f"Expected improvement: {opt_result}")
```

### With Custom Constraints
```python
from pymc_marketing.mmm.budget_optimizer import Constraint

# Example: TV must be at least 20% of total budget
def tv_min_constraint(x):
    tv_idx = channel_columns.index("spend_TV")
    return x[tv_idx] - 0.20 * total_budget  # >= 0

optimizer.set_constraints([
    Constraint(type="ineq", fun=tv_min_constraint),
])
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
