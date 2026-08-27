---
description: Run diagnostics on a completed MMM run — convergence, fit, generalisation, baseline health, prior-to-posterior learning, and attribution plausibility.
---

# MMM Diagnostics

Run the full diagnostic suite on a completed run.

## Steps

1. List completed runs: `ls ./mmm-workspace/runs/ 2>/dev/null || echo "No runs found"`

2. If no runs: "Run `/mmm-fit` first."

3. If multiple runs, ask: "Which run to diagnose? (Enter run-id or press Enter for latest)"

4. Run diagnostics:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib, json, os
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   from agent_mmm.diagnostics import run_diagnostics

   runs_dir = pathlib.Path("./mmm-workspace/runs")
   run_id = "RUN_ID_PLACEHOLDER"  # replace with chosen run-id
   
   if not run_id or run_id == "latest":
       runs = sorted(runs_dir.iterdir()) if runs_dir.exists() else []
       if not runs:
           print("No runs found"); sys.exit(1)
       run_id = runs[-1].name

   model_path = runs_dir / run_id / "model.nc"
   metrics_path = runs_dir / run_id / "metrics.json"

   findings = run_diagnostics(
       run_id=run_id,
       idata_path=str(model_path) if model_path.exists() else None,
       metrics_path=str(metrics_path) if metrics_path.exists() else None,
       base=".",
   )

   tier = findings["summary"]["tier"]
   print(f"\nDiagnostics: {tier}")
   for e in findings["errors"]: print(f"  ❌ {e}")
   for w in findings["warnings"]: print(f"  ⚠️  {w}")
   print(f"\nReport: ./mmm-workspace/runs/{run_id}/diagnostics_report.md")
   EOF
   ```

5. Display the report: `cat ./mmm-workspace/runs/<run-id>/diagnostics_report.md`

6. Read the report in order and **stop at the first failure** — an answer at a later stage
   is meaningless if an earlier one failed.

   | Failure | What it means | What to do |
   |---|---|---|
   | Divergences | The sampler could not reach part of the posterior, so the samples are biased | `target_accept` to 0.95, then 0.99. If they persist, the geometry is the problem — usually a channel with almost no spend variation. Group collinear channels or drop constant ones |
   | r-hat > 1.05 | Chains disagree about where the posterior is | More `tune` first; then check `az.plot_trace` on the named parameter for multimodality, which in MMM means two channels can swap roles |
   | Low ESS | Intervals on that parameter are noisy | More draws; if only one parameter, check its prior scaling |
   | **Negative baseline** | The model claims the business would sell less than nothing without marketing. **Every channel number is inflated** | Add the missing driver (price, distribution, trend). Do not proceed to attribution |
   | Baseline < 30% | An omitted confounder is being credited to media | Usually price or distribution |
   | Baseline > 95% | The trend or seasonality is eating media | Coarsen the lengthscale, cut Fourier modes |
   | Overfit gap > 0.20 | The model memorised the history | Tighten priors, cut Fourier modes, merge collinear channels |
   | Prior-dominated parameters | The data did not identify them | Not a bug — a reporting obligation. Say so, and propose an experiment |
   | One channel > 70% (≥3 channels) | Collinearity | Check VIF and the spend correlation matrix |
   | Media share > 60% | The baseline is starved | Check the control set before the channels |

7. If PASS: "Model passes diagnostics. Run `/mmm-improve` for the tournament, or `/mmm-report` for stakeholder reports."

8. If no cross-validation has been run, say so: a model that has never been asked to predict
   an unseen week has not been validated, whatever its in-sample fit.
