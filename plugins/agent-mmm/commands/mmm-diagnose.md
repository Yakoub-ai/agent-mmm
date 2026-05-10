---
description: Run diagnostics on a completed MMM run — convergence (rhat/ESS/divergences), overfit gap, prior-pull detection, and attribution plausibility.
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

   idata_path = runs_dir / run_id / "idata.nc"
   metrics_path = runs_dir / run_id / "metrics.json"

   findings = run_diagnostics(
       run_id=run_id,
       idata_path=str(idata_path) if idata_path.exists() else None,
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

6. Based on findings, provide concrete recommendations:
   - High rhat: suggest `target_accept=0.99` or widen priors
   - Divergences: suggest widening priors first, then reparameterization
   - Overfit: suggest adding controls, reducing Fourier modes, or widening priors
   - Prior pull: identify which parameter is pulled and suggest widening its sigma
   - Low ESS: suggest more draws or fewer parameters

7. If PASS: "✅ Model passes diagnostics — run `/mmm-improve` to run the improvement tournament, or `/mmm-report` to generate stakeholder reports."
