---
description: Generate stakeholder reports for the best MMM run. Supports cmo, cfo, mops, ds, or all.
---

# MMM Stakeholder Report

Generate reports for one or all stakeholder personas.

## Usage

`/mmm-report [persona]` where persona is `cmo`, `cfo`, `mops`, `ds`, or `all` (default: all).

## Steps

1. Check `./mmm-workspace/spec.yaml` exists.

2. Find the best run from leaderboard or latest run:
   ```bash
   python3 -c "
   import json, pathlib, sys
   lb = pathlib.Path('./mmm-workspace/leaderboard.json')
   if lb.exists():
       data = json.load(open(lb))
       print(data.get('best_run_id', ''))
   else:
       runs = sorted(pathlib.Path('./mmm-workspace/runs').iterdir()) if pathlib.Path('./mmm-workspace/runs').exists() else []
       print(runs[-1].name if runs else '')
   "
   ```

3. Load metrics and diagnostics from the selected run.

4. Run the appropriate report generators:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib, json
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   from agent_mmm.spec import load_spec
   from agent_mmm.reports import (
       generate_cmo_report, generate_cfo_report,
       generate_mops_report, generate_ds_report
   )

   spec = load_spec("./mmm-workspace/spec.yaml")
   run_id = "RUN_ID"  # replace

   # Load available data
   metrics, diagnostics = {}, {}
   run_dir = pathlib.Path(f"./mmm-workspace/runs/{run_id}")
   if (run_dir / "metrics.json").exists():
       metrics = json.load(open(run_dir / "metrics.json"))
   if (run_dir / "diagnostics.json").exists():
       diagnostics = json.load(open(run_dir / "diagnostics.json"))

   persona = "PERSONA"  # replace with cmo/cfo/mops/ds/all

   generated = []
   if persona in ("cmo", "all"):
       generate_cmo_report(spec, run_id, metrics, diagnostics, base=".")
       generated.append("./mmm-workspace/reports/cmo.md")
   if persona in ("cfo", "all"):
       generate_cfo_report(spec, run_id, metrics, diagnostics, base=".")
       generated.append("./mmm-workspace/reports/cfo.md")
   if persona in ("mops", "all"):
       generate_mops_report(spec, run_id, metrics, diagnostics, base=".")
       generated.append("./mmm-workspace/reports/mops.md")
   if persona in ("ds", "all"):
       generate_ds_report(spec, run_id, metrics, diagnostics, base=".")
       generated.append("./mmm-workspace/reports/ds.md")

   for path in generated:
       print(f"Generated: {path}")
   EOF
   ```

5. Display each generated report or offer to open them.

6. Explain key findings to the user in plain language tailored to the requested persona.
