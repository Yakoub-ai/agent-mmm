---
description: Generate stakeholder reports for the best MMM run (cmo, cfo, mops, ds, or all), plus the visual report — PNG charts, a self-contained interactive HTML report and a markdown summary, covering every framework fitted.
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
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))

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

5. Build the visual report from every run with `results.json` (the latest per framework),
   including the comparison when there are two or more:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   from agent_mmm.results_schema import load_all
   from agent_mmm.compare import compare
   from agent_mmm.charts import build_report

   latest = {}
   for r in load_all("mmm-workspace/runs"):
       latest[r.framework] = r
   results = list(latest.values())
   if not results:
       print("No results.json found. Run /mmm-fit first."); raise SystemExit(1)
   comp = compare(results) if len(results) > 1 else None
   out = build_report(results, comp, base=".")
   print(f"HTML report: {out['html']}")
   print(f"Summary:     {out['markdown']}")
   for group, paths in out["figures"].items():
       print(f"{group}: {len(paths)} charts")
   EOF
   ```
   The HTML file is self-contained (no internet needed, nothing loaded from outside) and safe to
   email. If matplotlib or plotly is missing, re-run `/mmm-setup` (they are in the `report` extra).

6. Display each generated report or offer to open them.

7. Explain key findings to the user in plain language tailored to the requested persona.
