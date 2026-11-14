---
description: Run the data quality audit on the dataset in spec.yaml, then turn it into a prioritised improvement plan (fix now + improve data collection going forward) and ideas for better use of the dataset. Writes to ./mmm-workspace/audit/.
---

# MMM Data Audit and Improvement Plan

## Steps

1. Check that `./mmm-workspace/spec.yaml` exists. If not: "Run `/mmm-intake` first to create
   the project spec."

2. Run the audit and the advisor:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))

   from agent_mmm.spec import load_spec
   from agent_mmm.utils.io import load_data
   from agent_mmm.data_audit import run_audit
   from agent_mmm.data_advisor import run_advisor

   spec = load_spec("./mmm-workspace/spec.yaml")
   df = load_data(spec.data_path)
   findings = run_audit(spec, base=".", df=df)
   tips = run_advisor(spec, findings, df=df, base=".")

   s = findings["summary"]
   print(f"Data Quality Tier: {s['data_quality_tier']}")
   print(f"Rows: {s['rows']} | Channels: {s['channels']} | "
         f"Errors: {s['total_errors']} | Warnings: {s['total_warnings']}")
   print("\nScorecard:")
   for area, status in tips.scorecard.items():
       print(f"  {area:32s} {status}")
   print(f"\nTips: {len(tips.tips)} | Opportunities: {len(tips.opportunities)}")
   print("Reports: mmm-workspace/audit/audit_report.md, mmm-workspace/audit/data_quality_tips.md")
   EOF
   ```

3. Read both reports. Present them in this order:
   - **Verdict and blocking issues** from `audit_report.md`.
   - **Top 3-5 tips** from `data_quality_tips.md`, critical first. For each give the
     evidence, the fix for this model, and the change to data collection.
   - **Better use of this dataset**: unused columns, geo detail, delivery metrics.
   - Offer the **data collection checklist** as something the user can hand to whoever
     owns the data feeds.

4. Advise on next steps:
   - **FAIL** → what must be fixed before modelling. Offer `/mmm-prepare-data` for the
     fixes that are mechanical; the rest need the data owner.
   - **WARN** → which warnings are handled by controls/priors and which limit what the
     model can claim.
   - **PASS** → "Data looks good — run `/mmm-recommend-controls` next."

Do not edit the data or the spec based on a tip without the user's agreement. Tips about
geo panels, unused columns and alternative KPIs change the model's scope; discuss them.
