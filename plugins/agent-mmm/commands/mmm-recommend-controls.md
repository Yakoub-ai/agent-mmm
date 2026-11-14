---
description: Recommend external factor and control variable columns for your MMM based on industry, region, and detected data patterns.
---

# MMM Controls & External Factors Recommendation

Generate control variable recommendations tailored to your industry and region.

## Steps

1. Check that `./mmm-workspace/spec.yaml` exists. If not: "Run `/mmm-intake` first."

2. Check if `./mmm-workspace/audit/audit.json` exists (load findings if present).

3. Run the controls engine:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))

   from agent_mmm.spec import load_spec
   from agent_mmm.controls_engine import recommend_controls

   spec = load_spec("./mmm-workspace/spec.yaml")
   
   audit = None
   try:
       with open("./mmm-workspace/audit/audit.json") as f:
           audit = json.load(f)
   except FileNotFoundError:
       pass

   result = recommend_controls(spec, audit_findings=audit, base=".")
   print(f"Generated {result['total_recommendations']} recommendations")
   print(f"Report: ./mmm-workspace/controls/recommendations_report.md")
   EOF
   ```

4. Display the recommendations report: `cat ./mmm-workspace/controls/recommendations_report.md`

5. Summarize the top recommendations and explain:
   - Which controls are most important for this industry and why
   - How to generate the "generated_from_date" controls (provide Python snippets)
   - Where to obtain external data sources (macro, search trends)

6. Ask the user: "Which of these controls do you have data for or can generate? I can help you create the date-based ones right now."

   If they want help generating date-based controls, provide Python code to generate them from the date column in their data file.

7. After the user decides, remind them:
   > Add chosen controls to `./mmm-workspace/spec.yaml` under `controls:`, then run `/mmm-analyze-data` again to verify the updated dataset.
