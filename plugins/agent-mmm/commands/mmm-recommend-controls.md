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
   python3 - <<'EOF'
   import sys, pathlib, json

   # Find agent_mmm lib
   home = pathlib.Path.home()
   for search_root in [home / '.claude', home / '.config/claude']:
       for p in search_root.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent))
           break

   try:
       import agent_mmm
   except ImportError:
       print("ERROR: agent_mmm library not found. Check plugin installation.")
       sys.exit(1)

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
