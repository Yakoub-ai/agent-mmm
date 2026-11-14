---
description: Compare fitted models across frameworks (pymc-marketing, Meridian) on the same holdout — forecast error, convergence, ROAS and contribution intervals per channel, where the frameworks agree and disagree, and which model (if any) is fit for decisions.
---

# MMM Compare

Needs two or more runs with `results.json` from the same spec and holdout (`/mmm-fit` with
"both"). The goal is not a winner by R². Where frameworks disagree, the data cannot pin the
channel down; where they agree with tight intervals, the number is robust.

## Steps

1. List runs with results: `ls mmm-workspace/runs/*/results.json`. If fewer than two, offer
   `/mmm-fit` with both frameworks. Ask which runs to compare if there are more than two
   per framework (default: the latest of each).

2. Run the comparison:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   from agent_mmm.results_schema import load_all
   from agent_mmm.compare import run_comparison

   results = load_all("mmm-workspace/runs")
   RUN_IDS = None   # replace with a list of run ids to restrict, e.g. ["..._pymc", "..._meridian"]
   if RUN_IDS:
       results = [r for r in results if r.run_id in RUN_IDS]
   else:  # latest per framework
       latest = {}
       for r in results:
           latest[r.framework] = r
       results = list(latest.values())
   c, md = run_comparison(results, base=".")
   print(f"Compared: {[m['label'] for m in c['models']]}")
   print(f"Recommendation: {c['recommendation']['model'] or 'none'} — {c['recommendation']['reason']}")
   for f in c["findings"]:
       print(f"- [{f['kind']}] {f['finding']}")
   print(f"Report: {md}")
   EOF
   ```

3. Present `mmm-workspace/comparison/comparison.md` in this order:
   - **Blocking or trust findings first.** A model that did not converge is not evidence.
   - **Forecast quality.** Under 2 points of MAPE is a tie; say so instead of picking.
   - **Per channel:** robust (agree, tight), uncertain (agree, wide), disagree. For each
     disagreement, offer `/mmm-experiment-plan`; that channel's budget should not move much
     on model evidence alone.
   - **Baseline share.** A big gap between models means the media/baseline split is not
     identified; go back to controls (stage 6) before trusting channel numbers.
   - **Priors caveat.** Meridian uses its default ROI priors; part of any gap is priors.

4. Record the outcome as part of stage 10 (validate and compare) in `/mmm-run`.
