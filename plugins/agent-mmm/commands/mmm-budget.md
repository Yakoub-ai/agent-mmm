---
description: Budget scenarios with a guardrail ladder — the same budget optimised at small (±10%), moderate (±25%), large (±50%) and aggressive (0–2×) levels, at several total budgets, with uncertainty, an efficiency frontier, cross-framework agreement and a recommendation that explains what small vs larger adjustments buy.
---

# MMM Budget Scenarios

Never hand over one "optimal" allocation. Show what each level of ambition buys and what it
risks, then recommend one with the reason.

## Steps

1. Needs `results.json` from `/mmm-fit` (and ideally `/mmm-compare`). If the model did not
   converge, say first that every scenario is indicative only.

2. Ask the user (offer the defaults, accept "defaults"):
   - **Budget levels** — default −20%, −10%, same, +10%, +20%.
   - **Guardrails** — default ladder: small ±10%, moderate ±25%, large ±50%, aggressive 0–2×.
     Offer custom levels if they have real constraints (e.g. "nothing may move more than 15%").
   - **Locked channels** — committed spend that cannot move (e.g. a TV upfront), and any
     per-channel min/max.
   - **Disputed channels** (frameworks disagree) are held to ±10% at every level. Release them
     only if the user explicitly asks.

3. Run the scenarios. Fill in the lists/dicts from step 2, or leave the defaults:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   from agent_mmm.results_schema import load_all
   from agent_mmm.compare import compare
   from agent_mmm.budget import run_budget, GUARDRAILS, BUDGET_FACTORS, Guardrail

   BUDGET = BUDGET_FACTORS            # e.g. (0.85, 1.0, 1.15)
   LADDER = GUARDRAILS                # e.g. (Guardrail("tight", "Tight (±15%)", 0.85, 1.15),) + GUARDRAILS
   LOCKED = {}                        # e.g. {"spend_tv": 1200000}  (amount over the training window)
   BOUNDS = {}                        # e.g. {"spend_search": (400000, 600000)}
   RELEASE_DISPUTED = False

   latest = {}
   for r in load_all("mmm-workspace/runs"):
       latest[r.framework] = r
   results = list(latest.values())
   if not results:
       print("No results.json found. Run /mmm-fit first."); raise SystemExit(1)
   comp = compare(results) if len(results) > 1 else None
   out = run_budget(results, comp, base=".", budget_factors=tuple(BUDGET), guardrails=tuple(LADDER),
                    locked=LOCKED, custom_bounds=BOUNDS, release_disputed=RELEASE_DISPUTED)
   for p in out["plans"]:
       print(f"\n== {p.framework} ({p.run_id})")
       for f, rec in p.recommendations.items():
           print(f"  budget {float(f) - 1:+.0%}: {rec['summary']}")
       for n in p.notes:
           print(f"  note: {n}")
   print(json.dumps(out["written"], indent=1))
   EOF
   ```

4. Present it as a conversation, not a table dump:
   - **The ladder at today's budget**, smallest first: "Small adjustments (±10%) bring
     +X (+a%) with P% confidence; moderate brings +Y; large brings +Z but relies more on
     the curves." Use `guardrail_ladder.png` and `allocation_by_guardrail.png`.
   - **The recommendation and why it stops where it does** (confidence bar, diminishing
     returns, extrapolation), from the report's reasons.
   - **What moves:** which channels go up and down at the recommended level.
   - **Budget levels:** for cuts and increases, the value of re-allocating versus scaling
     today's mix, and the marginal return on extra budget (`efficiency_frontier.png`).
   - **Agreement:** if both frameworks were run, `agreement.md` — act on channels where they
     agree, hold and test where they conflict.
   - **Caveats:** non-convergence, disputed channels held, channels not optimised, and that
     amounts cover the training window.

5. Offer next steps: `/mmm-experiment-plan` for channels where the recommendation depends
   on extrapolation or disagreement, and `/mmm-report` to put the scenarios in the HTML report.
