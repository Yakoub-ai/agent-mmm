---
description: Run the iterative improvement tournament — fits N model variants per round, scores them, and uses the winner's posterior to tighten priors for the next round. Persists leaderboard.
---

# MMM Iterative Improvement

Run the tournament + posterior-informed refinement loop.

## Steps

1. Check `./mmm-workspace/spec.yaml` and `./mmm-workspace/priors/model_config.json` exist.

2. Ask the user:
   - "How many tournament rounds? (default: 3)"
   - "How many variants per round? (default: 4)"

3. Explain what will happen:
   > I'll run [rounds × variants] model configurations, score each on CV R² × (1 − overfit gap) × convergence × plausibility, then use the best model's posterior to tighten priors for the next round. Each round takes [rounds × variants × ~15min] minutes approximately.

4. Run the tournament:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib, json
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   import logging
   logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

   from agent_mmm.iter_loop import run_tournament

   result = run_tournament(
       spec_path="./mmm-workspace/spec.yaml",
       model_config_path="./mmm-workspace/priors/model_config.json",
       max_rounds=MAX_ROUNDS,
       n_variants_per_round=N_VARIANTS,
       base=".",
   )

   print(f"\nTournament complete!")
   print(f"Best score: {result['best_score']:.3f}")
   print(f"Best run: {result['best_run_id']}")
   print(f"Leaderboard: ./mmm-workspace/leaderboard.json")

   # Show top 3
   print("\nTop 3 runs:")
   for i, entry in enumerate(result['leaderboard'][:3]):
       print(f"  {i+1}. {entry['run_id']} | score={entry['score']:.3f} | R²={entry.get('r2_insample', 'N/A')}")
   EOF
   ```
   Replace `MAX_ROUNDS` and `N_VARIANTS` with user choices.

5. Display the leaderboard: `cat ./mmm-workspace/leaderboard.json | python3 -m json.tool | head -80`

6. Identify the best run and recommend it for reporting:
   > Tournament complete. Best run: `<run-id>` (score: X.XXX)
   >
   > Run `/mmm-diagnose` to get the full diagnostic report for the best run, then `/mmm-report` to generate stakeholder reports.
