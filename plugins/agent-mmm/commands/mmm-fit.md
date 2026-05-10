---
description: Fit the MMM model — runs prior predictive check, MCMC sampling, and posterior predictive check. Saves InferenceData to ./mmm-workspace/runs/<run-id>/.
---

# MMM Fit

Run the full fit pipeline: prior predictive check → MCMC sampling → posterior predictive check → save artifacts.

## Steps

1. Check `./mmm-workspace/spec.yaml` and `./mmm-workspace/priors/model_config.json` exist.

2. Ask the user: "Which sampling mode? `quick` (draws=500, tune=1000, chains=2) for iteration, or `final` (draws=2000, tune=3000, chains=4) for production. Default: quick."

3. Run the fit:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib, json
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   import logging
   logging.basicConfig(level=logging.INFO)

   from agent_mmm.spec import load_spec
   from agent_mmm.fit_runner import run_fit, SAMPLER_QUICK, SAMPLER_FINAL

   spec = load_spec("./mmm-workspace/spec.yaml")

   mode = "QUICK_OR_FINAL"  # replace with user choice
   sampler = SAMPLER_FINAL if mode == "final" else SAMPLER_QUICK

   metrics = run_fit(
       spec,
       model_config_path="./mmm-workspace/priors/model_config.json",
       sampler_config=sampler,
       base=".",
   )
   
   print(f"\n✅ Run complete: {metrics['run_id']}")
   print(f"In-sample R²: {metrics.get('r2_insample')}")
   print(f"InferenceData: {metrics.get('idata_path')}")
   EOF
   ```
   Replace `QUICK_OR_FINAL` with the user's choice.

4. After fit completes:
   - Read and display `./mmm-workspace/runs/<run-id>/metrics.json`
   - Run `/mmm-diagnose` automatically (or tell the user to run it)

5. If fit fails, check for common issues:
   - Divergences during sampling → suggest widening priors
   - Low ESS → suggest increasing draws or target_accept
   - Memory errors → suggest fewer chains
