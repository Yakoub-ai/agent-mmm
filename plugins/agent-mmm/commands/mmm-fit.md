---
description: Fit the MMM model — runs prior predictive check, MCMC sampling, and posterior predictive check. Attaches any lift-test constraints, then saves the model and metrics to ./mmm-workspace/runs/<run-id>/.
---

# MMM Fit

Run the full fit pipeline: build → prior predictive check → lift-test calibration → MCMC → posterior predictive → save.

## Steps

1. Check `./mmm-workspace/spec.yaml` and `./mmm-workspace/priors/model_config.json` exist.

2. Ask the user: "Which sampling profile? `quick` (500 draws / 1000 tune / 4 chains) for
   iteration, or `final` (2000 / 3000 / 4 at target_accept 0.97) for anything that will
   move budget. Default: quick."

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

   with open("./mmm-workspace/priors/model_config.json") as f:
       priors = json.load(f)

   metrics = run_fit(spec, priors=priors, sampler_config=sampler, base=".")

   print(f"\nRun complete: {metrics['run_id']}")
   print(f"In-sample R2: {metrics.get('r2_insample')} "
         f"(per posterior-predictive draw, so it includes observation noise)")
   ppc = metrics.get("prior_pc", {})
   if ppc.get("coverage_90") is not None:
       print(f"Prior predictive 90% coverage: {ppc['coverage_90']}")
       for note in ppc.get("notes", []):
           print(f"  ! {note}")
   cal = metrics.get("calibration", {})
   print("Calibration: " + (
       f"{cal.get('n_tests')} lift test(s) on {cal.get('channels')}"
       if cal.get("applied") else f"none — {cal.get('reason','')}"))
   print(f"Model: {metrics.get('model_path')}")
   EOF
   ```
   Replace `QUICK_OR_FINAL` with the user's choice.

4. After fit completes:
   - Read and display `./mmm-workspace/runs/<run-id>/metrics.json`
   - Run `/mmm-diagnose` automatically (or tell the user to run it)

5. If the fit fails or warns:
   - **Divergences** → raise `target_accept`; if they persist the geometry is the problem,
     usually a channel with almost no spend variation. That is a specification fix.
   - **Low ESS** → more draws.
   - **Memory** → fewer chains, or fewer geos while validating the pipeline.
   - **`ValueError: cannot write NetCDF files`** → `pip install h5netcdf`. The model fitted
     fine; only the save failed.

6. Report the prior predictive result even when the fit succeeds. Coverage below ~0.8 means
   the priors assign little probability to what actually happened; a band more than ~10x
   the data range means they are so vague the sampler will waste its time.
