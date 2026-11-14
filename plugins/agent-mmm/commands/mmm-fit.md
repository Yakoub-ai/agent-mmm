---
description: Fit the MMM on pymc-marketing and, optionally, Google Meridian (in its own environment). Every framework uses the same training window and holdout, and writes a framework-neutral results.json for /mmm-compare, charts and budget scenarios.
---

# MMM Fit

Run the full fit pipeline: build → prior predictive check → lift-test calibration → MCMC → posterior predictive → save.

## Steps

1. Check `./mmm-workspace/spec.yaml` and `./mmm-workspace/priors/model_config.json` exist.

2. Ask which frameworks to fit: **pymc-marketing** (default), **Meridian**, or **both** (needed
   for `/mmm-compare`). Meridian needs `.venv-meridian` (`/mmm-setup`, profile `meridian`).
   Every framework holds out the same trailing periods (`holdout="auto"`, about a quarter of a
   year, or `spec.validation` dates) so their forecast errors compare.

3. Ask the user: "Which sampling profile? `quick` (500 draws / 1000 tune / 4 chains) for
   iteration, or `final` (2000 / 3000 / 4 at target_accept 0.97) for anything that will
   move budget. Default: quick."

4. Run the pymc-marketing fit:
   ```bash
   PY="${MMM_PYTHON:-.venv-mmm/bin/python}"
   [ -x "$PY" ] || { echo "No MMM environment found. Run /mmm-setup first."; exit 1; }
   "$PY" -P - "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}" <<'EOF'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))

   import logging
   logging.basicConfig(level=logging.INFO)

   from agent_mmm.spec import load_spec
   from agent_mmm.fit_runner import run_fit, SAMPLER_QUICK, SAMPLER_FINAL

   spec = load_spec("./mmm-workspace/spec.yaml")

   mode = "QUICK_OR_FINAL"  # replace with user choice
   sampler = SAMPLER_FINAL if mode == "final" else SAMPLER_QUICK

   with open("./mmm-workspace/priors/model_config.json") as f:
       priors = json.load(f)

   metrics = run_fit(spec, priors=priors, sampler_config=sampler, base=".", holdout="auto")

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
   hf = metrics.get("holdout_fit") or {}
   if hf.get("mape_holdout") is not None:
       print(f"Holdout MAPE ({metrics['holdout_periods']} periods): {hf['mape_holdout']:.1%}")
   print(f"Results: {metrics.get('results_path') or metrics.get('results_error')}")
   EOF
   ```

5. If Meridian was chosen, run it as a separate process in its own environment. Map the
   profile: quick → `--chains 4 --adapt 500 --burnin 500 --keep 500`; final → `--keep 1000`
   with `--adapt 1000 --burnin 1000`.
   ```bash
   MER="${MMM_MERIDIAN_PYTHON:-.venv-meridian/bin/python}"
   [ -x "$MER" ] || { echo "No Meridian environment. Run /mmm-setup with profile meridian."; exit 1; }
   RUN_ID="$(date +%Y-%m-%dT%H-%M)_meridian"
   TF_CPP_MIN_LOG_LEVEL=3 "$MER" -P "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}/lib/agent_mmm/meridian_runner.py" \
       --spec mmm-workspace/spec.yaml --run-id "$RUN_ID" --holdout auto \
       --chains 4 --adapt 500 --burnin 500 --keep 500
   ```
   Meridian runs with its default ROI priors; say so when presenting its numbers.
   Replace `QUICK_OR_FINAL` with the user's choice.

6. After fit completes:
   - Read and display `./mmm-workspace/runs/<run-id>/metrics.json`
   - Run `/mmm-diagnose` automatically (or tell the user to run it)

7. If the fit fails or warns:
   - **Divergences** → raise `target_accept`; if they persist the geometry is the problem,
     usually a channel with almost no spend variation. That is a specification fix.
   - **Low ESS** → more draws.
   - **Memory** → fewer chains, or fewer geos while validating the pipeline.
   - **`ValueError: cannot write NetCDF files`** → `pip install h5netcdf`. The model fitted
     fine; only the save failed.

8. Report the prior predictive result even when the fit succeeds. Coverage below ~0.8 means
   the priors assign little probability to what actually happened; a band more than ~10x
   the data range means they are so vague the sampler will waste its time.
