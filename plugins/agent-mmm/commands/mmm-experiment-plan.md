---
description: Produce the experiment roadmap — which incrementality tests to run, in what order, whether each is adequately powered, what result to expect, how to execute it, and how it feeds back into the model.
---

# MMM Experiment Plan

An MMM without an experiment is an argument from correlation. This produces the plan that
fixes that.

**An underpowered test is worse than no test**: it returns a null, the null is read as "the
channel does nothing", and budget moves on evidence that was never there.

## Steps

1. Establish what is available:
   * How many geos can be bought and reported separately?
   * Which channels can be suppressed by geo, and which are national-only?
   * How much revenue is the business willing to put at risk, and who signs that off?
   * Which calendar periods are off-limits? This usually constrains the roadmap more than
     statistics does.

2. Measure the noise from the data — do not assume it:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-.}"
   python3 - "$PLUGIN_ROOT" <<'PY'
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   import pandas as pd
   from agent_mmm.experiments import estimate_geo_cv, estimate_pre_period_correlation

   panel = pd.read_csv("mmm-workspace/prepared/dataset.csv")          # <-- edit
   cv  = estimate_geo_cv(panel, geo_column="geo", target_column="revenue")
   rho = estimate_pre_period_correlation(panel, geo_column="geo",
                                         target_column="revenue", date_column="date")
   print(f"cross-geo CV: {cv:.3f}")
   print(f"pre/post correlation: {rho:.3f}")
   PY
   ```

   `rho` is the single most important input to geo-test power and it more than halves the
   detectable effect at rho = 0.9. Measure it.

3. Build the roadmap:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-.}"
   python3 - "$PLUGIN_ROOT" <<'PY'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   from agent_mmm.experiments import ChannelTestCandidate, build_roadmap, render_roadmap

   candidates = [                                                     # <-- from the fitted model
       ChannelTestCandidate("search_brand", spend=900_000, contribution_share=0.12,
                            roas_point=8.2, roas_ci_low=3.0, roas_ci_high=14.0,
                            always_on=True, geo_testable=True),
       ChannelTestCandidate("tv", spend=2_400_000, contribution_share=0.09,
                            roas_point=1.4, roas_ci_low=0.6, roas_ci_high=2.6,
                            geo_testable=False),
   ]

   roadmap = build_roadmap(candidates, n_geos_available=40, geo_cv=0.35,
                           pre_period_correlation=0.92, duration_periods=8,
                           pre_period_periods=12)

   out = pathlib.Path("mmm-workspace/experiments"); out.mkdir(parents=True, exist_ok=True)
   (out / "roadmap.md").write_text(render_roadmap(roadmap), encoding="utf-8")
   (out / "roadmap.json").write_text(json.dumps(roadmap.to_dict(), indent=2), encoding="utf-8")

   print(f"{len(roadmap.viable_plans)}/{len(roadmap.plans)} viable; "
         f"revenue at risk {roadmap.total_spend_at_risk:,.0f}")
   for p in roadmap.plans:
       pw = p.power
       exp = f"{pw.expected_lift_relative:.1%}" if pw.expected_lift_relative is not None else "n/a"
       print(f"  {p.rank}. {p.channel:22s} {p.design:14s} mde={pw.mde_relative:6.1%} "
             f"expected={exp:>6s} {'OK' if p.viable else 'BLOCKED'}")
   for w in roadmap.warnings:  print("WARN:", w)
   for q in roadmap.questions: print("Q   :", q)
   PY
   ```

   If no model has been fitted, leave `contribution_share` unset. The roadmap will say
   there is no pre-registered expectation rather than inventing one.

4. Walk the user through `mmm-workspace/experiments/roadmap.md`, test by test:
   * what it can detect, versus what we expect it to show,
   * what the result would have to be to change our mind, in both directions,
   * how much revenue is at risk,
   * how it is executed, and what would invalidate it.

5. For blocked tests, present the options concretely: more geos, a longer window, a larger
   holdout share, or a better pre-period covariate. Say which is cheapest here.

6. Agree the first test, its owner and its window. **Write the expected result down before
   it runs** — a test with no prediction cannot be surprising, and therefore teaches nothing.

7. When the result comes back, feed it in as a calibration constraint
   (`add_lift_test_measurements`, Meridian ROI priors, Robyn `calibration_input`), refit,
   and check both that the channel moved towards the test and that the other channels did
   not quietly absorb the difference.

Load `agent-mmm:mmm-experiment-roadmap` before interpreting any of this.
