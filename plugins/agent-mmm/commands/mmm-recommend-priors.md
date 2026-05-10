---
description: Generate Bayesian prior parameter recommendations for all channels in your MMM. Produces model_config.json and a prior audit report.
---

# MMM Prior Recommendations

Generate channel-specific Bayesian prior parameters using the prior recommendation engine.

## Steps

1. Check `./mmm-workspace/spec.yaml` exists. If not: "Run `/mmm-intake` first."

2. Run the prior engine:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib

   home = pathlib.Path.home()
   for search_root in [home / '.claude', home / '.config/claude']:
       for p in search_root.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent))
           break

   try:
       import agent_mmm
   except ImportError:
       print("ERROR: agent_mmm library not found.")
       sys.exit(1)

   from agent_mmm.spec import load_spec
   from agent_mmm.prior_engine import recommend_priors

   spec = load_spec("./mmm-workspace/spec.yaml")
   result = recommend_priors(spec, base=".")

   print(f"Generated priors for {result['n_channels']} channels")
   if result['warnings']:
       print("Warnings:")
       for w in result['warnings']:
           print(f"  {w}")
   print("Report: ./mmm-workspace/priors/prior_audit_report.md")
   print("Model config: ./mmm-workspace/priors/model_config.json")
   EOF
   ```

3. Display the prior audit report: `cat ./mmm-workspace/priors/prior_audit_report.md`

4. Review each channel's priors with the user:
   - Explain the rationale for each channel type classification
   - Highlight any sparse channels with widened priors
   - Walk through the adstock recommendation
   - Confirm the prior predictive check guidance

5. Ask: "Do any channel type classifications look wrong? For example, if `spend_fb` was classified as 'social' but you want it treated as 'meta', we can update the channel_type in spec.yaml."

6. If corrections needed, help the user update `spec.yaml` channels[] and re-run.

7. After confirmation:
   > Priors saved to `./mmm-workspace/priors/model_config.json`
   >
   > **Next step**: Run `/mmm-build` to construct the model, then `/mmm-fit` to run the prior predictive check and fit.
