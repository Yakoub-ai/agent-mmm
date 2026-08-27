---
description: Compile spec.yaml plus recommended priors into the target framework, validate the model structure, and report anything the framework cannot express.
---

# MMM Build

Build the MMM model configuration from your spec and priors.

## Steps

1. Check `./mmm-workspace/spec.yaml` exists.
2. Check `./mmm-workspace/priors/model_config.json` exists. If not: "Run `/mmm-recommend-priors` first."
3. Compile the spec into the target framework and validate the structure:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   import json, pathlib
   from agent_mmm.spec import load_spec
   from agent_mmm.model_factory import compile_spec

   spec = load_spec("./mmm-workspace/spec.yaml")
   with open("./mmm-workspace/priors/model_config.json") as f:
       priors = json.load(f)

   result = compile_spec(spec, priors=priors)

   print(f"Framework: {result.framework} ({result.capability.value})")
   print(f"Channels:  {spec.channel_columns()}")
   print(f"Controls:  {spec.control_columns() or 'none'}")
   print(f"Structure: {result.structure}")

   # Anything the target framework cannot express MUST reach the user. A spec
   # feature that vanishes silently is how a model ends up not meaning what its
   # author thinks it means.
   for u in result.unsupported:
       print(f"  NOT SUPPORTED: {u}")
   for w in result.warnings:
       print(f"  NOTE: {w}")

   if result.model is not None:
       print("Model object constructed successfully.")
   else:
       out = pathlib.Path("./mmm-workspace/model_code")
       out.mkdir(parents=True, exist_ok=True)
       ext = "R" if result.framework == "robyn" else "py"
       path = out / f"model.{ext}"
       path.write_text(result.code)
       print(f"Generated {result.framework} code at {path}")
       print(f"Data contract: {json.dumps(result.data_contract, indent=2, default=str)}")
   EOF
   ```

4. If the framework is pymc-marketing, tell the user:
   > Model structure validated. Run `/mmm-fit` to run the prior predictive check and fit.

   For Meridian or Robyn the plugin generates code rather than fitting in-process: point the
   user at the generated file and the data contract, and say which environment it needs
   (Meridian requires TensorFlow, Robyn requires R).

5. Explain the structural choices and what they cost:
   - Which adstock and saturation were chosen, and why (delayed adstock for offline media
     whose response peaks after exposure; geometric for immediate-onset digital).
   - The shared `l_max`, set to the widest channel requirement so no carryover is truncated.
   - The Fourier order, and that seasonality and media compete for the same variance.
   - Anything in `result.unsupported` — repeat it here, do not let it scroll past.
