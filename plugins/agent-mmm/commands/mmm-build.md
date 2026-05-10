---
description: Generate the MMM model configuration from spec.yaml and recommended priors. Validates the model structure before fitting.
---

# MMM Build

Build the MMM model configuration from your spec and priors.

## Steps

1. Check `./mmm-workspace/spec.yaml` exists.
2. Check `./mmm-workspace/priors/model_config.json` exists. If not: "Run `/mmm-recommend-priors` first."
3. Validate model config by importing the library and constructing the MMM:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   from agent_mmm.spec import load_spec
   from agent_mmm.model_factory import build_mmm, prepare_data
   import json

   spec = load_spec("./mmm-workspace/spec.yaml")
   with open("./mmm-workspace/priors/model_config.json") as f:
       raw = json.load(f)
   model_config = raw.get("model_config", raw)

   X, y = prepare_data(spec)
   print(f"Data: {len(X)} rows, {len(spec.channel_columns())} channels")
   
   try:
       model = build_mmm(spec, model_config_dict=model_config)
       print(f"✅ Model constructed successfully")
       print(f"   Adstock: {type(model.adstock).__name__}")
       print(f"   Saturation: {type(model.saturation).__name__}")
       print(f"   Channels: {spec.channel_columns()}")
       print(f"   Controls: {spec.control_columns() or 'none'}")
       print(f"   Fourier modes: {spec.seasonality.yearly_fourier_modes}")
   except Exception as e:
       print(f"❌ Model build failed: {e}")
       sys.exit(1)
   EOF
   ```

4. If successful, tell the user:
   > ✅ Model structure validated. Run `/mmm-fit` to run the prior predictive check and fit the model.

5. Explain the adstock/saturation choices made (GeometricAdstock for digital, DelayedAdstock for offline) and the yearly Fourier modes setting.
