---
description: Clean and shape the raw dataset for MMM — aggregate to the modelling period, make the calendar dense, impute missing values with an explicit rule per column, and add calendar features. Writes a prepared dataset plus a record of every change.
---

# MMM Prepare Data

Turn a raw extract into a modelling dataset, with every transformation recorded.

The point of this command is that **every transformation is a claim about what happened**.
`fillna(0)` claims no spend occurred. Averaging when aggregating claims a variable is a
rate. Nothing here guesses on the user's behalf without saying so.

## Steps

1. Check `./mmm-workspace/spec.yaml` exists. If not: "Run `/mmm-intake` first — the
   preparation rules come from the roles declared in the spec."

2. Show the proposed imputation strategy **before** applying it, and ask the user to
   confirm or amend:

   ```bash
   python3 - <<'EOF'
   import sys, pathlib, json
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   from agent_mmm.spec import load_spec
   from agent_mmm.utils.io import load_data
   from agent_mmm.data_prep import suggest_imputation

   spec = load_spec("./mmm-workspace/spec.yaml")
   df = load_data(spec.data_path)
   for col, how in suggest_imputation(spec, df).items():
       n = int(df[col].isna().sum()) if col in df.columns else 0
       print(f"{col:30s} {how:14s} ({n} missing)")
   EOF
   ```

   Explain what each rule asserts:

   | Rule | Claim | Right for |
   |---|---|---|
   | `zero` | No activity happened | Media spend, impressions |
   | `ffill` | The last known state persisted | Price, distribution, store count |
   | `interpolate` | The series moved smoothly through the gap | Slow macro series |
   | `median` | This period was typical | Last resort — shrinks variance and biases the coefficient towards zero |
   | `drop` | The period is unusable | A missing target. Never fill a target |
   | `leave` | Let the framework decide | Rarely |

3. Run the pipeline with the confirmed strategies:

   ```bash
   python3 - <<'EOF'
   import sys, pathlib
   home = pathlib.Path.home()
   for r in [home / '.claude', home / '.config/claude']:
       for p in r.rglob('agent_mmm/__init__.py'):
           sys.path.insert(0, str(p.parent.parent)); break

   from agent_mmm.spec import load_spec
   from agent_mmm.data_prep import prepare_dataset

   spec = load_spec("./mmm-workspace/spec.yaml")
   df, report = prepare_dataset(
       spec,
       impute_strategies=None,   # replace with the confirmed dict
       aggregate=False,          # True to collapse daily rows to the modelling period
       add_calendar=True,
   )

   out = pathlib.Path("./mmm-workspace/data"); out.mkdir(parents=True, exist_ok=True)
   df.to_csv(out / "prepared.csv", index=False)
   (out / "prep_report.md").write_text(report.to_markdown())
   print(report.to_markdown())
   print(f"\nWrote {len(df)} rows to {out / 'prepared.csv'}")
   EOF
   ```

4. Read back **every** entry under "Judgement calls to confirm". These are the changes that
   could alter what the data means; do not let them scroll past.

5. Offer to point `spec.data_path` at the prepared file, then run `/mmm-analyze-data` on it.

## What this does not do

Deliberately conservative. It does **not** deflate for inflation, convert currency, drop
outliers, or apply any transformation the user did not ask for. Those are available as
`deflate`, `convert_currency` and `flag_outliers` in `agent_mmm.data_prep`, and each needs
a decision the command cannot make.

It also does not fix the problems that live upstream: spend dated by invoice rather than
delivery, a taxonomy mapping that drops campaigns, or a target whose definition changed
mid-window. `/mmm-analyze-data` will find those; only a change at source fixes them.
