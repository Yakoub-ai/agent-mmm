---
description: Reconcile modelled media spend against the finance ledger, classify the shape of any disagreement, and produce the variance report needed before results are presented.
---

# MMM Reconcile

Platform exports and the general ledger always disagree. The **shape** of the disagreement
is the diagnosis.

A model whose media numbers finance does not recognise will not survive its first review
with the CFO, and it should not: if the model cannot account for the money, its ROAS is
measuring something other than the money.

## Steps

1. Get the finance extract: media cost by period, on the same basis the model will use.
   Ask explicitly whether it is gross or net of agency fees, ad serving and tax.

2. Reconcile:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-.}"
   python3 - "$PLUGIN_ROOT" <<'PY'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   import pandas as pd
   from agent_mmm.reconciliation import reconcile_spend, render_reconciliation_report

   modelled = pd.read_csv("mmm-workspace/prepared/media_wide.csv")
   finance  = pd.read_csv("data/raw/finance_media_cost.csv")          # <-- edit

   rec = reconcile_spend(
       modelled, finance,
       date_column="date",
       finance_spend_column="media_cost",                            # <-- edit
       freq="W-MON",
       tolerance=0.02,
       max_total_variance=0.05,
   )

   out = pathlib.Path("mmm-workspace/discovery"); out.mkdir(parents=True, exist_ok=True)
   (out / "reconciliation.md").write_text(render_reconciliation_report(rec), encoding="utf-8")
   (out / "reconciliation.json").write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

   print("PASS" if rec.ok else "FAIL")
   print(f"Modelled {rec.modelled_total:,.0f} vs ledger {rec.finance_total:,.0f} ({rec.total_pct:+.2%})")
   print(f"Pattern: {rec.pattern}")
   print(f"Within tolerance: {rec.n_within_tolerance}/{rec.n_periods} periods")
   for e in rec.errors:    print("ERROR:", e)
   for w in rec.warnings:  print("WARN :", w)
   for q in rec.questions: print("Q    :", q)
   PY
   ```

3. Interpret the pattern with the user:

   | Pattern | Usual cause | Next step |
   |---|---|---|
   | `aligned` | — | Get sign-off and move on |
   | `constant_shortfall` | Agency fees, ad serving, VAT in the ledger only | Decide gross or net deliberately; use the same basis in the reported ROAS |
   | `constant_excess` | Double counting, or a currency/tax basis difference | Find the campaigns counted twice |
   | `drifting` | A channel added mid-window and never mapped | Find where the drift starts; look at what launched |
   | `timing_shift` | Platform reports delivery, ledger reports invoice | Reconcile on delivery date — this one corrupts carryover directly |
   | `isolated_spikes` | Rebates, credits, make-goods | Identify each; do not smooth them away |
   | `unstructured` | The sources are not measuring the same thing | Re-check the join keys before anything else |

4. Note that `timing_shift` can pass on totals while failing on periods. That is worse for
   an MMM than a level difference, not better — the totals are right and the periods are
   wrong, which corrupts carryover estimation directly.

5. **Get sign-off before the model is built**, not when the results are presented. Record
   who signed off and on what basis in `mmm-workspace/discovery/reconciliation.md`.
