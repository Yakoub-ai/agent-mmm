---
description: Map raw campaign names to modelling channels with auditable rules, report coverage and conflicts, and pivot a long export to one column per channel.
---

# MMM Map Channels

Campaign names are the naming convention of whoever bought the media. The model needs a
handful of channels. This mapping is where a surprising share of MMM error lives, because
it is usually done once by hand and never checked again.

**Unmapped spend above 1% is an error, not a rounding difference.**

## Steps

1. Establish the file, the campaign-name column and the spend column. `/mmm-discover`
   output names them if it has been run.

2. Propose rules and report coverage:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-.}"
   python3 - "$PLUGIN_ROOT" <<'PY'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   import pandas as pd
   from agent_mmm.taxonomy import suggest_rules, build_taxonomy, render_taxonomy_report

   FILE, NAME_COL, SPEND_COL = "data/raw/export.csv", "campaign_name", "spend"   # <-- edit

   df = pd.read_csv(FILE)
   rules  = suggest_rules(df[NAME_COL].dropna().unique())
   result = build_taxonomy(df, name_column=NAME_COL, spend_column=SPEND_COL, rules=rules)

   out = pathlib.Path("mmm-workspace/discovery"); out.mkdir(parents=True, exist_ok=True)
   (out / "taxonomy.md").write_text(render_taxonomy_report(result), encoding="utf-8")
   (out / "taxonomy.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

   print("PASS" if result.ok else "FAIL")
   print(f"Coverage: {result.coverage_rows:.1%} of rows, {result.coverage_spend:.1%} of spend")
   for e in result.errors:   print("ERROR:", e)
   for w in result.warnings: print("WARN :", w)
   print("\nChannels:")
   for ch, sp in sorted(result.channel_spend.items(), key=lambda kv: -kv[1]):
       print(f"  {ch:24s} {sp:>14,.0f}")
   if result.unmapped:
       print("\nUnmapped, by spend:")
       for u in result.unmapped[:20]:
           print(f"  {u['name'][:50]:52s} {u['spend']:>12,.0f}  ({u['rows']} rows)")
   for q in result.questions: print("\nQ:", q)
   PY
   ```

3. Show the user the proposed rules and the resulting channels. **The suggested rules are a
   draft to argue with, not an answer.**

4. Work the unmapped list **in spend order**, not alphabetically. For each, ask which
   channel it belongs to, then add a rule.

5. Check the conflicts. Where two rules claim a name, priority decided it — confirm the
   resolution is the one the user wants rather than an accident of ordering.

6. Confirm the splits that must survive:
   * **Brand vs generic search** — collapsed together, the number describes neither.
   * **Prospecting vs retargeting** — retargeting's ROAS is the most commonly overstated
     number in a media plan.
   * **PMax** stays its own channel unless a breakdown exists; say that its coefficient
     mixes search, shopping, display and video.
   * **Email and other organic** — carryover but no ROAS, because there is no media cost.

7. Once coverage passes, pivot the long export to modelling shape:

   ```python
   from agent_mmm.taxonomy import apply_rules, pivot_to_channels
   df["channel"] = apply_rules(df, rules, name_column=NAME_COL)
   wide = pivot_to_channels(df, date_column="date", channel_column="channel",
                            value_columns={"spend": "sum", "impressions": "sum"},
                            freq="W-MON")
   wide.to_csv("mmm-workspace/prepared/media_wide.csv", index=False)
   ```

   Use `sum` for spend and impressions, `mean` for rates and prices. Summing a price across
   campaigns produces a number with no meaning.

8. Save the final rules to `mmm-workspace/discovery/taxonomy_rules.json` so the next refresh
   starts from them rather than from scratch.
