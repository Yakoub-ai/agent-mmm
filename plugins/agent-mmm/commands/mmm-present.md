---
description: Build stakeholder presentation decks from the model results — slide-by-slide narrative, chart specs, speaker notes and anticipated objections, for CMO, CFO, Marketing Ops and Data Science.
---

# MMM Present

Turns model results into an **argument** for a specific room: a claim, the evidence for it,
the honest limits of that evidence, and the decision it supports.

The output is a deck spec, not a rendered file. The substance is reviewable as text and can
be argued with before anyone spends time on styling.

## Steps

1. Ask which audience, or build all four. They are genuinely different arguments from the
   same model — do not write one deck and retitle it.

2. Gather what the decks need: the diagnostics tier, the baseline share, contributions with
   intervals, the proposed reallocation, any experiments run, and the data caveats from
   discovery and reconciliation.

3. Build:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-.}"
   python3 - "$PLUGIN_ROOT" <<'PY'
   import sys, pathlib, json
   sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "lib"))
   from agent_mmm.reports.deck import build_deck, write_deck, AUDIENCES

   contributions = {                                                  # <-- from the fitted model
       "tv":           {"contribution_share": 0.31, "roas": 1.4, "roas_low": 0.6, "roas_high": 2.6},
       "search_brand": {"contribution_share": 0.24, "roas": 8.2, "roas_low": 3.0, "roas_high": 14.0},
   }
   reallocation = [
       {"channel": "tv", "current_share": 0.40, "proposed_share": 0.34, "rationale": "saturated at current spend"},
   ]

   for audience in AUDIENCES:            # cmo, cfo, mops, ds
       deck = build_deck(
           audience,
           company="...", target_label="revenue",
           tier="WARN",                  # from diagnostics
           baseline_share=0.62,
           contributions=contributions,
           reallocation=reallocation,
           experiments=[],               # [] when none have been run — this is load-bearing
           roadmap_questions=[],
           data_caveats=[],
       )
       path = write_deck(deck, base=".")
       print(f"{audience}: {len(deck.slides)} slides -> {path}")
   PY
   ```

4. Read each deck back and check the argument holds:
   * Read the **headlines alone, in order**. They should be the whole case. If they are
     not, it is a collection of slides rather than an argument.
   * Every headline asserts something. "Display returns less than it costs at current
     spend", never "Channel Performance".
   * Uncertainty is on the slide, not in an appendix.
   * Every chart has a stated takeaway.
   * The deck ends with an ask that has an owner and a date.

5. Rehearse the objections. Each slide carries the question it will draw and an answer.
   The predictable ones: *why is the baseline so large*, *which number do I put in the
   plan*, *we are contractually committed*, *why has this changed since last time*, *the
   platform says 8x ROAS*.

6. Render only once the argument is agreed. The deck spec is Markdown and converts cleanly
   to HTML, PowerPoint or Slides; keep the spec as the source of truth so the next refresh
   edits an argument rather than a slide file.

Load `agent-mmm:mmm-presentations` before writing or reviewing any of this.
