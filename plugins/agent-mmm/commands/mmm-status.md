---
description: Show the status of the current MMM workspace — what's been done, what's next.
---

# MMM Workspace Status

Check the current state of `./mmm-workspace/` and report what has been completed and what remains.

## Step 1: List workspace contents

Run: `ls -la ./mmm-workspace/ 2>/dev/null || echo "No workspace found"`

## Step 2: Check spec.yaml

If `./mmm-workspace/spec.yaml` exists, read it and display:
- `company_name`
- `target_column` and `target_unit.label`
- number of channels (count entries in `channels` list)
- `mmm_type` (greenfield / brownfield)
- `data_path`

If spec.yaml does not exist, note that intake has not been run yet.

## Step 3: Check completed runs

List any completed runs in `./mmm-workspace/runs/`:

Run: `ls ./mmm-workspace/runs/ 2>/dev/null || echo "No runs found"`

For each run directory found, check if `metrics.json` is present and show the run ID alongside its key metrics (e.g. R², MAPE, LOO-IC if present).

## Step 4: Check leaderboard

If `./mmm-workspace/leaderboard.json` exists, read it and show the top 3 runs with their scores.

## Step 5: Check generated reports

Run: `ls ./mmm-workspace/reports/ 2>/dev/null || echo "No reports found"`

List any generated report files.

## Step 6: Summary

Tell the user:

> **What's done:** [list completed artifacts — spec, runs, reports, or "Nothing yet"]
>
> **What's next:** [based on what's missing, suggest the next command — see logic below]

Next-step logic:
- No spec.yaml → suggest `/mmm-intake-quick` or `/mmm-intake`
- spec.yaml exists, no runs → suggest `/mmm-analyze-data`, then `/mmm-recommend-controls`, then `/mmm-build`
- Runs exist, no leaderboard → suggest `/mmm-improve`
- Runs exist, no reports → suggest `/mmm-report cmo` or `/mmm-report ds`
- Everything present → suggest reviewing the leaderboard and sharing reports

> **Full workflow:** `/mmm-intake` → `/mmm-analyze-data` → `/mmm-recommend-controls` → `/mmm-recommend-priors` → `/mmm-build` → `/mmm-fit` → `/mmm-diagnose` → `/mmm-improve` → `/mmm-report`
