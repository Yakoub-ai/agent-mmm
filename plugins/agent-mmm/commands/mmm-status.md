---
description: Show where an MMM project stands — project type, environment, data, each pipeline stage's status, stale outputs and what comes next. Read-only; /mmm-run uses the same scan to continue the work.
---

# MMM Status

1. Run the scan (read-only):
   ```bash
   python3 -I "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}/lib/agent_mmm/project_scan.py" scan
   ```

2. If `mmm-workspace/leaderboard.json` exists, show the top 3 runs and their scores. For the
   latest run in `mmm-workspace/runs/`, show the key figures from `metrics.json`.

3. Summarise in a few lines:
   > **Project:** greenfield/brownfield, and why
   > **Done:** stages passed or skipped
   > **Needs attention:** blocked or stale stages, open questions
   > **Next:** the scan's next stage and its command

4. Offer `/mmm-run` to continue from there. Do not start the next stage from this command.
