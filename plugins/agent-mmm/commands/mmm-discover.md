---
description: Profile a folder of raw, unstructured marketing data — infer date formats, grain, currency and column roles, propose joins, and produce the list of questions that must be answered before the data can be trusted. Read-only.
---

# MMM Discover

Point this at a folder of raw exports before anything else. It reads everything and
decides nothing.

**Its most valuable output is a list of questions.** Do not skip past them.

## Steps

1. Ask the user where the raw data is, if they have not said. A folder, a file, or several.

2. Run discovery:

   ```bash
   PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(dirname "$(dirname "$0")")}"
   RAW_DIR="${1:-data/raw}"

   python3 - "$PLUGIN_ROOT" "$RAW_DIR" <<'PY'
   import sys, pathlib, json
   plugin_root, raw_dir = sys.argv[1], sys.argv[2]

   lib = pathlib.Path(plugin_root) / "lib"
   if lib.exists():
       sys.path.insert(0, str(lib))
   try:
       from agent_mmm.discovery import discover, render_discovery_report
   except ImportError:
       print("ERROR: agent_mmm not importable. Expected it at", lib)
       raise SystemExit(1)

   report = discover(raw_dir)

   out = pathlib.Path("mmm-workspace/discovery")
   out.mkdir(parents=True, exist_ok=True)
   (out / "discovery.md").write_text(render_discovery_report(report), encoding="utf-8")
   (out / "discovery.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

   print(f"Files profiled: {len(report.files)}")
   print(f"Proposed grain: {report.recommended_grain}")
   if report.recommended_date_range:
       print(f"Usable window: {report.recommended_date_range[0]} -> {report.recommended_date_range[1]}")
   print(f"\nBLOCKING QUESTIONS ({len(report.blocking_questions)}):")
   for q in report.blocking_questions:
       print(f"\n  [{q['topic']}] {q['question']}")
       print(f"      why: {q['why']}")
   print(f"\nOther questions: {len(report.open_questions) - len(report.blocking_questions)}")
   for w in report.warnings:
       print(f"\nWARNING: {w}")
   print("\nFull report: mmm-workspace/discovery/discovery.md")
   PY
   ```

3. Read `mmm-workspace/discovery/discovery.md` and walk the user through what was found:
   what each file appears to be, the proposed grain and window, and where the sources do
   not line up.

4. **Ask the blocking questions.** Not as a list dumped at once — work through them, a few
   at a time, in the order they affect the work. Explain why each one matters; the report
   gives you the reason for every question.

5. Record the answers in `mmm-workspace/discovery/answers.md` so nothing is re-asked and
   the decisions are auditable later.

6. Do not proceed to `/mmm-map-channels` or `/mmm-intake` until the blocking questions are
   answered. If the user wants to move on anyway, say plainly which assumptions are being
   made on their behalf and write those into `answers.md` as assumptions rather than facts.

## Notes

* Discovery is read-only. It never modifies the source files.
* For a very large folder, narrow it: `discover(root, patterns=["**/*.csv"])`, or pass
  `sample_rows=50_000` for a fast first pass.
* Load `agent-mmm:mmm-unstructured-data` before interpreting the output.
