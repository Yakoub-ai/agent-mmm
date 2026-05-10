---
description: Run automated data quality audit on the dataset defined in spec.yaml. Produces audit.json and audit_report.md in ./mmm-workspace/audit/.
---

# MMM Data Audit

Run the automated data quality audit using the agent_mmm library.

## Steps

1. Check that `./mmm-workspace/spec.yaml` exists. If not, tell the user: "Run `/mmm-intake` first to create the project spec."

2. Find the `agent_mmm` library. It's at the path of the installed `agent-mmm` plugin. To locate it:
   ```bash
   # The plugin is installed via Claude Code marketplace. The lib path is relative to this command file.
   # Try: find the directory containing this command file, then go up to the plugin root and into lib/
   python3 -c "
   import sys, pathlib
   # Try to locate agent_mmm via common plugin paths
   candidates = [
       pathlib.Path.home() / '.claude/plugins',
       pathlib.Path.home() / '.claude/extensions',
   ]
   for c in candidates:
       for p in c.rglob('agent_mmm/__init__.py'):
           print(p.parent.parent)
           break
   "
   ```
   
   Alternatively, the library can be imported if the user's environment has the plugin installed. Try:
   ```bash
   python3 -c "import agent_mmm; print('found')" 2>/dev/null || echo "not installed"
   ```
   
   If neither works, construct the path from the plugin location (which is where this command file lives). The command file is in `{plugin_root}/commands/mmm-analyze-data.md` and the library is at `{plugin_root}/lib/`.

3. Run the audit:
   ```bash
   python3 - <<'EOF'
   import sys, pathlib

   # Add lib/ to path — derive from this script's location by finding the plugin root
   # The plugin root contains pyproject.toml; search upward from cwd and known paths
   import os
   candidates = []
   # Check if we can find agent_mmm already
   try:
       import agent_mmm
       lib_found = True
   except ImportError:
       lib_found = False
       # Try common plugin cache locations
       home = pathlib.Path.home()
       for search_root in [home / '.claude', home / '.config/claude']:
           for p in search_root.rglob('agent_mmm/__init__.py'):
               sys.path.insert(0, str(p.parent.parent))
               lib_found = True
               break
           if lib_found:
               break

   if not lib_found:
       print("ERROR: agent_mmm library not found. Check plugin installation.")
       sys.exit(1)

   from agent_mmm.spec import load_spec
   from agent_mmm.data_audit import run_audit

   spec = load_spec("./mmm-workspace/spec.yaml")
   findings = run_audit(spec, base=".")
   
   tier = findings["summary"]["data_quality_tier"]
   print(f"\nData Quality Tier: {tier}")
   print(f"Rows: {findings['summary']['rows']} | Channels: {findings['summary']['channels']}")
   if findings['errors']:
       print(f"\nErrors ({len(findings['errors'])}):")
       for e in findings['errors']:
           print(f"  ❌ {e}")
   if findings['warnings']:
       print(f"\nWarnings ({len(findings['warnings'])}):")
       for w in findings['warnings']:
           print(f"  ⚠️  {w}")
   print(f"\nFull report: ./mmm-workspace/audit/audit_report.md")
   EOF
   ```

4. Read and display the audit report: `cat ./mmm-workspace/audit/audit_report.md`

5. Based on findings, advise:
   - If FAIL (errors): tell the user what they must fix before proceeding
   - If WARN: explain the warnings and suggest whether they're blocking or can be addressed via controls/priors
   - If PASS: "✅ Data looks good — run `/mmm-recommend-controls` next"
