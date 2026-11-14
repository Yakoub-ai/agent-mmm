"""Tests for project_scan.py: project type, progress, staleness, questions."""
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm import project_scan as ps

PLUGIN_ROOT = Path(__file__).parent.parent
WEEKLY = Path(__file__).parent / "data" / "synthetic_weekly.csv"


def _status(res, num):
    return next(s for s in res.stages if s.num == num).status


def test_module_is_stdlib_only():
    tree = ast.parse(Path(ps.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots - set(sys.stdlib_module_names) - {"__future__", "agent_mmm"}


def test_stages_match_the_run_command():
    run = (PLUGIN_ROOT / "commands" / "mmm-run.md").read_text()
    documented = [int(n) for n in re.findall(r"^\*\*(\d+) — ", run, re.M)]
    assert documented == [s.num for s in ps.STAGES]


def test_empty_folder_is_greenfield_and_asks_basics(tmp_path):
    res = ps.scan(tmp_path)
    assert res.project_type == "greenfield"
    assert res.next_stage == 0
    topics = {q["topic"] for q in res.questions}
    assert {"decision", "environment", "data"} <= topics


def test_existing_model_code_makes_it_brownfield(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "model.py").write_text("from pymc_marketing.mmm import MMM\n")
    (tmp_path / "robyn.R").write_text("library(Robyn)\nrobyn_run(InputCollect)\n")
    (tmp_path / "idata.nc").write_bytes(b"\x89HDF")
    res = ps.scan(tmp_path)
    assert res.project_type == "brownfield"
    assert set(res.model_code) == {"pymc-marketing", "robyn"}
    assert any(a["kind"] == "fitted model (netcdf)" for a in res.model_artifacts)
    assert any("refresh" in q["question"] for q in res.questions)


def test_venvs_and_workspace_are_not_scanned(tmp_path):
    venv = tmp_path / ".venv-mmm" / "lib"
    venv.mkdir(parents=True)
    (tmp_path / ".venv-mmm" / "pyvenv.cfg").write_text("")
    (venv / "x.py").write_text("import pymc_marketing\n")
    other = tmp_path / "env2"
    other.mkdir()
    (other / "pyvenv.cfg").write_text("")
    (other / "y.py").write_text("import pymc_marketing\n")
    res = ps.scan(tmp_path)
    assert res.project_type == "greenfield"
    assert res.environments["pymc"] is True


def test_data_files_get_headers_and_a_guess(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mmm.csv").write_text(WEEKLY.read_text())
    (tmp_path / "data" / "meta.csv").write_text("campaign;spend;impressions\nx;1;2\n")
    res = ps.scan(tmp_path)
    by_name = {Path(d["path"]).name: d for d in res.data_files}
    assert by_name["mmm.csv"]["kind"].startswith("model-ready")
    assert by_name["meta.csv"]["columns"] == ["campaign", "spend", "impressions"]
    assert any(q["topic"] == "data" and "Which are in scope" in q["question"] for q in res.questions)


def test_artifacts_without_a_recorded_gate_are_unconfirmed(tmp_path):
    ws = tmp_path / "mmm-workspace"
    ws.mkdir()
    (ws / "spec.yaml").write_text("mmm_type: greenfield\n")
    assert _status(ps.scan(tmp_path), 2) == "unconfirmed"


def test_record_and_resume(tmp_path):
    for n, st in ((0, "passed"), (1, "skipped"), (2, "passed"), (3, "skipped")):
        ps.record_gate(tmp_path, n, st, "agreed with user")
    res = ps.scan(tmp_path)
    assert res.next_stage == 4
    assert next(s for s in res.stages if s.num == 0).note == "agreed with user"


def test_optional_research_does_not_block(tmp_path):
    for n in range(5):
        ps.record_gate(tmp_path, n, "passed")
    assert ps.scan(tmp_path).next_stage == 6


def test_blocked_gate_stops_progress(tmp_path):
    ps.record_gate(tmp_path, 0, "passed")
    ps.record_gate(tmp_path, 1, "blocked", "waiting on finance export")
    assert ps.scan(tmp_path).next_stage == 1


def test_passed_gate_reopens_when_inputs_change(tmp_path):
    ws = tmp_path / "mmm-workspace"
    (ws / "audit").mkdir(parents=True)
    audit = ws / "audit" / "audit.json"
    spec = ws / "spec.yaml"
    audit.write_text("{}")
    spec.write_text("mmm_type: greenfield\n")
    old = time.time() - 100
    os.utime(audit, (old, old))
    for n in range(4):
        ps.record_gate(tmp_path, n, "passed")
    ps.record_gate(tmp_path, 4, "passed")
    res = ps.scan(tmp_path)
    assert _status(res, 4) == "stale"
    assert res.next_stage == 4
    assert res.stale and "audit/audit.json" in res.stale[0]


def test_record_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        ps.record_gate(tmp_path, 4, "done")
    with pytest.raises(ValueError):
        ps.record_gate(tmp_path, 99, "passed")


def test_corrupt_progress_file_is_ignored(tmp_path):
    (tmp_path / "mmm-workspace").mkdir()
    (tmp_path / "mmm-workspace" / "progress.json").write_text("{not json")
    assert ps.scan(tmp_path).next_stage == 0


def test_unignored_workspace_raises_a_security_question(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    res = ps.scan(tmp_path)
    assert any(q["topic"] == "security" for q in res.questions)


def test_cli_scan_runs_as_isolated_script(tmp_path):
    (tmp_path / "mmm-workspace").mkdir()
    out = subprocess.run(
        [sys.executable, "-I", ps.__file__, "scan", "--project-dir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "# Project scan" in out.stdout and "**Next:**" in out.stdout
    data = json.loads((tmp_path / "mmm-workspace" / "scan.json").read_text())
    assert data["next_stage"] == 0


def test_scan_does_not_create_a_workspace(tmp_path):
    ps.main(["scan", "--project-dir", str(tmp_path)])
    assert not (tmp_path / "mmm-workspace").exists()
