"""Tests for env_setup.py. No network: plans are built from fake tool maps,
and the health check runs against the interpreter running the tests."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm import env_setup as es

UV = {"uv": {"path": "/x/uv", "version": "uv 0.9"}, "pyenv": None, "pyenv_versions": [], "pythons": {}}
NONE = {"uv": None, "pyenv": None, "pyenv_versions": [], "pythons": {}}


def test_module_is_stdlib_only():
    """It runs before any environment exists, on the system python."""
    tree = ast.parse(Path(es.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    third_party = roots - set(sys.stdlib_module_names) - {"__future__"}
    assert not third_party, f"env_setup must be stdlib-only, imports {third_party}"


def test_no_shell_true():
    tree = ast.parse(Path(es.__file__).read_text())
    shells = [k for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "shell"]
    assert not shells, "subprocess calls must not use shell="


def test_uv_plan_creates_venv_and_installs_extras(tmp_path):
    plan = es.build_plan("pymc", tmp_path, tools=UV)
    assert plan.ok and plan.tool == "uv"
    argvs = [s.argv for s in plan.steps]
    assert argvs[0][:4] == ["/x/uv", "venv", "--python", "3.12"]
    assert argvs[1][-1].endswith("[pymc,parquet]")
    assert all(isinstance(a, list) for a in argvs)


def test_meridian_uses_its_own_venv(tmp_path):
    pymc = es.build_plan("pymc", tmp_path, tools=UV)
    mer = es.build_plan("meridian", tmp_path, tools=UV)
    assert Path(pymc.venv).name == ".venv-mmm"
    assert Path(mer.venv).name == ".venv-meridian"


def test_missing_tool_is_a_blocker_not_an_install(tmp_path):
    plan = es.build_plan("core", tmp_path, tool="uv", tools=NONE)
    assert not plan.ok and not plan.steps
    assert "uv is not installed" in plan.blockers[0]


def test_pyenv_without_matching_python_is_blocked(tmp_path):
    tools = dict(NONE, pyenv={"path": "/x/pyenv", "version": "pyenv 2"}, pyenv_versions=["3.10.4"])
    plan = es.build_plan("core", tmp_path, tool="pyenv", tools=tools)
    assert not plan.ok and "pyenv install 3.12" in plan.blockers[0]


def test_venv_fallback_uses_system_python(tmp_path):
    tools = dict(NONE, pythons={"3.12": "/usr/bin/python3.12"})
    plan = es.build_plan("core", tmp_path, tool="auto", tools=tools)
    assert plan.tool == "venv" and plan.steps[0].argv[:3] == ["/usr/bin/python3.12", "-m", "venv"]


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(ValueError):
        es.build_plan("robyn", tmp_path, tools=UV)


def test_apply_refuses_plan_with_blockers(tmp_path):
    plan = es.build_plan("core", tmp_path, tool="uv", tools=NONE)
    with pytest.raises(RuntimeError):
        es.apply_plan(plan, tmp_path)


def test_cli_apply_requires_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "detect_tools", lambda: UV)
    assert es.main(["apply", "--profile", "core", "--project-dir", str(tmp_path)]) == 3


def _git(tmp_path, *args):
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def test_health_check_flags_unignored_workspace(tmp_path):
    _git(tmp_path, "init", "-q")
    report = es.health_check("core", tmp_path)
    assert report["status"] == "FAIL"
    assert any("Not git-ignored" in e for e in report["errors"])
    assert (tmp_path / "mmm-workspace" / "env" / "environment-core.json").exists()


def test_fix_gitignore_is_idempotent(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/")
    es.fix_gitignore(tmp_path, list(es.GITIGNORE_ENTRIES))
    es.fix_gitignore(tmp_path, list(es.GITIGNORE_ENTRIES))
    text = (tmp_path / ".gitignore").read_text()
    assert text.count("mmm-workspace/") == 1 and text.startswith("node_modules/\n")


def test_health_check_passes_with_working_env(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    es.fix_gitignore(tmp_path, list(es.GITIGNORE_ENTRIES))
    # Point the profile at the interpreter running this test.
    venv = Path(sys.executable).parent.parent
    monkeypatch.setitem(es.PROFILES, "test", es.Profile(
        name="test", venv=str(venv), python=f"{sys.version_info[0]}.{sys.version_info[1]}",
        extras=(), packages={"pandas": ("pandas", 2), "yaml": ("pyyaml", None)},
    ))
    report = es.health_check("test", tmp_path)
    assert report["packages"]["pandas"]["ok"]
    assert not [e for e in report["errors"] if "git" in e.lower()]


def test_health_check_detects_tracked_workspace(tmp_path):
    _git(tmp_path, "init", "-q")
    f = tmp_path / "mmm-workspace" / "data.csv"
    f.parent.mkdir()
    f.write_text("a")
    _git(tmp_path, "add", "mmm-workspace/data.csv")
    es.fix_gitignore(tmp_path, list(es.GITIGNORE_ENTRIES))
    report = es.health_check("core", tmp_path)
    assert any("already has files tracked" in e for e in report["errors"])
