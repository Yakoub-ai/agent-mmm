"""Environment setup and health check for agent-mmm projects.

Standard library only: this runs *before* any environment exists, using
whatever ``python3`` is on PATH, so it cannot import pandas, pydantic or
anything else from ``agent_mmm``.

Design rules:

* **Plan, then apply.** ``plan`` prints the exact commands; nothing is
  installed until ``apply --yes``.
* **No shell.** Every command is an argv list passed to ``subprocess.run``.
  No ``shell=True``, no ``curl | sh`` installers. If a tool is missing we say
  how to install it; we do not install it.
* **One environment per framework family.** Meridian pulls TensorFlow, which
  must not share a resolver with PyMC. Each gets its own venv.
* **Reproducible.** After install, the exact package set is frozen to
  ``mmm-workspace/env/<profile>.lock.txt`` and the versions to
  ``environment.json``.
* **Client data never reaches git.** The check fails if ``mmm-workspace/`` or
  the venvs are not git-ignored in the project.

Usage::

    python3 -I path/to/agent_mmm/env_setup.py check   --profile pymc
    python3 -I path/to/agent_mmm/env_setup.py plan    --profile pymc
    python3 -I path/to/agent_mmm/env_setup.py apply   --profile pymc --yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = "mmm-workspace"
GITIGNORE_ENTRIES = (f"{WORKSPACE}/", ".venv-mmm/", ".venv-meridian/")


@dataclass(frozen=True)
class Profile:
    name: str
    venv: str
    python: str                      # major.minor
    extras: tuple[str, ...]
    # import name -> (distribution name, minimum major version or None)
    packages: dict[str, tuple[str, Optional[int]]]
    note: str = ""


PROFILES: dict[str, Profile] = {
    "core": Profile(
        name="core",
        venv=".venv-mmm",
        python="3.12",
        extras=("parquet", "report"),
        packages={
            "numpy": ("numpy", None),
            "pandas": ("pandas", 2),
            "matplotlib": ("matplotlib", 3),
            "plotly": ("plotly", 5),
            "scipy": ("scipy", None),
            "statsmodels": ("statsmodels", None),
            "pydantic": ("pydantic", 2),
            "yaml": ("pyyaml", None),
            "pyarrow": ("pyarrow", None),
        },
        note="Data discovery, cleaning and audit only. No model fitting.",
    ),
    "pymc": Profile(
        name="pymc",
        venv=".venv-mmm",
        python="3.12",
        extras=("pymc", "parquet", "report"),
        packages={
            "matplotlib": ("matplotlib", 3),
            "plotly": ("plotly", 5),
            "numpy": ("numpy", None),
            "pandas": ("pandas", 2),
            "pydantic": ("pydantic", 2),
            "pymc": ("pymc", 5),
            "pymc_marketing": ("pymc-marketing", 1),
            "arviz": ("arviz", None),
            "h5netcdf": ("h5netcdf", None),
            "pyarrow": ("pyarrow", None),
        },
        note="Data work plus in-process pymc-marketing fitting. Superset of 'core'.",
    ),
    "meridian": Profile(
        name="meridian",
        venv=".venv-meridian",
        python="3.12",
        extras=("meridian", "parquet"),
        packages={
            "numpy": ("numpy", None),
            "pandas": ("pandas", 2),
            "pydantic": ("pydantic", 2),
            "meridian": ("google-meridian", 1),
            "tensorflow": ("tensorflow", 2),
            "pyarrow": ("pyarrow", None),
        },
        note="Google Meridian. Kept separate because TensorFlow conflicts with PyMC's stack.",
    ),
}


# --------------------------------------------------------------------------- #
# Tool detection
# --------------------------------------------------------------------------- #
def _run(argv: list[str], cwd: Optional[Path] = None, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def detect_tools() -> dict:
    tools: dict = {"uv": None, "pyenv": None, "pyenv_versions": [], "pythons": {}}
    uv = shutil.which("uv")
    if uv:
        rc, out = _run([uv, "--version"])
        tools["uv"] = {"path": uv, "version": out if rc == 0 else None}
    pyenv = shutil.which("pyenv")
    if pyenv:
        rc, out = _run([pyenv, "--version"])
        tools["pyenv"] = {"path": pyenv, "version": out if rc == 0 else None}
        rc, out = _run([pyenv, "versions", "--bare"])
        if rc == 0:
            tools["pyenv_versions"] = [v.strip() for v in out.splitlines() if v.strip()]
    for minor in ("3.12", "3.13", "3.11"):
        exe = shutil.which(f"python{minor}")
        if exe:
            tools["pythons"][minor] = exe
    return tools


def _pyenv_interpreter(pyenv: str, versions: list[str], want: str) -> Optional[str]:
    """Newest installed CPython ``want.*`` under pyenv, as an absolute path."""
    matches = [v for v in versions if re.fullmatch(rf"{re.escape(want)}\.\d+", v)]
    if not matches:
        return None
    best = max(matches, key=lambda v: int(v.rsplit(".", 1)[1]))
    rc, root = _run([pyenv, "root"])
    if rc != 0:
        return None
    exe = Path(root) / "versions" / best / "bin" / "python"
    return str(exe) if exe.exists() else None


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    description: str
    argv: list[str]


@dataclass
class SetupPlan:
    profile: str
    tool: str
    venv: str
    steps: list[Step] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers

    def to_markdown(self) -> str:
        lines = [f"# Setup plan — profile `{self.profile}` via `{self.tool}`", ""]
        if self.blockers:
            lines += ["## Blockers", ""] + [f"- {b}" for b in self.blockers] + [""]
        if self.steps:
            lines += ["## Commands that will run", ""]
            for i, s in enumerate(self.steps, 1):
                lines += [f"{i}. {s.description}", f"   `{' '.join(s.argv)}`"]
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def build_plan(
    profile: str,
    project_dir: Path,
    tool: str = "auto",
    plugin_root: Path = PLUGIN_ROOT,
    tools: Optional[dict] = None,
) -> SetupPlan:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    prof = PROFILES[profile]
    tools = tools if tools is not None else detect_tools()
    venv = (project_dir / prof.venv).resolve()
    vpy = _venv_python(venv)
    target = f"{plugin_root}[{','.join(prof.extras)}]"

    if tool == "auto":
        tool = "uv" if tools.get("uv") else "pyenv" if tools.get("pyenv") else "venv"
    plan = SetupPlan(profile=profile, tool=tool, venv=str(venv))
    plan.notes.append(prof.note)

    if not (plugin_root / "pyproject.toml").exists():
        plan.blockers.append(f"Plugin root {plugin_root} has no pyproject.toml.")
        return plan

    if tool == "uv":
        if not tools.get("uv"):
            plan.blockers.append(
                "uv is not installed. Install it from https://docs.astral.sh/uv/ "
                "(e.g. `pipx install uv` or your package manager), then re-run."
            )
            return plan
        uv = tools["uv"]["path"]
        if not vpy.exists():
            plan.steps.append(Step(
                f"Create {prof.venv} with Python {prof.python} (uv downloads it if missing)",
                [uv, "venv", "--python", prof.python, str(venv)],
            ))
        plan.steps.append(Step(
            f"Install agent-mmm with extras {list(prof.extras)}",
            [uv, "pip", "install", "--python", str(vpy), target],
        ))
    elif tool in ("pyenv", "venv"):
        interp: Optional[str] = None
        if tool == "pyenv":
            if not tools.get("pyenv"):
                plan.blockers.append("pyenv is not installed. Use --tool uv or --tool venv.")
                return plan
            interp = _pyenv_interpreter(tools["pyenv"]["path"], tools.get("pyenv_versions", []), prof.python)
            if interp is None:
                plan.blockers.append(
                    f"No Python {prof.python}.x under pyenv. Run "
                    f"`pyenv install {prof.python}` yourself (it compiles from source and "
                    "needs build dependencies), then re-run this plan."
                )
                return plan
        else:
            interp = tools.get("pythons", {}).get(prof.python)
            if interp is None:
                plan.blockers.append(
                    f"No python{prof.python} on PATH. Install Python {prof.python}, "
                    "or use --tool uv / --tool pyenv."
                )
                return plan
        if not vpy.exists():
            plan.steps.append(Step(f"Create {prof.venv}", [interp, "-m", "venv", str(venv)]))
        plan.steps.append(Step("Upgrade pip", [str(vpy), "-m", "pip", "install", "--upgrade", "pip"]))
        plan.steps.append(Step(
            f"Install agent-mmm with extras {list(prof.extras)}",
            [str(vpy), "-m", "pip", "install", target],
        ))
    else:
        plan.blockers.append(f"unknown tool {tool!r}; use uv, pyenv or venv")
        return plan

    if profile == "meridian":
        plan.notes.append(
            "TensorFlow is large (~1 GB). A GPU is optional; CPU sampling works but is slower."
        )
    plan.notes.append("Robyn is R-only and is not installed; the plugin generates Robyn code instead.")
    return plan


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #
def apply_plan(plan: SetupPlan, project_dir: Path, echo=print) -> dict:
    if not plan.ok:
        raise RuntimeError("plan has blockers: " + "; ".join(plan.blockers))
    for step in plan.steps:
        echo(f"→ {step.description}")
        p = subprocess.run(step.argv, cwd=project_dir)
        if p.returncode != 0:
            raise RuntimeError(f"step failed ({p.returncode}): {' '.join(step.argv)}")
    return write_lock(plan, project_dir)


def write_lock(plan: SetupPlan, project_dir: Path) -> dict:
    """Freeze the installed package set so the environment can be rebuilt exactly."""
    vpy = _venv_python(Path(plan.venv))
    if plan.tool == "uv" and shutil.which("uv"):
        rc, out = _run([shutil.which("uv"), "pip", "freeze", "--python", str(vpy)], timeout=120)
    else:
        rc, out = _run([str(vpy), "-m", "pip", "freeze"], timeout=120)
    env_dir = project_dir / WORKSPACE / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    lock = env_dir / f"{plan.profile}.lock.txt"
    if rc == 0:
        lock.write_text(out + "\n", encoding="utf-8")
    return {"lock_file": str(lock) if rc == 0 else None}


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #
_PROBE = r"""
import json, sys, importlib, importlib.metadata as md
out = {"python": sys.version.split()[0], "packages": {}}
for mod, dist in json.loads(sys.argv[1]).items():
    try:
        importlib.import_module(mod)
        try:
            out["packages"][mod] = {"ok": True, "version": md.version(dist)}
        except md.PackageNotFoundError:
            out["packages"][mod] = {"ok": True, "version": None}
    except Exception as e:
        out["packages"][mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
print(json.dumps(out))
"""


def _plugin_version(plugin_root: Path) -> Optional[str]:
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', (plugin_root / "pyproject.toml").read_text(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def _git_ignore_status(project_dir: Path) -> dict:
    git = shutil.which("git")
    if not git:
        return {"git_repo": None, "not_ignored": []}
    rc, _ = _run([git, "rev-parse", "--is-inside-work-tree"], cwd=project_dir)
    if rc != 0:
        return {"git_repo": False, "not_ignored": []}
    not_ignored = []
    for entry in GITIGNORE_ENTRIES:
        rc, _ = _run([git, "check-ignore", "-q", "--no-index", entry], cwd=project_dir)
        if rc != 0:
            not_ignored.append(entry)
    rc, tracked = _run([git, "ls-files", WORKSPACE], cwd=project_dir)
    return {
        "git_repo": True,
        "not_ignored": not_ignored,
        "tracked_workspace_files": tracked.splitlines()[:20] if rc == 0 and tracked else [],
    }


def fix_gitignore(project_dir: Path, entries: list[str]) -> Path:
    gi = project_dir / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    missing = [e for e in entries if e not in existing.splitlines()]
    if missing:
        block = ("\n" if existing and not existing.endswith("\n") else "")
        block += "\n# agent-mmm: client data and local environments\n" + "\n".join(missing) + "\n"
        gi.write_text(existing + block, encoding="utf-8")
    return gi


def health_check(profile: str, project_dir: Path, plugin_root: Path = PLUGIN_ROOT) -> dict:
    prof = PROFILES[profile]
    venv = (project_dir / prof.venv).resolve()
    vpy = _venv_python(venv)
    report: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "venv": str(venv),
        "plugin_root": str(plugin_root),
        "plugin_version": _plugin_version(plugin_root),
        "errors": [],
        "warnings": [],
        "packages": {},
    }
    err, warn = report["errors"].append, report["warnings"].append

    if not vpy.exists():
        err(f"No environment at {prof.venv}. Run the setup plan for profile '{profile}'.")
    else:
        probe = {mod: dist for mod, (dist, _) in prof.packages.items()}
        probe["agent_mmm"] = "agent-mmm"
        # -I: ignore PYTHONPATH, user site and the current directory, so a file
        # in the project cannot shadow a package during the check.
        rc, out = _run([str(vpy), "-I", "-c", _PROBE, json.dumps(probe)], cwd=project_dir, timeout=300)
        if rc != 0:
            err(f"Could not run the environment's Python: {out[:300]}")
        else:
            data = json.loads(out.splitlines()[-1])
            report["python"] = data["python"]
            report["packages"] = data["packages"]
            if not data["python"].startswith(prof.python + "."):
                warn(f"Environment runs Python {data['python']}; profile expects {prof.python}.")
            for mod, (dist, min_major) in prof.packages.items():
                p = data["packages"].get(mod, {})
                if not p.get("ok"):
                    err(f"{dist} does not import: {p.get('error', 'missing')}")
                    continue
                v = p.get("version")
                if min_major and v:
                    m = re.match(r"(\d+)", v)
                    if m and int(m.group(1)) < min_major:
                        err(f"{dist} {v} is too old; need {min_major}.x or newer.")
            installed = data["packages"].get("agent_mmm", {}).get("version")
            if installed and report["plugin_version"] and installed != report["plugin_version"]:
                warn(
                    f"Environment has agent-mmm {installed}, plugin is {report['plugin_version']}. "
                    "Commands load the plugin's own lib/, but dependencies may be stale — re-run apply."
                )

    gi = _git_ignore_status(project_dir)
    report["git"] = gi
    if gi.get("git_repo"):
        if gi["not_ignored"]:
            err(
                f"Not git-ignored: {gi['not_ignored']}. Client data or environments could be "
                "committed. Re-run with --fix-gitignore, or add them to .gitignore."
            )
        if gi.get("tracked_workspace_files"):
            err(
                f"{WORKSPACE}/ already has files tracked by git, e.g. "
                f"{gi['tracked_workspace_files'][:3]}. Ignoring does not untrack them: "
                f"`git rm -r --cached {WORKSPACE}` and check history if it held client data."
            )
    elif gi.get("git_repo") is False:
        warn("Project is not a git repository; nothing is version-controlled.")

    report["status"] = "FAIL" if report["errors"] else "WARN" if report["warnings"] else "PASS"
    env_dir = project_dir / WORKSPACE / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"environment-{profile}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render_health(report: dict) -> str:
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[report["status"]]
    lines = [
        f"# Environment check — `{report['profile']}` {icon} {report['status']}",
        "",
        f"- venv: `{report['venv']}`",
        f"- Python: {report.get('python', 'n/a')}",
        f"- Plugin version: {report.get('plugin_version')}",
        "",
    ]
    if report["packages"]:
        lines += ["| Package | Status | Version |", "|---|---|---|"]
        for mod, p in report["packages"].items():
            lines.append(f"| {mod} | {'ok' if p.get('ok') else 'FAIL'} | {p.get('version') or ''} |")
        lines.append("")
    for title, key in (("Errors", "errors"), ("Warnings", "warnings")):
        if report[key]:
            lines += [f"## {title}", ""] + [f"- {x}" for x in report[key]] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="env_setup", description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["tools", "plan", "apply", "check"])
    ap.add_argument("--profile", default="pymc", choices=sorted(PROFILES))
    ap.add_argument("--tool", default="auto", choices=["auto", "uv", "pyenv", "venv"])
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--yes", action="store_true", help="required for apply")
    ap.add_argument("--fix-gitignore", action="store_true",
                    help="append missing agent-mmm entries to the project's .gitignore")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    project = Path(a.project_dir).resolve()

    if a.action == "tools":
        print(json.dumps(detect_tools(), indent=2))
        return 0

    if a.fix_gitignore:
        gi = fix_gitignore(project, list(GITIGNORE_ENTRIES))
        print(f"Updated {gi}")

    if a.action in ("plan", "apply"):
        plan = build_plan(a.profile, project, a.tool)
        print(json.dumps(asdict(plan), indent=2) if a.json else plan.to_markdown())
        if a.action == "plan" or not plan.ok:
            return 0 if plan.ok else 2
        if not a.yes:
            print("Refusing to install without --yes. Review the commands above first.")
            return 3
        print(json.dumps(apply_plan(plan, project), indent=2))
        a.action = "check"

    report = health_check(a.profile, project)
    print(json.dumps(report, indent=2) if a.json else render_health(report))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
