"""Project scan: what is here, what has been done, what comes next.

``/mmm-run`` starts here. The scan reads the project folder and answers:

* **What kind of project is this?** Greenfield (nothing yet) or brownfield
  (existing MMM code, fitted models, Robyn outputs, a previous spec).
* **Can anything run?** Project environments, git-ignore status.
* **What data is here?** Candidate data files and their header columns.
* **How far has the pipeline got?** Each stage's artefacts plus the gate
  decisions recorded in ``mmm-workspace/progress.json``.
* **What is stale?** Outputs older than the inputs they were built from.
* **What must the user be asked?** Questions only a human can answer.

Standard library only, so it runs before any environment exists. Read-only,
except ``record`` which writes a gate decision to ``progress.json``.

Usage::

    python3 -I path/to/agent_mmm/project_scan.py scan [--json]
    python3 -I path/to/agent_mmm/project_scan.py record --stage 4 --status passed --note "..."
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Run as a script: load env_setup from this plugin's lib/, never from
    # another agent_mmm that happens to be installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_mmm import env_setup  # noqa: E402

WS = "mmm-workspace"
PROGRESS = "progress.json"
STATUSES = ("passed", "blocked", "skipped")

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "site-packages", ".tox", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".idea", ".vscode", WS,
}
MAX_FILES = 5000
MAX_DEPTH = 6
MAX_TEXT_BYTES = 1_000_000
DATA_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
CODE_EXT = {".py", ".ipynb", ".r", ".rmd", ".qmd", ".sql"}

MODEL_MARKERS = {
    "pymc-marketing": re.compile(r"\bpymc_marketing\b"),
    "meridian": re.compile(r"^\s*(from|import)\s+meridian\b", re.M),
    "robyn": re.compile(r"library\(\s*Robyn\s*\)|\brobyn_(inputs|run|outputs)\b|\bimport robyn\b"),
    "lightweight-mmm": re.compile(r"\blightweight_mmm\b"),
}
ARTIFACT_PATTERNS = {
    "fitted model (netcdf)": re.compile(r".*\.(nc|netcdf)$", re.I),
    "Robyn output": re.compile(r"(pareto_(hyperparameters|aggregated|alldecomp).*\.csv|.*robyn.*\.(rds|json))$", re.I),
    "model config": re.compile(r"^model_config\.(json|ya?ml)$", re.I),
}
_KPI = re.compile(r"^(y|kpi|target)$|revenue|sales|orders|conversions|units|transactions|signups|leads|gmv", re.I)
_MEDIA = re.compile(r"spend|cost|impr|grp|clicks|budget", re.I)
_DATE = re.compile(r"^(date|week|day|month|period|ds|time)", re.I)


# --------------------------------------------------------------------------- #
# Pipeline stages, mirroring commands/mmm-run.md
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    num: int
    title: str
    command: str
    artifacts: tuple[str, ...]    # globs relative to mmm-workspace/
    optional: bool = False


STAGES: tuple[Stage, ...] = (
    Stage(0, "Frame the decision", "(conversation)", ()),
    Stage(1, "Discover", "/mmm-discover", ("discovery/discovery.json", "discovery/answers.md")),
    Stage(2, "Intake and spec", "/mmm-intake", ("spec.yaml",)),
    Stage(3, "Build the dataset", "/mmm-map-channels, /mmm-reconcile, /mmm-prepare-data",
          ("prepared/dataset.csv", "discovery/reconciliation.md")),
    Stage(4, "Audit and improvement plan", "/mmm-analyze-data",
          ("audit/audit.json", "audit/data_quality_tips.json")),
    Stage(5, "Research", "/mmm-research", ("research/*",), optional=True),
    Stage(6, "Causal specification", "/mmm-recommend-controls", ("controls/*",)),
    Stage(7, "Priors", "/mmm-recommend-priors", ("priors/model_config.json",)),
    Stage(8, "Fit", "/mmm-fit", ("runs/*/metrics.json",)),
    Stage(9, "Diagnose", "/mmm-diagnose", ("runs/*/diagnostics_report.md",)),
    Stage(10, "Validate and compare", "/mmm-compare", ("comparison/comparison.md",)),
    Stage(11, "Improve", "/mmm-improve", ("leaderboard.json",)),
    Stage(12, "Interpret", "(conversation)", ()),
    Stage(13, "Experiment roadmap", "/mmm-experiment-plan", ("experiments/roadmap.md",)),
    Stage(14, "Deliver", "/mmm-report, /mmm-present", ("reports/*.md", "reports/mmm_report.html")),
)

# (output glob, input globs): output is stale if any input is newer.
DEPENDENCIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audit/audit.json", ("spec.yaml", "prepared/dataset.csv")),
    ("controls/*", ("spec.yaml",)),
    ("priors/model_config.json", ("spec.yaml",)),
    ("runs/*/metrics.json", ("priors/model_config.json", "spec.yaml")),
    ("reports/*.md", ("runs/*/metrics.json",)),
    ("comparison/comparison.json", ("runs/*/results.json",)),
)


@dataclass
class StageState:
    num: int
    title: str
    command: str
    status: str                       # passed | blocked | skipped | stale | unconfirmed | todo
    optional: bool = False
    artifacts_found: list[str] = field(default_factory=list)
    note: str = ""
    recorded_at: str = ""


@dataclass
class ScanResult:
    project_dir: str
    project_type: str = "greenfield"
    project_type_evidence: list[str] = field(default_factory=list)
    environments: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)
    data_files: list[dict] = field(default_factory=list)
    model_code: dict = field(default_factory=dict)
    model_artifacts: list[dict] = field(default_factory=list)
    stages: list[StageState] = field(default_factory=list)
    next_stage: Optional[int] = None
    stale: list[str] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Progress file
# --------------------------------------------------------------------------- #
def load_progress(project_dir: Path) -> dict:
    p = project_dir / WS / PROGRESS
    if not p.exists():
        return {"stages": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data.get("stages"), dict) else {"stages": {}}
    except (OSError, ValueError):
        return {"stages": {}}


def record_gate(project_dir: Path, stage: int, status: str, note: str = "") -> dict:
    """Record a gate decision. The note is the user's reason, kept verbatim."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if stage not in {s.num for s in STAGES}:
        raise ValueError(f"unknown stage {stage}")
    progress = load_progress(project_dir)
    progress["stages"][str(stage)] = {
        "status": status,
        "note": note[:2000],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = project_dir / WS / PROGRESS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    return progress


# --------------------------------------------------------------------------- #
# Walking the project
# --------------------------------------------------------------------------- #
def _walk(root: Path) -> tuple[list[Path], bool]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".venv") and d != "venv"
            and not (Path(dirpath) / d / "pyvenv.cfg").exists()
            and depth < MAX_DEPTH
        ]
        for f in filenames:
            files.append(Path(dirpath) / f)
            if len(files) >= MAX_FILES:
                return files, True
    return files, False


def _header(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            line = f.readline(4096)
    except OSError:
        return []
    sep = "\t" if path.suffix.lower() == ".tsv" else (";" if line.count(";") > line.count(",") else ",")
    return [c.strip().strip('"') for c in line.rstrip("\r\n").split(sep)][:40]


def _guess_kind(cols: list[str]) -> str:
    has_date = any(_DATE.search(c) for c in cols)
    media = sum(bool(_MEDIA.search(c)) for c in cols)
    kpi = sum(bool(_KPI.search(c)) for c in cols)
    if has_date and media and kpi:
        return "model-ready candidate (date + media + KPI)"
    if media:
        return "media / spend"
    if kpi:
        return "KPI / sales"
    return "unknown"


def _read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _mtime(project_dir: Path, pattern: str) -> Optional[float]:
    times = [p.stat().st_mtime for p in (project_dir / WS).glob(pattern) if p.is_file()]
    return max(times) if times else None


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def scan(project_dir: Path) -> ScanResult:
    project_dir = project_dir.resolve()
    res = ScanResult(project_dir=str(project_dir))
    ws = project_dir / WS

    # Environments: existence only. The full health check is /mmm-setup's job.
    for name, prof in (("pymc", env_setup.PROFILES["pymc"]), ("meridian", env_setup.PROFILES["meridian"])):
        res.environments[name] = (project_dir / prof.venv / "pyvenv.cfg").exists()
    res.environments["MMM_PYTHON"] = os.environ.get("MMM_PYTHON") or None
    res.git = env_setup._git_ignore_status(project_dir)

    files, res.truncated = _walk(project_dir)
    for p in files:
        rel = str(p.relative_to(project_dir))
        ext = p.suffix.lower()
        if ext in DATA_EXT:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            cols = _header(p)
            res.data_files.append({
                "path": rel, "size_kb": round(size / 1024, 1),
                "columns": cols, "kind": _guess_kind(cols) if cols else "unknown",
            })
        if ext in CODE_EXT:
            text = _read_text(p)
            for fw, rx in MODEL_MARKERS.items():
                if text and rx.search(text):
                    res.model_code.setdefault(fw, []).append(rel)
        for label, rx in ARTIFACT_PATTERNS.items():
            if rx.match(p.name):
                res.model_artifacts.append({"path": rel, "kind": label})

    # Project type.
    spec_text = _read_text(ws / "spec.yaml")
    if re.search(r"^mmm_type:\s*brownfield", spec_text, re.M):
        res.project_type_evidence.append("spec.yaml declares mmm_type: brownfield")
    for fw, paths in res.model_code.items():
        res.project_type_evidence.append(f"existing {fw} code: {paths[:3]}")
    for a in res.model_artifacts[:5]:
        res.project_type_evidence.append(f"{a['kind']}: {a['path']}")
    if res.project_type_evidence:
        res.project_type = "brownfield"
    elif re.search(r"^mmm_type:\s*greenfield", spec_text, re.M):
        res.project_type_evidence.append("spec.yaml declares mmm_type: greenfield")

    # Staleness: outputs older than their inputs.
    stale_outputs: set[str] = set()
    for out, ins in DEPENDENCIES:
        t_out = _mtime(project_dir, out)
        if t_out is None:
            continue
        newer = [i for i in ins if (_mtime(project_dir, i) or 0) > t_out]
        if newer:
            stale_outputs.add(out)
            res.stale.append(f"{out} is older than {newer} — re-run before relying on it")

    # Stages.
    progress = load_progress(project_dir)["stages"]
    for st in STAGES:
        found = sorted({str(p.relative_to(ws)) for g in st.artifacts for p in ws.glob(g) if p.is_file()})
        rec = progress.get(str(st.num), {})
        status = rec.get("status") if rec.get("status") in STATUSES else ("unconfirmed" if found else "todo")
        if status == "passed" and stale_outputs & set(st.artifacts):
            status = "stale"   # inputs changed after the gate was passed; redo it
        res.stages.append(StageState(
            st.num, st.title, st.command, status, st.optional,
            found[:10], rec.get("note", ""), rec.get("recorded_at", ""),
        ))
    for s in res.stages:
        if s.status in ("passed", "skipped") or (s.optional and s.status == "todo"):
            continue
        res.next_stage = s.num
        break

    res.questions = _questions(res, progress, spec_text)
    return res


def _questions(res: ScanResult, progress: dict, spec_text: str) -> list[dict]:
    q: list[dict] = []

    def ask(topic: str, question: str, why: str) -> None:
        q.append({"topic": topic, "question": question, "why": why})

    if res.project_type == "brownfield" and not re.search(r"^mmm_type:\s*brownfield", spec_text, re.M):
        ask("project", "Existing MMM work was found (see evidence). Is this a refresh of that model, "
            "a rebuild, or unrelated?",
            "A refresh reuses its priors and must explain differences from its results; a rebuild starts clean.")
    if res.project_type == "brownfield" and res.model_code:
        ask("project", "Which results from the previous model do stakeholders already believe "
            "(ROAS, contribution shares)?",
            "New results are judged against them; large differences need an explanation ready.")
    if "0" not in progress:
        ask("decision", "What decision will this model inform, by when, and what would change "
            "if the answer came back differently?",
            "It sets the precision needed, the channel granularity, and whether an MMM is the right tool.")
    if not any(res.environments.get(k) for k in ("pymc", "meridian", "MMM_PYTHON")):
        ask("environment", "No project environment yet. Set one up now with /mmm-setup "
            "(profile: core, pymc or meridian)?",
            "Every command after this runs inside it.")
    git = res.git or {}
    if git.get("git_repo") and git.get("not_ignored"):
        ask("security", f"These are not git-ignored: {git['not_ignored']}. May I add them to .gitignore?",
            "mmm-workspace/ will hold client spend and sales data.")
    if git.get("tracked_workspace_files"):
        ask("security", "Workspace files are already tracked by git. Did they contain client data?",
            "Untracking does not remove them from history.")
    has_spec = (Path(res.project_dir) / WS / "spec.yaml").exists()
    if not res.data_files and not has_spec:
        ask("data", "No data files found in the project. Where is the data (folder, warehouse export)?",
            "Nothing can be assessed without it.")
    elif len(res.data_files) > 1 and not has_spec and "1" not in progress:
        ask("data", f"Found {len(res.data_files)} data files. Which are in scope, and is any one "
            "of them already the model-ready table?",
            "Raw exports go through discovery; a ready table can skip to intake.")
    return q


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_ICON = {"passed": "✅", "skipped": "⏭️", "blocked": "⛔", "stale": "🔁", "unconfirmed": "🟡", "todo": "▫️"}


def render_scan(res: ScanResult) -> str:
    env = res.environments
    lines = [
        "# Project scan",
        "",
        f"**Project:** `{res.project_dir}` — **{res.project_type}**",
    ]
    lines += [f"- {e}" for e in res.project_type_evidence[:8]]
    lines += [
        "",
        "## Environment",
        "",
        f"- pymc env: {'yes' if env.get('pymc') else 'no'} · meridian env: "
        f"{'yes' if env.get('meridian') else 'no'} · MMM_PYTHON: {env.get('MMM_PYTHON') or 'unset'}",
    ]
    git = res.git or {}
    if git.get("git_repo"):
        lines.append(f"- git: not ignored {git.get('not_ignored') or 'none'}; "
                     f"tracked workspace files: {len(git.get('tracked_workspace_files') or [])}")
    else:
        lines.append("- git: not a repository")
    lines += ["", f"## Data files ({len(res.data_files)})", ""]
    if res.data_files:
        lines += ["| File | KB | Looks like | Columns |", "|---|---|---|---|"]
        for d in res.data_files[:25]:
            cols = ", ".join(d["columns"][:8]) + ("…" if len(d["columns"]) > 8 else "")
            lines.append(f"| {d['path']} | {d['size_kb']} | {d['kind']} | {cols} |")
    else:
        lines.append("None found.")
    if res.truncated:
        lines.append(f"\n*Stopped after {MAX_FILES} files; point the scan at a narrower folder.*")
    lines += ["", "## Pipeline progress", "", "| # | Stage | Status | Evidence / note |", "|---|---|---|---|"]
    for s in res.stages:
        label = s.status + (" (optional)" if s.optional and s.status == "todo" else "")
        ev = s.note or ", ".join(s.artifacts_found[:3])
        marker = " ← next" if s.num == res.next_stage else ""
        lines.append(f"| {s.num} | {s.title}{marker} | {_ICON[s.status]} {label} | {ev} |")
    if res.stale:
        lines += ["", "## Stale outputs", ""] + [f"- {x}" for x in res.stale]
    if res.questions:
        lines += ["", "## Questions for you", ""]
        for i, x in enumerate(res.questions, 1):
            lines += [f"{i}. **[{x['topic']}]** {x['question']}", f"   *Why:* {x['why']}"]
    if res.next_stage is not None:
        st = next(s for s in res.stages if s.num == res.next_stage)
        verb = {"unconfirmed": "Confirm the gate for", "stale": "Re-run", "blocked": "Unblock"}.get(st.status, "Resume at")
        lines += ["", f"**Next:** {verb} stage {st.num} — {st.title} (`{st.command}`)"]
    else:
        lines += ["", "**Next:** every stage is passed or skipped."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="project_scan")
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--project-dir", default=".")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("record")
    r.add_argument("--project-dir", default=".")
    r.add_argument("--stage", type=int, required=True)
    r.add_argument("--status", choices=STATUSES, required=True)
    r.add_argument("--note", default="")
    a = ap.parse_args(argv)
    project = Path(a.project_dir).resolve()

    if a.action == "record":
        record_gate(project, a.stage, a.status, a.note)
        print(f"Recorded stage {a.stage}: {a.status}")
        return 0

    res = scan(project)
    if (project / WS).is_dir():
        (project / WS / "scan.json").write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(res.to_dict(), indent=2) if a.json else render_scan(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
