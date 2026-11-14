---
description: Set up or verify the Python environment for an MMM project (uv, pyenv or venv). Shows the exact install commands before running anything, freezes a lock file, and checks that client data is git-ignored.
---

# MMM Setup

Every other command runs in the project environment this creates. Run it first, in both
greenfield projects (empty folder) and brownfield projects (existing repo or existing model).

## Profiles

| Profile | venv | For |
|---|---|---|
| `core` | `.venv-mmm` | Discovery, cleaning, audit, data-quality tips. Fast install. |
| `pymc` | `.venv-mmm` | `core` plus pymc-marketing fitting. Default. |
| `meridian` | `.venv-meridian` | Google Meridian (TensorFlow). Separate venv because its stack conflicts with PyMC. |

Robyn is R-only. It is not installed; `/mmm-build` generates Robyn code instead.

## Steps

1. **Check what exists.** Look for `.venv-mmm/`, `.venv-meridian/`, `mmm-workspace/`,
   `pyproject.toml`, `requirements*.txt` and any existing model code. If the project
   already has its own environment (brownfield), tell the user and ask whether to reuse it
   (`MMM_PYTHON=/path/to/python` makes every command use it) or create a separate one. Do
   not install into an environment you did not create.

2. **Ask which profile** (default `pymc`) and which tool (default: `uv` if installed, else
   `pyenv`, else system `venv`).

3. **Show the plan.** Nothing is installed by this step:
   ```bash
   python3 -I "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}/lib/agent_mmm/env_setup.py" plan --profile PROFILE --tool TOOL
   ```
   Show the user the commands verbatim. If there are **blockers** (no uv, no matching Python
   under pyenv), explain them and stop. Never install uv, pyenv or Python for the user with
   a downloaded script — tell them the command and let them run it.

4. **Apply only after the user confirms:**
   ```bash
   python3 -I "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}/lib/agent_mmm/env_setup.py" apply --profile PROFILE --tool TOOL --yes
   ```
   This installs, writes `mmm-workspace/env/PROFILE.lock.txt`, then runs the health check.

5. **Read the health check.** It is also available alone:
   ```bash
   python3 -I "${CLAUDE_PLUGIN_ROOT:?agent-mmm plugin root is not set}/lib/agent_mmm/env_setup.py" check --profile PROFILE
   ```
   - **Not git-ignored** → explain that `mmm-workspace/` will hold client spend and sales
     data. Ask before changing their `.gitignore`; if they agree, re-run `check` with
     `--fix-gitignore`.
   - **Workspace files already tracked** → the data may already be in git history. Tell the
     user plainly; untracking does not remove history.
   - **Version warnings** → re-run `apply`.

6. **Finish** with what was installed, where the lock file is, and the next step:
   `/mmm-discover` for a folder of raw exports, or `/mmm-intake` to write the spec.

## Rules

- Never put credentials in `.env` files inside the project, in the spec, or in command
  arguments. Data-warehouse credentials belong in the user's own secret store or
  environment variables.
- Commit the lock file only if the user wants a reproducible environment in git; it
  contains package names and versions, no data.
