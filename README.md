# AgentMMM - Marketing Mix Model Expert Plugin for Claude Code

A Claude Code plugin providing an expert MMM agent and specialized skills for building, evaluating, and optimizing Marketing Mix Models with [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) v0.18.2+.

## Features

- **AgentMMM Agent** -- Deep consultation subagent for MMM tasks (model review, debugging, building, interpretation)
- **6 Specialized Skills** -- Loaded on-demand for focused guidance:
  - `mmm-data-quality` -- Data preparation, validation, quality assessment
  - `mmm-model-building` -- Model construction, prior specification, moment matching
  - `mmm-diagnostics` -- Convergence debugging, fit metrics, validation
  - `mmm-attribution` -- Channel contributions, ROAS, response curves, interpretation
  - `mmm-budget-optimization` -- Budget allocation, sensitivity analysis
  - `mmm-api-reference` -- Complete pymc-marketing API reference

## Installation

### Option 1: Clone to plugins directory

```bash
git clone https://github.com/yakoub-ai/agent-mmm.git ~/.claude/plugins/agent-mmm
```

Then add to `~/.claude/plugins/installed_plugins.json`:

```json
{
  "agent-mmm": [{
    "scope": "user",
    "installPath": "~/.claude/plugins/agent-mmm",
    "version": "1.0.0"
  }]
}
```

Restart Claude Code or run `/reload-plugins`.

### Option 2: Manual

Copy the plugin directory to `~/.claude/plugins/agent-mmm/` and register as above.

## Usage

### Agent (Deep Consultation)

The agent is automatically dispatched when you discuss MMM topics, or you can reference it directly:

- "Review my MMM model convergence diagnostics"
- "Help me build an MMM with pymc-marketing for 6 channels"
- "Interpret the channel contributions and ROAS from my model"
- "Debug the divergences I'm getting during sampling"
- "Optimize my media budget allocation"

### Skills (Auto-Activating)

Skills activate automatically when relevant context is detected (e.g., working with pymc-marketing code, discussing MMM concepts).

## Architecture

The agent has core MMM knowledge embedded (imports, scaling rules, common pitfalls) and loads specialized skills on-demand based on the task. Multiple skills can be loaded in parallel for tasks spanning multiple areas (e.g., model review needs diagnostics + attribution).

```
agent-mmm (core routing + pitfalls)
  ├── mmm-data-quality        (data prep & validation)
  ├── mmm-model-building      (priors, transforms, fitting)
  ├── mmm-diagnostics         (convergence, metrics, debugging)
  ├── mmm-attribution         (contributions, ROAS, interpretation)
  ├── mmm-budget-optimization (budget optimizer, sensitivity)
  └── mmm-api-reference       (full API signatures & methods)
```

## Grounded In

- [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) v0.18.2+ (multidimensional MMM API)
- [PyMC](https://www.pymc.io/) Bayesian modeling framework
- [ArviZ](https://arviz-devs.github.io/arviz/) diagnostics and visualization

## Requirements

- Claude Code with plugin support
- pymc-marketing v0.18.2+ installed in your Python environment (for running model code)

## License

MIT
