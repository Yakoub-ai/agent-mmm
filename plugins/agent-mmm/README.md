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

### Method 1 — Interactive (recommended)

1. Open Claude Code and type `/plugins`
2. Go to the **Marketplaces** tab → **+ Add Marketplace**
3. Enter the repo URL: `https://github.com/Yakoub-ai/agent-mmm`
4. Go to the **Discover** tab → find **agent-mmm** → select **Install for you (user scope)**
5. Restart Claude Code

### Method 2 — Project-level (for teams)

Add this to your project's `.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "yakoub-ai-plugins": {
      "source": {
        "source": "git",
        "url": "https://github.com/Yakoub-ai/agent-mmm"
      }
    }
  },
  "enabledPlugins": {
    "agent-mmm@yakoub-ai-plugins": true
  }
}
```

Teammates will be prompted to install the plugin when they open the project in Claude Code.

## Usage

### Agent (Deep Consultation)

The agent is automatically dispatched when you discuss MMM topics:

- "Review my MMM model convergence diagnostics"
- "Help me build an MMM with pymc-marketing for 6 channels"
- "Interpret the channel contributions and ROAS from my model"
- "Debug the divergences I'm getting during sampling"
- "Optimize my media budget allocation"

### Skills (Auto-Activating)

Skills activate automatically when relevant context is detected (e.g., working with pymc-marketing code, discussing MMM concepts).

## Architecture

The agent has core MMM knowledge embedded (imports, scaling rules, common pitfalls) and loads specialized skills on-demand based on the task. Multiple skills can be loaded in parallel for tasks spanning multiple areas.

```
agent-mmm (core routing + pitfalls)
  ├── mmm-data-quality        (data prep & validation)
  ├── mmm-model-building      (priors, transforms, fitting)
  ├── mmm-diagnostics         (convergence, metrics, debugging)
  ├── mmm-attribution         (contributions, ROAS, interpretation)
  ├── mmm-budget-optimization (budget optimizer, sensitivity)
  └── mmm-api-reference       (full API signatures & methods)
```

## Compatibility

This plugin is **Claude Code only**. It uses Claude Code's plugin system (agents, skills, YAML frontmatter) which is not compatible with:

- **GitHub Copilot** -- Uses a different extension system (VS Code extensions / Copilot Extensions API)
- **Cursor** -- Uses its own rules/context system
- **Other AI coding tools** -- Each has its own plugin format

However, the knowledge content (markdown files in `skills/`) can be manually adapted as context files for other tools.

## Grounded In

- [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) v0.18.2+ (multidimensional MMM API)
- [PyMC](https://www.pymc.io/) Bayesian modeling framework
- [ArviZ](https://arviz-devs.github.io/arviz/) diagnostics and visualization

## Requirements

- Claude Code with plugin support
- pymc-marketing v0.18.2+ installed in your Python environment (for running model code)

## License

MIT
