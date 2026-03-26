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

This plugin is distributed as a **custom marketplace**. Follow these steps:

### Step 1: Clone the marketplace

```bash
git clone https://github.com/Yakoub-ai/agent-mmm.git ~/.claude/plugins/marketplaces/yakoub-ai-plugins
```

### Step 2: Register the marketplace

Add this to `~/.claude/plugins/known_marketplaces.json` (create the file if it doesn't exist):

```json
{
  "yakoub-ai-plugins": {
    "source": {
      "source": "github",
      "repo": "Yakoub-ai/agent-mmm"
    },
    "installLocation": "~/.claude/plugins/marketplaces/yakoub-ai-plugins",
    "lastUpdated": "2026-03-26T00:00:00.000Z"
  }
}
```

> If the file already exists with other marketplaces, merge the `yakoub-ai-plugins` entry into the existing JSON object.

### Step 3: Create the plugin cache

```bash
mkdir -p ~/.claude/plugins/cache/yakoub-ai-plugins/agent-mmm/unknown
cp -r ~/.claude/plugins/marketplaces/yakoub-ai-plugins/plugins/agent-mmm/* \
      ~/.claude/plugins/cache/yakoub-ai-plugins/agent-mmm/unknown/
mkdir -p ~/.claude/plugins/cache/yakoub-ai-plugins/agent-mmm/unknown/.claude-plugin
cat > ~/.claude/plugins/cache/yakoub-ai-plugins/agent-mmm/unknown/.claude-plugin/plugin.json << 'EOF'
{
  "name": "agent-mmm",
  "description": "Marketing Mix Model expert agent and skill for pymc-marketing v0.18.2+",
  "author": { "name": "George" }
}
EOF
```

### Step 4: Enable the plugin

Add to the `enabledPlugins` section of `~/.claude/settings.json`:

```json
{
  "enabledPlugins": {
    "agent-mmm@yakoub-ai-plugins": true
  }
}
```

### Step 5: Register in installed_plugins.json

Add this entry inside the `"plugins"` object in `~/.claude/plugins/installed_plugins.json`:

```json
"agent-mmm@yakoub-ai-plugins": [
  {
    "scope": "user",
    "installPath": "~/.claude/plugins/cache/yakoub-ai-plugins/agent-mmm/unknown",
    "version": "unknown",
    "installedAt": "2026-03-26T00:00:00.000Z",
    "lastUpdated": "2026-03-26T00:00:00.000Z"
  }
]
```

> Replace `~` with your actual home directory path (e.g., `/home/youruser` or `/Users/youruser`).

### Step 6: Activate

Restart Claude Code or run `/reload-plugins`. You should see the plugin count increase by 1.

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
