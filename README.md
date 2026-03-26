# agent-mmm

A Claude Code plugin providing an expert MMM agent and specialized skills for building, evaluating, and optimizing Marketing Mix Models with [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) v0.18.2+.

## What's included

- **agent-mmm** — Deep consultation subagent for MMM tasks: model review, convergence debugging, building, and results interpretation
- **6 specialized skills** loaded on-demand:
  - `mmm-data-quality` — Data preparation, validation, collinearity checks
  - `mmm-model-building` — Model construction, adstock/saturation, prior specification
  - `mmm-diagnostics` — Convergence debugging, fit metrics, overfitting detection
  - `mmm-attribution` — Channel contributions, ROAS, response curves
  - `mmm-budget-optimization` — Budget allocation, BudgetOptimizer, sensitivity analysis
  - `mmm-api-reference` — Complete pymc-marketing API reference

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

The agent triggers automatically on MMM-related discussions:

- "Review my MMM model convergence diagnostics"
- "Help me build an MMM with pymc-marketing for 6 channels"
- "Debug the divergences I'm getting during sampling"
- "Interpret the channel contributions and ROAS"
- "Optimize my media budget allocation"

Skills activate automatically when pymc-marketing code or MMM concepts are detected in context.

## Requirements

- Claude Code with plugin support
- pymc-marketing v0.18.2+ in your Python environment (for running model code)

## Compatibility

Claude Code only. Not compatible with GitHub Copilot, Cursor, or other AI coding tools.

## License

MIT
