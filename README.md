# Yakoub AI Plugins — Claude Code Marketplace

A Claude Code plugin marketplace with production-ready AI agents, skills, and tools.

## Plugins

### [agent-mmm](./plugins/agent-mmm/) — Marketing Mix Model Framework

A complete MMM framework built on pymc-marketing v0.19.1+. Goes from raw CSV to stakeholder-ready reports with automated data auditing, Bayesian prior recommendation, iterative model improvement, and four stakeholder report formats.

**Agents**: `agent-mmm` (orchestrator), `mmm-modeler`, `mmm-diagnostician`, `mmm-improver`, `mmm-reporter`

**Commands**: `/mmm-intake`, `/mmm-intake-quick`, `/mmm-analyze-data`, `/mmm-recommend-controls`, `/mmm-recommend-priors`, `/mmm-build`, `/mmm-fit`, `/mmm-diagnose`, `/mmm-improve`, `/mmm-report`, `/mmm-status`

**Skills**: data quality, model building, diagnostics, attribution, budget optimization, API reference, intake questionnaire, GF/BF guide, external factors catalog, iterative improvement, stakeholder reporting, target units, multi-geo panel

## Installation

### Method 1 — Interactive (recommended)

1. Open Claude Code and type `/plugins`
2. Go to the **Marketplaces** tab → **+ Add Marketplace**
3. Enter: `https://github.com/Yakoub-ai/agent-mmm`
4. Go to **Discover** → find **agent-mmm** → **Install for you (user scope)**
5. Restart Claude Code

### Method 2 — Project-level (for teams)

Add to your project's `.claude/settings.json`:

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

## Requirements

- Claude Code with plugin support
- Python 3.11+ (for agent-mmm library)
- `pymc-marketing >= 0.19.1`
