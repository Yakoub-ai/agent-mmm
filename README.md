# Yakoub AI Plugins — Claude Code Marketplace

A Claude Code plugin marketplace for data science and marketing analytics.

## Plugins

### [agent-mmm](./plugins/agent-mmm/) — Marketing Mix Model framework

Expert MMM knowledge plus an executable pipeline: a framework-agnostic project spec, data
preparation and auditing, causal specification, prior generation, model fitting,
diagnostics, and stakeholder reporting.

Works across **pymc-marketing 1.1**, **Google Meridian 1.8** and **Meta Robyn 3.12** —
pymc-marketing runs in-process, the others are generated as runnable code with their data
contracts, and the plugin reports what each framework cannot express rather than dropping
it silently.

The knowledge is organised around the things that actually make MMMs wrong: variable roles
(a price index is not a media channel and email has no ROAS), baseline health (a negative
baseline invalidates every ROAS below it), identifiability (collinear channels cannot be
separated, whatever the model reports), and calibration (an MMM without an experiment is an
argument from correlation).

**Agents** — `agent-mmm`, `mmm-modeler`, `mmm-diagnostician`, `mmm-improver`, `mmm-reporter`

**Commands** — `/mmm-intake`, `/mmm-intake-quick`, `/mmm-prepare-data`, `/mmm-analyze-data`,
`/mmm-recommend-controls`, `/mmm-recommend-priors`, `/mmm-build`, `/mmm-fit`,
`/mmm-diagnose`, `/mmm-improve`, `/mmm-report`, `/mmm-status`

**Skills** — project planning, intake, data engineering, data quality, channel semantics,
causal design, baseline and trend, model building, experimentation and calibration,
validation, diagnostics, attribution, budget optimisation, multi-geo panels, target units,
stakeholder reporting, iterative improvement, greenfield vs brownfield, external factors,
and reference guides for pymc-marketing, Meridian, Robyn and framework selection.

## Installation

### Interactive (recommended)

1. In Claude Code, run `/plugins`
2. **Marketplaces** → **+ Add Marketplace**
3. Enter `https://github.com/Yakoub-ai/agent-mmm`
4. **Discover** → **agent-mmm** → **Install**
5. Restart Claude Code

### Project-level

```json
{
  "extraKnownMarketplaces": {
    "yakoub-ai-plugins": {
      "source": { "source": "git", "url": "https://github.com/Yakoub-ai/agent-mmm" }
    }
  },
  "enabledPlugins": { "agent-mmm@yakoub-ai-plugins": true }
}
```

## Requirements

- Claude Code with plugin support
- Python 3.12+ for the `agent_mmm` library
- `pip install -e "plugins/agent-mmm[pymc]"` for the executable pymc-marketing backend
