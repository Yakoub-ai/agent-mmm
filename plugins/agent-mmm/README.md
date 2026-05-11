# agent-mmm

A complete Marketing Mix Model (MMM) framework plugin for Claude Code, built on [pymc-marketing](https://github.com/pymc-labs/pymc-marketing) v0.19.1+.

## What it does

Transforms raw marketing data into fitted Bayesian MMMs with full stakeholder reporting:

1. **Interactive intake** → captures company context, target unit, channels, controls
2. **Automated data audit** → 11 quality checks, VIF, structural breaks, seasonality
3. **Controls recommender** → external factors catalog by industry + region
4. **Prior recommendation engine** → channel-type-aware Bayesian priors with moment matching
5. **Model build + fit** → pymc-marketing multidimensional MMM with MCMC
6. **Diagnostics** → convergence, overfit detection, prior-pull, plausibility checks
7. **Iterative improvement** → tournament of variants + posterior-informed prior tightening
8. **Stakeholder reports** → CMO / CFO / Marketing Ops / Data Science, target-unit-aware

## Agents

| Agent | Role |
|-------|------|
| `agent-mmm` | Orchestrator — routes tasks, loads skills, delegates to specialists |
| `mmm-modeler` | Model construction and fitting specialist |
| `mmm-diagnostician` | Convergence and validation specialist |
| `mmm-improver` | Iterative improvement and tournament specialist |
| `mmm-reporter` | Stakeholder report generation specialist |

## Slash Commands

| Command | Purpose |
|---------|---------|
| `/mmm-intake-quick` | 5-question quick intake — creates minimal `spec.yaml` |
| `/mmm-intake` | Full ~25-question intake — creates production-ready `spec.yaml` |
| `/mmm-analyze-data` | Run automated data quality audit |
| `/mmm-recommend-controls` | Recommend external factors and control columns |
| `/mmm-recommend-priors` | Generate channel-specific Bayesian priors |
| `/mmm-build` | Build MMM from spec.yaml and model_config.json |
| `/mmm-fit` | Run MCMC fitting pipeline (prior PC → fit → posterior PC) |
| `/mmm-diagnose` | Run diagnostics on a fitted model run |
| `/mmm-improve` | Launch tournament + posterior-informed refinement loop |
| `/mmm-report` | Generate all four stakeholder reports |
| `/mmm-status` | Show workspace status and run history |

## Skills

| Skill | Covers |
|-------|--------|
| `mmm-data-quality` | Data validation, completeness, collinearity, structural breaks |
| `mmm-model-building` | Adstock/saturation selection, prior specification, fitting strategy |
| `mmm-diagnostics` | Convergence thresholds, overfit detection, debugging decision tree |
| `mmm-attribution` | Channel contributions, ROAS, response curves, decomposition |
| `mmm-budget-optimization` | BudgetOptimizer API, sensitivity analysis, allocation |
| `mmm-api-reference` | Complete pymc-marketing v0.19.1+ API reference |
| `mmm-intake-questionnaire` | Intake Q&A structure, spec.yaml schema |
| `mmm-greenfield-vs-brownfield` | GF/BF decision guide, Series A/B/C funnel design |
| `mmm-external-factors-catalog` | Industry/region control catalog |
| `mmm-iterative-improvement` | Tournament mechanics, posterior-informed refinement |
| `mmm-stakeholder-reporting` | CMO/CFO/MOps/DS report templates and content guides |
| `mmm-target-units` | Non-monetary targets, CPA vs ROAS framing, value_per_unit |
| `mmm-multi-geo-panel` | Panel data format, multidimensional API, geo validation |

## Python Library

The plugin ships `agent_mmm` at `lib/agent_mmm/`. Install with:
```bash
pip install -e plugins/agent-mmm
```

Key modules:
- `agent_mmm.data_audit` — `run_audit(spec, base)`
- `agent_mmm.controls_engine` — `recommend_controls(spec, audit_findings, base)`
- `agent_mmm.prior_engine` — `recommend_priors(spec, audit_findings, base)`
- `agent_mmm.model_factory` — `build_mmm(spec, model_config_dict)`
- `agent_mmm.fit_runner` — `run_fit(spec, ..., base)`
- `agent_mmm.diagnostics` — `run_diagnostics(run_id, ..., base)`
- `agent_mmm.iter_loop` — `run_tournament(spec_path, ..., base)`
- `agent_mmm.reports.{cmo,cfo,mops,ds}` — stakeholder report generators

## Workspace Layout

All artifacts live in `./mmm-workspace/` in your project directory:

```
./mmm-workspace/
├── spec.yaml                  # single source of truth
├── audit/                     # data quality report
├── controls/                  # recommended controls
├── priors/                    # model_config.json + prior audit
├── runs/                      # one directory per model run
│   └── <run-id>/
│       ├── idata.nc           # ArviZ InferenceData (fitted model)
│       ├── metrics.json       # in-sample + CV metrics
│       └── diagnostics.json   # convergence + fit checks
├── leaderboard.json           # tournament scores
└── reports/                   # cmo.md, cfo.md, mops.md, ds.md
```

## Requirements

- Python 3.11+
- `pymc-marketing >= 0.19.1`
- `pymc-extras` (for `Prior`)
- `pydantic >= 2`
- `statsmodels`, `scipy`, `holidays`
- Claude Code with plugin support

## Workflow

```
/mmm-intake           # answer questions → spec.yaml
/mmm-analyze-data     # audit data quality
/mmm-recommend-controls  # get external factor suggestions
/mmm-recommend-priors # generate priors for your channels
/mmm-build            # build the model
/mmm-fit              # fit with MCMC
/mmm-diagnose         # check convergence and fit quality
/mmm-improve          # run tournament to find best variant
/mmm-report           # generate stakeholder reports
```
