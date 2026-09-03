# agent-mmm

A Marketing Mix Model framework for Claude Code: expert knowledge, a framework-agnostic
project spec, and an executable pipeline from a folder of raw exports to stakeholder
decks — with an orchestrator that dispatches seven specialist sub-agents and stops at
every gate that needs a human decision.

Verified against **pymc-marketing 1.1.0**, **google-meridian 1.8.0** and **Robyn 3.12.1**
(August 2026).

---

## What it is for

Most bad MMMs are not bad models — they are good models fitted to data that does not mean
what the modeller thinks, or specified so the coefficient is not the causal effect. This
plugin is built around that: the roles of variables, the health of the baseline, whether a
channel is identified at all, and what an experiment would resolve.

It covers the full lifecycle:

1. **Discover** — profile a folder of unknown exports and generate the questions that must
   be answered before any of it can be trusted
2. **Plan** — scope the decision, frame the project, set the gates
3. **Intake** — a framework-agnostic `spec.yaml` with channel roles and experiments
4. **Build the dataset** — campaign-name taxonomy, finance reconciliation, aggregation,
   dense reindexing, explicit imputation
5. **Audit** — contract, shape, integrity, identifiability, signal, semantics
6. **Research** — category benchmarks and framework behaviour, treated as untrusted evidence
7. **Specify** — causal design, channel semantics, baseline, priors in interpretable units
8. **Build & fit** — compile to pymc-marketing (executable), Meridian or Robyn (code)
9. **Diagnose** — convergence, baseline, generalisation, learning, plausibility
10. **Improve** — tournament over structural variants, scored on generalisation
11. **Experiment** — power analysis, prioritisation, and a sequenced roadmap with
    pre-registered expectations
12. **Deliver** — CMO, CFO, Marketing Ops and Data Science reports and presentation decks

The whole sequence runs with `/mmm-run`, which stops at every gate.

### What it will not do

It will not guess past a question only a human can answer — whether a date marks delivery
or billing, whether revenue is gross or net, whether a variable is media or a business
lever. Those produce a model that is wrong in a way no diagnostic catches, so the pipeline
stops and asks. On a raw data drop the questions *are* the deliverable, and there are
usually ten to fifteen blocking ones.

### Example: starting from a folder nobody can explain

```
/mmm-discover data/raw
```

```
Files profiled: 6
Proposed grain: weekly
Usable window: 2022-06-05 -> 2023-12-01

BLOCKING QUESTIONS (12):

  [target] Several columns could be the target (finance.csv:Net Revenue,
      finance.csv:Orders, google_ads.csv:conversions). Which one is the decision
      made on, and is it gross or net of returns, tax and discounts?
      why: Modelling gross revenue and reporting net ROAS overstates every
           channel by the return rate.

  [grain] 'tv_plan_monthly.csv' is monthly while the rest of the data supports
      weekly. Can it be re-exported at weekly, should we model at the coarser
      grain, or do we have a schedule to allocate it down?
      why: Splitting monthly totals evenly across weeks invents variation the
           model will read as real, and modelling everything monthly costs most
           of the observations.

  [coverage] The usable window is only 78 weekly periods. Can history be
      extended, or should we model fewer channels, or move to a geo panel?
      why: Below roughly two years the model cannot separate an annual seasonal
           cycle from a channel that happens to be seasonal.
  ...
```

Nothing is written, nothing is decided, and the twelfth question is always whether any
channel has ever been tested.

---

## Install

```
/plugins → Marketplaces → + Add Marketplace → https://github.com/Yakoub-ai/agent-mmm
Discover → agent-mmm → Install
```

For the Python library:

```bash
pip install -e "plugins/agent-mmm[pymc]"     # executable pymc-marketing backend
pip install -e "plugins/agent-mmm"           # spec, audit, prep and code generation only
```

Requires Python ≥ 3.12 (pymc-marketing 1.x does).

---

## Agents

| Agent | Role |
|---|---|
| `agent-mmm` | Orchestrator and consultant across the whole lifecycle; dispatches the rest |
| `mmm-data-engineer` | Raw file discovery, taxonomy mapping, reconciliation, dataset assembly |
| `mmm-researcher` | Benchmarks, methodology and framework docs — read-only by design |
| `mmm-experiment-designer` | Power analysis, prioritisation, the experiment roadmap |
| `mmm-modeler` | Compiles a spec, generates priors, runs the fit pipeline |
| `mmm-diagnostician` | Full diagnostic review of a fitted run |
| `mmm-improver` | Tournament and posterior-informed refinement |
| `mmm-reporter` | Stakeholder reports and presentation decks |

`mmm-researcher` has `WebSearch` and `WebFetch` but no `Write`, `Edit` or `Bash`. It
processes content from the open web, and that content must never be able to reach a file or
a shell in the project. Retrieved content is treated as evidence, never as instruction: a
page containing embedded instructions is reported as a finding rather than obeyed.

## Commands

| Command | Does |
|---|---|
| `/mmm-run` | The whole pipeline, stopping at every gate |
| `/mmm-discover` | Profile a folder of raw exports; produce the blocking questions |
| `/mmm-map-channels` | Campaign names → channels, with coverage and conflict reporting |
| `/mmm-reconcile` | Modelled spend vs the finance ledger, with the variance pattern named |
| `/mmm-intake` | Full intake → `spec.yaml` |
| `/mmm-intake-quick` | Six questions, enough to run the audit |
| `/mmm-prepare-data` | Aggregate, reindex, impute with explicit rules, add calendar features |
| `/mmm-analyze-data` | Data audit with modelling consequences |
| `/mmm-recommend-controls` | External-factor recommendations |
| `/mmm-recommend-priors` | Half-life priors, ROI priors, per-channel structure |
| `/mmm-build` | Compile the spec into the target framework |
| `/mmm-fit` | Prior PC → calibration → MCMC → posterior PC → save |
| `/mmm-diagnose` | Convergence, baseline, generalisation, plausibility |
| `/mmm-improve` | Tournament + refinement loop |
| `/mmm-experiment-plan` | Power analysis and the sequenced experiment roadmap |
| `/mmm-report` | CMO / CFO / MOps / DS reports |
| `/mmm-present` | Stakeholder decks: narrative, charts, speaker notes, objections |
| `/mmm-research` | Benchmarks and methodology, under the untrusted-content rules |
| `/mmm-status` | Where the project is |

## Skills

**Practice**
`mmm-orchestration` · `mmm-project-plan` · `mmm-intake-questionnaire` ·
`mmm-greenfield-vs-brownfield` · `mmm-research`

**Data**
`mmm-unstructured-data` · `mmm-data-engineering` · `mmm-data-quality` ·
`mmm-external-factors-catalog`

**Specification**
`mmm-channel-semantics` · `mmm-causal-design` · `mmm-baseline-and-trend` ·
`mmm-model-building` · `mmm-multi-geo-panel` · `mmm-target-units`

**Measurement**
`mmm-experiment-roadmap` · `mmm-experimentation-calibration` · `mmm-validation` ·
`mmm-diagnostics`

**Results**
`mmm-attribution` · `mmm-budget-optimization` · `mmm-stakeholder-reporting` ·
`mmm-presentations` · `mmm-iterative-improvement`

**Frameworks**
`mmm-api-reference` (pymc-marketing) · `mmm-meridian` · `mmm-robyn` ·
`mmm-framework-selection`

---

## Frameworks

One spec, three targets. `agent_mmm.backends` reports what each framework **cannot**
express, so a feature never vanishes silently.

| | pymc-marketing 1.1 | Meridian 1.8 | Robyn 3.12 |
|---|---|---|---|
| Capability here | **executable** | code generation | code generation |
| Paradigm | Bayesian | Bayesian | Frequentist (ridge) |
| Geo hierarchy | `dims=("geo",)` | native, population-weighted | none |
| Media prior | coefficient | **ROI / mROI / contribution** | bounds + signs |
| Reach & frequency | no | **native + optimal frequency** | no |
| Log link | **yes** | no | no |
| DAG / mediation | **yes** | no | no |
| Experiment calibration | **likelihood term** | ROI prior | third objective |

See `mmm-framework-selection`.

---

## Python library

```python
from agent_mmm.spec import load_spec
from agent_mmm.data_prep import prepare_dataset
from agent_mmm.data_audit import run_audit
from agent_mmm.prior_engine import recommend_priors
from agent_mmm.model_factory import compile_spec
from agent_mmm.fit_runner import run_fit, SAMPLER_FINAL
from agent_mmm.diagnostics import run_diagnostics, decompose

spec           = load_spec("mmm-workspace/spec.yaml")
clean, prep    = prepare_dataset(spec)
audit          = run_audit(spec, df=clean)
priors         = recommend_priors(spec, audit)
compiled       = compile_spec(spec, priors=priors)   # .model .code .unsupported
metrics        = run_fit(spec, priors=priors, sampler_config=SAMPLER_FINAL)
findings       = run_diagnostics(metrics["run_id"], idata_path=metrics["model_path"])
```

| Module | Purpose |
|---|---|
| `discovery` | Profiles unknown files: date format, grain, shape, column roles, joins, and the questions that must be answered |
| `taxonomy` | Campaign names → channels, with rule provenance, coverage and conflict reporting |
| `reconciliation` | Modelled spend vs the finance ledger, classifying the *shape* of the disagreement |
| `experiments` | Power analysis for geo and time-based holdouts, prioritisation, the roadmap |
| `spec` | Framework-agnostic project definition with channel roles and experiments |
| `data_prep` | Aggregation, dense reindexing, explicit imputation, deflation, calendar/event flags |
| `data_audit` | Six-group audit; every finding names the modelling consequence |
| `prior_engine` | Half-life priors, ROI priors, identifiability-aware widening |
| `backends` | pymc-marketing / Meridian / Robyn, each reporting what it cannot express |
| `fit_runner` | Prior PC → calibration → fit → posterior PC → save with provenance |
| `diagnostics` | Shared decomposition, baseline health, prior contraction, plausibility |
| `iter_loop` | Tournament and refinement |
| `reports` | CMO / CFO / MOps / DS reports, and `reports.deck` for presentation specs |

### Workspace

```
mmm-workspace/
  discovery/discovery.md, taxonomy.md, reconciliation.md, answers.md
  spec.yaml
  data/prepared.csv, prep_report.md
  audit/audit.json, audit_report.md
  controls/, priors/model_config.json, prior_audit_report.md
  runs/<run-id>/model.nc, metrics.json, diagnostics.json, diagnostics_report.md
  leaderboard.json
  experiments/roadmap.md
  reports/cmo.md, cfo.md, mops.md, ds.md
  reports/deck_cmo.md, deck_cfo.md, deck_mops.md, deck_ds.md
```

---

## Notes on pymc-marketing 1.x

1.0 was a breaking release. If you learned the library before August 2026:

* `from pymc_marketing.mmm import MMM` — `multidimensional` is deprecated, the legacy MMM
  class is removed
* `idata` is an `xarray.DataTree`, not `arviz.InferenceData`
* `az.waic` no longer exists — use `az.loo`; `az.summary` takes `ci_prob=`
* Budget optimisation is `mmm.budget_optimizer(start, end)`
* `dims` is a tuple: `dims=("geo",)`
* Saving needs `h5netcdf` or `netCDF4`

## Tests

```bash
cd plugins/agent-mmm && python -m pytest tests/ -q
```

467 tests, no MCMC required. Backend translation and decomposition are checked against known ground
truth; power calculations are checked against standard normal tables; discovery is checked
against a fixture set of deliberately messy exports (day-first dates, currency-as-text,
duplicate rows, mismatched grains, a geo panel joined to national files).

The plugin's own prompts are tested too — `tests/test_plugin_components.py` checks that the
orchestrator can actually dispatch the sub-agents it documents, that the researcher has no
write or shell access, that every referenced skill and command exists, and that documented
library imports resolve. Both manifests are validated against the official schemas.
