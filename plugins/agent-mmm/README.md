# agent-mmm

A Marketing Mix Model framework for Claude Code: expert knowledge, a framework-agnostic
project spec, and an executable pipeline from raw CSV to stakeholder reports.

Verified against **pymc-marketing 1.1.0**, **google-meridian 1.8.0** and **Robyn 3.12.1**
(August 2026).

---

## What it is for

Most bad MMMs are not bad models — they are good models fitted to data that does not mean
what the modeller thinks, or specified so the coefficient is not the causal effect. This
plugin is built around that: the roles of variables, the health of the baseline, whether a
channel is identified at all, and what an experiment would resolve.

It covers the full lifecycle:

1. **Plan** — scope the decision, frame the project, set the gates
2. **Intake** — a framework-agnostic `spec.yaml` with channel roles and experiments
3. **Prepare** — aggregation, dense reindexing, explicit imputation, calendar features
4. **Audit** — contract, shape, integrity, identifiability, signal, semantics
5. **Specify** — causal design, channel semantics, baseline, priors in interpretable units
6. **Build & fit** — compile to pymc-marketing (executable), Meridian or Robyn (code)
7. **Diagnose** — convergence, baseline, generalisation, learning, plausibility
8. **Improve** — tournament over structural variants, scored on generalisation
9. **Report** — CMO, CFO, Marketing Ops and Data Science

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
| `agent-mmm` | Orchestrator and consultant across the whole lifecycle |
| `mmm-modeler` | Compiles a spec, generates priors, runs the fit pipeline |
| `mmm-diagnostician` | Full diagnostic review of a fitted run |
| `mmm-improver` | Tournament and posterior-informed refinement |
| `mmm-reporter` | Stakeholder reports |

## Commands

| Command | Does |
|---|---|
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
| `/mmm-report` | CMO / CFO / MOps / DS reports |
| `/mmm-status` | Where the project is |

## Skills

**Practice**
`mmm-project-plan` · `mmm-intake-questionnaire` · `mmm-greenfield-vs-brownfield`

**Data**
`mmm-data-engineering` · `mmm-data-quality` · `mmm-external-factors-catalog`

**Specification**
`mmm-channel-semantics` · `mmm-causal-design` · `mmm-baseline-and-trend` ·
`mmm-model-building` · `mmm-multi-geo-panel` · `mmm-target-units`

**Measurement**
`mmm-experimentation-calibration` · `mmm-validation` · `mmm-diagnostics`

**Results**
`mmm-attribution` · `mmm-budget-optimization` · `mmm-stakeholder-reporting` ·
`mmm-iterative-improvement`

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
| `spec` | Framework-agnostic project definition with channel roles and experiments |
| `data_prep` | Aggregation, dense reindexing, explicit imputation, deflation, calendar/event flags |
| `data_audit` | Six-group audit; every finding names the modelling consequence |
| `prior_engine` | Half-life priors, ROI priors, identifiability-aware widening |
| `backends` | pymc-marketing / Meridian / Robyn, each reporting what it cannot express |
| `fit_runner` | Prior PC → calibration → fit → posterior PC → save with provenance |
| `diagnostics` | Shared decomposition, baseline health, prior contraction, plausibility |
| `iter_loop` | Tournament and refinement |
| `reports` | CMO / CFO / MOps / DS |

### Workspace

```
mmm-workspace/
  spec.yaml
  data/prepared.csv, prep_report.md
  audit/audit.json, audit_report.md
  controls/, priors/model_config.json, prior_audit_report.md
  runs/<run-id>/model.nc, metrics.json, diagnostics.json, diagnostics_report.md
  leaderboard.json
  reports/cmo.md, cfo.md, mops.md, ds.md
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

145 tests, no MCMC required. Backend translation and decomposition are checked against
known ground truth.
