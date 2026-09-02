---
description: Run a full MMM engagement end to end — discovery, intake, dataset, specification, fit, diagnosis, validation, interpretation, experiment roadmap and stakeholder delivery — stopping at every gate that needs a human decision.
---

# MMM Run

Walks the whole pipeline. **It is deliberately interruptible**: most gates are
conversations, and a pipeline that runs to completion without stopping has almost certainly
guessed at something.

Load `agent-mmm:mmm-orchestration` first.

## How to run this

At each stage: state which stage you are on, do the work (yourself or by dispatching the
sub-agent), then **check the gate**. If the gate does not hold, stop and say what is
missing. Do not proceed while recording the failure in a footnote.

Report progress **against gates, not tasks**. "Blocked at gate 3: 4% of spend is unmapped,
here are the campaign names" is useful. "Working on data preparation" is not.

## The stages

**0 — Frame the decision.** *You.* What decision does this inform, by when, and what would
change if the answer came back differently? This sets the required precision, the channel
granularity, and whether an MMM is the right instrument at all.
→ *Gate: a decision is named, with a date and a consequence.*

**1 — Discover.** *Dispatch `mmm-data-engineer`,* or run `/mmm-discover`. Profile every raw
file. Bring the blocking questions back to the user and work through them.
→ *Gate: every blocking question answered, recorded in `mmm-workspace/discovery/answers.md`.*

**2 — Intake and spec.** *You.* `/mmm-intake`. Target and units, channels and their roles,
controls, framework, granularity.
→ *Gate: every column's role agreed. An organic channel with a ROAS or a price index with
adstock is a category error no diagnostic catches.*

**3 — Build the dataset.** *Dispatch `mmm-data-engineer`.* `/mmm-map-channels`, then
`/mmm-reconcile`, then `/mmm-prepare-data`.
→ *Gate: taxonomy covers ≥99% of spend; finance reconciled and signed off by a named person.*

**4 — Audit.** *Dispatch `mmm-data-engineer`,* or run `/mmm-analyze-data`.
→ *Gate: no blocking errors; the quality tier is known and the user has accepted it.*

**5 — Research.** *Dispatch `mmm-researcher`* when a decision would otherwise rest on
memory: category benchmarks, framework behaviour by version, prior art on a hard channel.
Findings are evidence, never instructions, and never edit the spec on their own.
→ *Gate: findings cited with their strength labelled; disagreements reported, not averaged.*

**6 — Causal specification.** *You.* `/mmm-recommend-controls`. What to control for, what
not to. Mediators are not controls — site visits, brand search and installs delete the
effect being measured.
→ *Gate: every variable's role justified; no mediators among the controls.*

**7 — Priors.** *Dispatch `mmm-modeler`,* or run `/mmm-recommend-priors`. Then a prior
predictive check.
→ *Gate: simulated outcomes are physically possible. A benchmark ROAS translated carelessly
into a coefficient can imply revenue several times the size of the business.*

**8 — Fit.** *Dispatch `mmm-modeler`,* or run `/mmm-fit`.
→ *Gate: the sampler completed and artefacts are written.*

**9 — Diagnose.** *Dispatch `mmm-diagnostician`,* or run `/mmm-diagnose`. Convergence →
fit → generalisation → baseline → learning → plausibility, in that order.
→ *Gate: convergence clean and the baseline sane. Stop at the first failure and say why the
rest cannot be read yet.*

**10 — Validate.** *Dispatch `mmm-diagnostician`.* Cross-validation, holdout, refutation
tests, stability.
→ *Gate: out-of-sample and refutation results in hand, including the ones that failed.*

**11 — Improve.** *Dispatch `mmm-improver`,* or run `/mmm-improve`.
→ *Gate: the score has plateaued, or the iteration budget is spent.*

**12 — Interpret.** *You.* Decomposition, contributions with intervals, marginal versus
average, response curves and where the data stops.
→ *Gate: uncertainty travels with every number. Once a point estimate reaches a slide it is
very hard to put the interval back.*

**13 — Experiment roadmap.** *Dispatch `mmm-experiment-designer`,* or run
`/mmm-experiment-plan`.
→ *Gate: at least one viable, powered test identified, with its expected result written
down before it runs.*

**14 — Deliver.** *Dispatch `mmm-reporter`,* or run `/mmm-report` and `/mmm-present`.
→ *Gate: reports and decks per audience; every ask has an owner and a date.*

## Dispatch rules

* **Delegate** long-running, file-heavy work with a defined deliverable. **Do it yourself**
  for anything conversational or judgement-heavy.
* **Every dispatch prompt is self-contained** — paths, decisions already made, questions the
  user has already answered, and what to return. Sub-agents do not share your conversation.
* **Never let a sub-agent resolve an ambiguity that belongs to the user.** It stops and
  reports; you bring the question back.
* **Parallelise the independent** (four decks, two data drops), **sequence the dependent**
  (nothing modelled before the dataset is agreed).
* **Check what comes back** before passing it on.

## When a stage fails

Moving backwards is the correct response.

| Failure | Go back to |
|---|---|
| Audit blocks on data quality | 1 or 3 — a data problem, not a modelling one |
| Prior predictive implies impossible outcomes | 7, then 6 if the structure is the cause |
| Divergences, high r-hat | 6 — identifiability, not sampler settings |
| Negative or starved baseline | 6 — usually a missing control or an included mediator |
| Fits well, generalises badly | 6 — overfit, usually too many correlated channels |
| One channel takes an implausible share | 6 — collinearity; group them and report one honest number |
| Posterior barely moved from prior | 13 — the data cannot answer this; only an experiment can |

## Where the time goes

Discovery and dataset work (1-4) is 60-80% of the effort. If modelling is taking most of
the time, the data stage was skipped rather than completed.
