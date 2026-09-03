---
name: mmm-orchestration
description: |
  Running a full MMM engagement end to end — the phases, the gates between them, which specialist sub-agent handles which stage, and what must be true before each stage begins. Use when running a complete project, when asked "what do we do next", when a stage has finished and the next one has to start, or when deciding whether to delegate work to a sub-agent or handle it directly.
---

# Orchestrating an MMM Engagement

The pipeline runs discovery → intake → dataset → specification → research → model →
diagnosis → validation → interpretation → experiments → delivery. Each stage has a **gate**:
a condition that must hold before the next stage means anything.

The gates are the point. A model fitted before the dataset was agreed produces a number
somebody will act on, and the fact that the number is meaningless is invisible from the
output.

---

## The pipeline

| # | Stage | Owner | Gate before moving on |
|---|---|---|---|
| 0 | Frame the decision | you | A decision is named, with a date and a consequence |
| 1 | Discover the raw data | `mmm-data-engineer` | Every blocking question answered by the user |
| 2 | Intake and spec | you | Roles agreed per column; target and units agreed |
| 3 | Build the dataset | `mmm-data-engineer` | Taxonomy ≥99% of spend; finance reconciled and signed off |
| 4 | Audit | `mmm-data-engineer` | No blocking audit errors; tier known and accepted |
| 5 | Research | `mmm-researcher` | Findings cited, with strengths labelled |
| 6 | Causal specification | you | Every variable's role justified; no mediators as controls |
| 7 | Priors and prior predictive | `mmm-modeler` | Simulated outcomes physically possible |
| 8 | Fit | `mmm-modeler` | Sampler completed; artefacts written |
| 9 | Diagnose | `mmm-diagnostician` | Convergence clean; baseline sane |
| 10 | Validate | `mmm-diagnostician` | Out-of-sample and refutation results in hand |
| 11 | Improve | `mmm-improver` | Score plateaued, or the budget for iteration is spent |
| 12 | Interpret | you | Contributions with intervals; marginal separated from average |
| 13 | Experiment roadmap | `mmm-experiment-designer` | At least one viable, powered test identified |
| 14 | Deliver | `mmm-reporter` | Reports and decks per audience; asks have owners |

---

## The gates that matter most

**Gate 1 → 2. Blocking discovery questions are answered.** There are usually ten to fifteen
and they are not inferable: which column is the target and whether it is gross or net,
which variables are media versus business levers, whether the date is delivery or billing,
what the duplicate rows mean, whether any channel has ever been tested. Guessing at any of
these produces a model that is wrong in a way no diagnostic catches.

**Gate 3 → 4. Spend reconciles to the ledger and someone has signed it off.** Before the
model is built, not when the results are presented. A model whose media numbers finance
does not recognise will not survive its first review, and it should not.

**Gate 6 → 7. Every variable's role is justified.** A price index modelled as media, an
email programme with a ROAS, brand search credited with the demand TV created — these are
category errors, and no diagnostic catches them. This gate is a conversation, not a check.

**Gate 9 → 10. Convergence is clean and the baseline is sane.** In that order. Divergences
and high r-hat in an MMM are usually identifiability problems, not sampler settings, and
a negative or starved baseline makes every channel number downstream wrong. Stop at the
first failure and say why the rest cannot be read yet.

**Gate 12 → 14. Uncertainty travels with every number.** A contribution without an interval
is not a finding, and once a point estimate reaches a slide it is very hard to put the
interval back.

---

## Dispatch: when to delegate and when not to

**Delegate** long-running, file-heavy work with a well-defined deliverable: profiling a
folder of exports, running a fit, scoring a tournament, producing four stakeholder decks.
These fill a context window with intermediate output nobody needs to keep.

**Handle yourself** anything conversational, judgement-heavy or short: interpreting a
result, answering a question, reviewing a snippet, deciding a channel's role, choosing
between two model structures. A sub-agent starts cold and would need the context you
already hold.

**Every dispatch prompt must be self-contained** — sub-agents do not share your
conversation. Include the absolute paths to read and write, the decisions already made and
by whom, the questions the user has already answered so it does not re-ask them, and what
to return.

**Never let a sub-agent resolve an ambiguity that belongs to the user.** Instruct it to
stop and report the question. Bring the question back yourself. A guess that enters the
pipeline through a delegated task is the hardest kind to find later.

**Parallelise the independent, sequence the dependent.** Four stakeholder decks in one
message; discovery across two data drops in one message. But nothing is modelled before the
dataset is agreed, and nothing is reported before it is diagnosed.

**Check what comes back.** Read the artefact, confirm it answers what you asked, and
confirm the sub-agent did not quietly decide something it was told to escalate.

---

## When a stage fails

Failure is normal and moving backwards is the correct response. What is not acceptable is
proceeding while recording the failure in a footnote.

| Failure | Go back to |
|---|---|
| Audit blocks on data quality | Stage 1 or 3 — it is a data problem, not a modelling one |
| Prior predictive implies impossible outcomes | Stage 7, then 6 if the structure is the cause |
| Divergences, high r-hat | Stage 6 — identifiability, not sampler settings |
| Negative or starved baseline | Stage 6 — usually a missing control or a mediator included |
| Fits well, out-of-sample poor | Stage 6 — overfit, usually too many correlated channels |
| A channel takes an implausible share | Stage 6 — collinearity; group the channels and report one honest number |
| Posterior barely moved from prior | Stage 13 — the data cannot answer this; only an experiment can |

That last row is the one that most often gets skipped. A prior-dominated parameter
presented as a finding is a fabrication, however good the model is otherwise.

---

## Running the whole thing

The `/mmm-run` command walks these stages and stops at every gate. It is deliberately
interruptible: most gates are conversations, and a pipeline that runs to completion without
stopping has almost certainly guessed at something.

**Report progress against gates, not against tasks.** "Blocked at gate 3: 4% of spend is
unmapped, here are the campaign names" is useful. "Working on data preparation" is not.

---

## Where the time actually goes

| Stage | Share of effort |
|---|---|
| Discovery, dataset, audit (1-4) | 60-80% |
| Specification and priors (6-7) | 10% |
| Fitting (8) | 5% |
| Diagnosis, validation, iteration (9-11) | 10% |
| Interpretation and delivery (12-14) | 10% |

If modelling is taking most of the time, the data stage was skipped rather than completed,
and the model is being tuned to compensate for a dataset nobody agreed on.

---

## Related skills

`mmm-project-plan` for timelines, stakeholder management and refresh cadence.
`mmm-unstructured-data` for stages 1 and 3.
`mmm-intake-questionnaire` for stage 2.
`mmm-greenfield-vs-brownfield` for whether to build new or improve an existing model.
