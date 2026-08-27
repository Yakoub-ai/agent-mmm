---
name: mmm-project-plan
description: |
  Planning and running an MMM engagement end to end — scoping, stakeholder alignment, phase-by-phase workflow, timelines, gates, deliverables, refresh cadence and common failure modes. Use when starting a new MMM project, writing a project plan or proposal, deciding what to do next, estimating effort, defining what "done" means for a phase, or setting up a recurring measurement practice.
---

# Planning an MMM Engagement

An MMM project fails for organisational reasons far more often than statistical ones: the
wrong question, data that arrives in week six, a stakeholder who was never going to accept
the answer, or a model delivered without anyone deciding what it would be used for.

This is the workflow, the gates, and what to say when a phase is not ready to close.

---

## 0. Before anything: is an MMM the right instrument?

| Question | Right instrument |
|---|---|
| How should we split next year's budget across channels? | **MMM** |
| Did this specific campaign work? | Experiment / lift test |
| Which creative performs better? | A/B test |
| Which users should we target? | Attribution / uplift modelling |
| What will sales be next quarter? | Forecasting model |
| Is this channel incremental at all? | **Experiment**, then MMM to generalise |

MMM answers *allocation across channels over time*. It is bad at within-channel tactics,
bad at short horizons, and bad at anything requiring user-level resolution.

**Minimum viability:** ≥ 2 years of history (or a geo panel that substitutes for it);
genuine variation in spend; the target and spend measured consistently throughout; and at
least one stakeholder who will act on the result.

If someone needs an answer in three weeks and the data does not exist, say so in week one.

---

## 1. Phases

### Phase 1 — Scope and align (week 1)

**Do:** agree the decision the model will inform; agree the target variable; list channels
and their roles; identify who must believe the result; surface existing beliefs ("everyone
knows TV works") because those are what the model will collide with.

**Deliverable:** a one-page scope — decision, target, channels, geography, granularity,
timeline, and what is explicitly out of scope.

**Gate:** a named decision-maker has agreed the target variable in writing. Changing the
target later invalidates everything downstream, and it happens more often than it should.

**Warning sign:** nobody can say what they would do differently depending on the answer.

### Phase 2 — Data assembly (weeks 2–5; the long pole)

**Do:** collect platform exports, finance actuals, sales, price, distribution, competitive
and macro data; build the taxonomy mapping; align dates to *delivery*; reconcile spend
against finance.

**Deliverable:** a modelling dataset plus a data audit report.

**Gate:** the audit has no blocking errors, and every warning has a decision recorded
against it.

```bash
/mmm-intake            # capture the spec
/mmm-analyze-data      # run the audit
```

**Reality check:** this phase always takes longer than planned. Start data requests in
week 1, in parallel with scoping. The commonest week-6 surprise is that TV spend is only
available monthly.

### Phase 3 — Specification (week 5–6)

**Do:** draw the DAG; assign each column a role (paid / RF / organic / non-media
treatment / control); choose granularity and geography; set priors from the channel
taxonomy and any existing experiments; decide the baseline specification deliberately.

**Deliverable:** a completed spec plus a prior audit report that explains every prior.

**Gate:** the prior predictive check passes — the implied media share is plausible and the
band covers the observed range without being ten times too wide.

```bash
/mmm-recommend-controls
/mmm-recommend-priors
/mmm-build
```

**This is the phase people skip.** Specification is where causal validity is decided;
everything after it is estimation.

### Phase 4 — Fit and diagnose (week 6–7)

**Do:** fit; check convergence; check the baseline *before* looking at any ROAS; run
cross-validation; run refutation tests.

**Deliverable:** a fitted model plus a diagnostics report.

**Gate:** r-hat ≤ 1.01, no divergences, baseline never negative, CV gap < 0.20, no channel
taking an implausible share.

```bash
/mmm-fit
/mmm-diagnose
```

**Do not proceed to interpretation on a model that fails these.** A negative baseline or
unconverged chains means there are no results to interpret.

### Phase 5 — Iterate (week 7–9)

**Do:** run a tournament over structural variants; test sensitivity to the baseline
specification; group channels the data cannot separate; calibrate to any experiments.

**Deliverable:** a leaderboard and a chosen model with the reasoning recorded.

**Gate:** conclusions are stable across reasonable alternative specifications, or the
instability is documented as a range.

```bash
/mmm-improve
```

**Score on out-of-sample performance and stability, not fit.** A tournament scored on R²
selects the model that memorises best.

### Phase 6 — Interpret and report (week 9–10)

**Do:** build the decomposition narrative; compute ROAS/CPA with intervals via counterfactual
incrementality; run budget optimisation scenarios; write for each audience.

**Deliverable:** CMO, CFO, Marketing Ops and Data Science reports.

**Gate:** the recommendation is expressed as a range with a stated confidence, and the
caveats are in the deck rather than in a footnote.

```bash
/mmm-report
```

### Phase 7 — Activate and measure (ongoing)

**Do:** agree what actually changes; instrument the change so its effect is measurable;
schedule the experiments that would most reduce uncertainty; set a refresh cadence.

**Deliverable:** an activation plan and a measurement roadmap.

**Gate:** at least one budget decision has been made and one experiment scheduled.

**An MMM that changes nothing was an expensive way to confirm priors.**

---

## 2. Timeline

| Situation | Realistic |
|---|---|
| Clean data already assembled, national | 3–4 weeks |
| Typical first MMM, national | 8–12 weeks |
| First MMM, geo panel, multi-market | 12–16 weeks |
| Quarterly refresh of an existing model | 1–2 weeks |

Roughly 60% of the elapsed time is data. Any plan that allocates less is a plan to
discover the data problems late.

---

## 3. Effort allocation, and where it should go

| Activity | Typical | Better |
|---|---|---|
| Data assembly and cleaning | 50% | 45% |
| Specification and causal design | 5% | **20%** |
| Fitting and tuning | 25% | 10% |
| Validation | 5% | **15%** |
| Reporting and activation | 15% | 10% |

The commonest misallocation is hours spent on sampler tuning and model variants that would
have been better spent on the control set. Convergence problems are usually specification
problems wearing a different hat.

---

## 4. Stakeholder management

| Stakeholder | Wants | Fears | Give them |
|---|---|---|---|
| CMO | A defensible story | Being told brand does not work | Plain-language contribution, ranges, an explicit note on long-term effects |
| CFO | Efficiency and accountability | Numbers they cannot audit | Spend vs return with intervals, method transparency, reconciliation to finance |
| Channel leads | Their channel treated fairly | Being cut on a number they distrust | Per-channel saturation position, what would change the estimate |
| Data science | A model that holds up | Being handed a black box | Full diagnostics, specification, code, and known limitations |

**Manage the collision early.** If the model is likely to say brand search is
over-credited, say so in week two and explain why, so the finding lands as expected rather
than as an attack. The most common way a good MMM dies is that its first appearance is a
surprise to the person whose budget it questions.

**Two things to promise and keep:** every number comes with a range, and every caveat is
in the deck.

---

## 5. Refresh cadence

| Cadence | Suits | Watch for |
|---|---|---|
| Quarterly | Most businesses | Instability across refreshes = an unidentified model |
| Monthly | Fast-moving, high-spend | Over-reacting to noise; media plans do not turn that fast |
| Annual | Stable, slow categories | Structural changes accumulating unnoticed |

At each refresh: re-run the audit (new data brings new gaps and new campaigns); compare
conclusions against the last run and explain any large move; fold in any new experiment;
check whether the taxonomy still matches how media is bought.

**Large swings on 13 weeks of new data mean the model is not identified**, not that the
world changed. Investigate before publishing.

---

## 6. Failure modes

| Failure | Prevention |
|---|---|
| Target changed mid-project | Sign it off in Phase 1 |
| Data arrives in week 6 | Start requests in week 1; escalate on a schedule |
| Model says a favoured channel is inefficient, and is rejected | Surface the possibility early; bring an experiment, not just a model |
| Results swing between refreshes | Group collinear channels; tighten priors; calibrate |
| Beautiful model, nothing changes | Agree the activation plan before the readout |
| Media explains 90% of sales | Baseline is starved — missing price, distribution or trend |
| Everyone argues about the method | Agree validation criteria in Phase 1, before anyone sees results |
| Model becomes unmaintainable | Versioned pipeline, versioned taxonomy, fingerprinted runs |

---

## 7. Scoping questions to ask on day one

**Decision.** What will you do differently depending on the answer? Who signs it off?
When is the budget set?

**Target.** Revenue, conversions, or something else? Which system is the source of truth?
Has its definition changed in the modelling window?

**Channels.** What is bought? Who buys it? Is spend available at the granularity and
timing of delivery? Which channels are always-on?

**Confounders.** Does price move? Distribution? Did anything structural change — a
relaunch, a pandemic, a tracking migration, a competitor exiting?

**Experiments.** Has anything ever been tested? Can we run a test during the project?

**Geography.** Is geo-level data available? Can media be geo-targeted?

**History.** How many periods? Any known breaks? Any prior MMM, and what did it say?

**Politics.** Who will disagree with the answer, and what would change their mind?

---

## 8. Definition of done

An MMM engagement is complete when:

- [ ] The model converges and passes its diagnostics.
- [ ] The baseline is plausible and never negative.
- [ ] Conclusions survive reasonable alternative specifications, or the range is reported.
- [ ] At least one channel is calibrated to an experiment, or the absence is stated as the top limitation.
- [ ] Each stakeholder has a report at the right level of abstraction.
- [ ] A budget decision has been made or explicitly deferred with a reason.
- [ ] The next experiment is scheduled.
- [ ] The pipeline is reproducible by someone else from the spec and the fingerprinted data.

---

## Related skills

`mmm-intake-questionnaire` for the spec; `mmm-data-engineering` for Phase 2;
`mmm-causal-design` and `mmm-channel-semantics` for Phase 3; `mmm-diagnostics` and
`mmm-validation` for Phases 4–5; `mmm-stakeholder-reporting` for Phase 6;
`mmm-experimentation-calibration` for Phase 7.
