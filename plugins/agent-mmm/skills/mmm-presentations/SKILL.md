---
name: mmm-presentations
description: |
  Building stakeholder presentations from MMM results — deck structure per audience, headline writing, chart selection, speaker notes, and preparing for the questions a room will actually ask. Use when preparing to present MMM results to a CMO, CFO, marketing ops, an agency or a technical reviewer, when asked for a deck or a readout, or when a set of results needs to become a decision.
---

# Presenting MMM Results

A deck fails in one of two ways. Either it is a data dump — every chart the modeller found
interesting, in the order they were produced — or it is a conclusion with the reasoning
removed, which the audience correctly refuses to act on.

What travels is an **argument**: a claim, the evidence for it, the honest limits of that
evidence, and the decision it supports.

---

## 1. Build the argument before the slides

```python
from agent_mmm.reports.deck import build_deck, write_deck, render_deck_markdown, AUDIENCES

deck = build_deck("cmo", company=..., tier=..., baseline_share=...,
                  contributions=..., reallocation=..., experiments=...)
write_deck(deck, base=".")        # mmm-workspace/reports/deck_cmo.md
```

The output is a **spec, not a rendered file**. Slide by slide: a headline, the supporting
points, a chart specification with its takeaway, speaker notes, and the question the slide
will draw with its answer.

This is deliberate. The substance is reviewable as text and can be argued with before
anyone spends time on styling, and styling a deck whose argument is wrong is the most
expensive way to discover that it is wrong. Render to HTML, PowerPoint or Slides afterwards.

---

## 2. Headlines assert something

> "Display returns less than it costs at current spend"

not

> "Channel Performance"

A category label makes the audience find the point themselves, and they will find a
different one from the one you intended. If the headline cannot be a sentence with a verb,
the slide has no point and should be cut.

Read the headlines alone, in order. They should be the argument in full. If they are not,
the deck is a collection of slides rather than a case.

---

## 3. Four audiences, four arguments

The same model produces four different decks. Do not write one and retitle it.

| | CMO | CFO | Marketing Ops | Data Science |
|---|---|---|---|---|
| **The argument** | Where the next pound goes | What was returned on what was spent | What changes in the plan | How it was built and where it is weak |
| **They want** | A defensible reallocation | A number for the plan, with its error bar | Executable line items | Enough detail to attack it |
| **They fear** | Being told their channel does not work, publicly | A number that moves next quarter | A change they cannot execute | A structural error found after socialising |
| **Register** | Plain language | Reconciled, ranges not points | Concrete, names amounts | Technical, leads with weakness |
| **Never say** | Adstock, posterior, r-hat | "Roughly", a point estimate alone | "Optimise the mix" | "It converged fine" |

---

## 4. The slides that earn their place

**The decomposition, first, always.** The baseline is everything that would have happened
with no advertising — brand equity, distribution, price, season, existing demand. Label it
explicitly, because audiences assume the whole bar is media. Every downstream disagreement
traces back to someone assuming the baseline is zero.

**Contribution and efficiency, separately.** They get confused constantly. A big channel is
not necessarily a good one; a small one is not necessarily bad. Say it out loud.

**Uncertainty, on the slide.** A channel whose interval spans a factor of three has not
been measured. This is the most valuable slide in the deck and the one most often cut for
time — do not cut it. Presenting a wide interval as a point estimate is how a model loses
its credibility six months later when the number moves.

**Marginal versus average, for anyone making a budget decision.** Average return answers
"was this worth doing", a question about the past. Marginal return is what the next pound
produces; it is always lower because channels saturate, and it is the only one that informs
the next budget. Reallocating on average return systematically overfunds saturated
channels.

**Response curves with current spend marked, and the extrapolation shaded.** Beyond the
observed spend range the model is guessing, and the shading is what stops an optimiser
result being read as a promise.

**What the model cannot tell you.** Correlated channels whose split is an assumption;
long-term brand effects outside the window; spend levels never observed. Audiences trust
models that declare their limits and distrust ones that do not. This slide buys credibility
for everything before it.

**The experiment roadmap.** If no test has been run, this is not optional and it is not a
vague "we should test more" — bring the specific first test, its cost, its duration, and
what result would change the recommendation.

**An ask with owners and dates.** A deck that closes without owners produces another
meeting instead of a decision.

---

## 5. Charts

| Question | Chart | Trap |
|---|---|---|
| Where does the outcome come from? | Waterfall | Omitting or under-labelling the baseline |
| Which channel is biggest? | Bar, ordered by contribution | Alphabetical ordering |
| How certain are we? | Interval plot | Plotting the midpoint alone |
| Where does spend stop working? | Response curve, current spend marked | Not shading beyond observed spend |
| Does it predict? | Actual vs predicted, out of sample | Showing in-sample fit and calling it validation |
| What should change? | Current vs proposed, paired bars | Showing the proposal without the current |

Every chart needs a stated takeaway. A chart nobody can state the point of does not belong
in the deck.

---

## 6. Prepare for the questions

They are predictable. Being ready is the difference between a decision and a follow-up
meeting.

**"Why is the baseline so large? What are we paying the agency for?"** The baseline is the
business you have built, not wasted spend. Media's job is the increment on top of it, and
an increment on a large base is still large in absolute terms.

**"Which single number do I put in the plan?"** The lower bound if it must be defensible;
the midpoint if it is a forecast you will revise. Say which, and never mix the two across
channels.

**"Why has this changed since the last model?"** Models are re-estimated on new data and
estimates move. Track the range, not the point — if the new range overlaps the old, nothing
has actually changed.

**"So we should stop spending on X?"** Almost never the right conclusion from
observational evidence alone. Propose a step down, measured, then another. That sequence is
also the cheapest experiment available.

**"The platform says this channel has 8x ROAS."** Platform-reported ROAS is last-touch
credit for conversions that would substantially have happened anyway. The two numbers
measure different things, and the gap is usually largest exactly where the channel is least
incremental.

**"Can we do the whole reallocation at once?"** Move part of the way, hold for a full
purchase cycle, check the outcome against what the model predicted, then move the rest.

**"How confident are you?"** Answer with the actual confidence, including when it is low.
A modeller who says "not very, here is why, and here is the test that would fix it" is
trusted far longer than one who never qualifies anything.

---

## 7. Things that quietly destroy credibility

* A point estimate where the interval was wide.
* In-sample fit presented as validation.
* An optimiser recommendation extrapolating beyond observed spend, unmarked.
* An organic channel given a ROAS.
* A channel at 40% of contribution with no comment on why.
* A number that moved since last quarter with no explanation offered.
* Any hint of defensiveness when challenged. The uncertainty *is* the finding; presenting
  it as a weakness to be minimised invites exactly the scepticism it deserves.

---

## Related skills

`mmm-stakeholder-reporting` for the written reports behind the decks.
`mmm-attribution` for contribution, ROAS and marginal-versus-average.
`mmm-budget-optimization` for the reallocation itself.
`mmm-experiment-roadmap` for the roadmap slide.
