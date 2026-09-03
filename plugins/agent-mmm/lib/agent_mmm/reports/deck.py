"""Presentation decks: the argument, not the slides.

A stakeholder deck fails in one of two ways. Either it is a data dump — every
chart the modeller found interesting, in the order they were produced — or it is
a conclusion with the reasoning removed, which the audience correctly refuses to
act on.

What travels is an *argument*: a claim, the evidence for it, the honest limits
of that evidence, and the decision it supports. This module builds that argument
as a structured deck spec — one slide at a time, each with a headline that says
something (never "Channel Performance"), the evidence it rests on, a chart
specification, speaker notes, and the objection it will draw.

It deliberately produces a spec rather than a rendered file. The spec is
reviewable as text, and the substance can be argued with before anyone spends
time on styling. Rendering to HTML, PowerPoint or Google Slides is a separate,
later, easier problem.

Four audiences, four different arguments from the same model:

* **CMO** — where the next pound goes, and what we are still guessing about.
* **CFO** — what was returned on what was spent, with the uncertainty attached
  and the accounting basis stated.
* **Marketing Ops** — what changes in the plan next cycle, concretely.
* **Data Science** — how the model was built and where it is weak, in enough
  detail to be attacked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "ChartSpec",
    "Slide",
    "DeckSpec",
    "AUDIENCES",
    "build_deck",
    "render_deck_markdown",
    "write_deck",
]

AUDIENCES = ("cmo", "cfo", "mops", "ds")

_AUDIENCE_FRAMING: dict[str, dict[str, str]] = {
    "cmo": {
        "title": "Where the next pound goes",
        "audience": "Chief Marketing Officer and the marketing leadership team",
        "duration": "20 minutes, 10 slides, half the time in discussion",
        "wants": "a defensible reallocation and a clear view of what is still unknown",
        "fears": "being told a channel they have championed does not work, in front of their team",
        "register": "plain language; no adstock, no posterior, no r-hat",
    },
    "cfo": {
        "title": "Return on marketing investment",
        "audience": "Chief Financial Officer and finance business partners",
        "duration": "15 minutes, 8 slides, expect interruption",
        "wants": "a number they can put in a plan, with its error bar and its accounting basis",
        "fears": "a number that will move next quarter without explanation",
        "register": "reconciled to the ledger; state gross vs net; ranges, never point estimates",
    },
    "mops": {
        "title": "What changes in the plan",
        "audience": "Marketing operations, channel owners and the media agency",
        "duration": "30 minutes, working session",
        "wants": "specific changes to specific line items, and the reason for each",
        "fears": "a reallocation that cannot be executed within contracted commitments",
        "register": "concrete; name channels, campaigns and amounts; acknowledge constraints",
    },
    "ds": {
        "title": "Model construction and its limits",
        "audience": "Data science, analytics and any external reviewer",
        "duration": "45 minutes, adversarial",
        "wants": "enough detail to reproduce the model and to attack it",
        "fears": "discovering a structural error after the results have been socialised",
        "register": "technical and complete; lead with the weaknesses",
    },
}


@dataclass
class ChartSpec:
    """What to plot, and what the reader should take from it.

    ``takeaway`` is required by convention: a chart nobody can state the point
    of is a chart that should not be in the deck.
    """

    kind: str
    """waterfall | bar | line | interval | scatter | response_curve | table | none"""
    title: str
    takeaway: str
    x: str = ""
    y: str = ""
    series: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Slide:
    """One slide: a claim, its support, and the challenge it invites."""

    headline: str
    """A sentence that asserts something. Never a category label."""
    body: list[str] = field(default_factory=list)
    chart: Optional[ChartSpec] = None
    speaker_notes: str = ""
    anticipated_question: str = ""
    answer: str = ""
    confidence: str = ""
    """high | medium | low — stated on the slide, not buried in an appendix."""

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "chart"}
        d["chart"] = self.chart.to_dict() if self.chart else None
        return d


@dataclass
class DeckSpec:
    audience: str
    title: str
    subtitle: str = ""
    company: str = ""
    prepared: str = ""
    framing: dict[str, str] = field(default_factory=dict)
    slides: list[Slide] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audience": self.audience,
            "title": self.title,
            "subtitle": self.subtitle,
            "company": self.company,
            "prepared": self.prepared,
            "framing": self.framing,
            "slides": [s.to_dict() for s in self.slides],
            "caveats": self.caveats,
            "open_questions": self.open_questions,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float], digits: int = 1) -> str:
    return f"{x:.{digits}%}" if isinstance(x, (int, float)) else "—"


def _num(x: Optional[float], digits: int = 2) -> str:
    return f"{x:,.{digits}f}" if isinstance(x, (int, float)) else "—"


def _confidence_from_tier(tier: str) -> str:
    return {"PASS": "high", "WARN": "medium", "FAIL": "low"}.get(str(tier).upper(), "low")


def _sorted_channels(contributions: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = [(k, v) for k, v in contributions.items() if isinstance(v, dict)]
    return sorted(items, key=lambda kv: -(kv[1].get("contribution_share") or 0.0))


def _wide_interval(entry: dict[str, Any]) -> bool:
    """A channel whose interval spans a factor of three has not been measured."""
    low, high = entry.get("roas_low"), entry.get("roas_high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return False
    return low > 0 and high / low >= 3.0


# --------------------------------------------------------------------------- #
# Deck construction
# --------------------------------------------------------------------------- #
def build_deck(
    audience: str,
    *,
    company: str = "",
    target_label: str = "revenue",
    is_monetary: bool = True,
    tier: str = "UNKNOWN",
    baseline_share: Optional[float] = None,
    contributions: Optional[dict[str, Any]] = None,
    reallocation: Optional[list[dict[str, Any]]] = None,
    experiments: Optional[list[dict[str, Any]]] = None,
    roadmap_questions: Optional[list[str]] = None,
    data_caveats: Optional[list[str]] = None,
    model_notes: Optional[dict[str, Any]] = None,
) -> DeckSpec:
    """Build the deck spec for one audience.

    Every argument is optional: the deck degrades to stating what is not yet
    known rather than inventing content, because a slide that quietly omits its
    own uncertainty is the failure mode this module exists to prevent.
    """
    audience = audience.lower()
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got '{audience}'")

    framing = dict(_AUDIENCE_FRAMING[audience])
    deck = DeckSpec(
        audience=audience,
        title=framing["title"],
        subtitle=f"Marketing Mix Model — {company}" if company else "Marketing Mix Model",
        company=company,
        prepared=datetime.now().strftime("%Y-%m-%d"),
        framing=framing,
    )

    contributions = contributions or {}
    ranked = _sorted_channels(contributions)
    confidence = _confidence_from_tier(tier)

    builder = {
        "cmo": _build_cmo,
        "cfo": _build_cfo,
        "mops": _build_mops,
        "ds": _build_ds,
    }[audience]
    builder(
        deck,
        target_label=target_label,
        is_monetary=is_monetary,
        tier=tier,
        confidence=confidence,
        baseline_share=baseline_share,
        ranked=ranked,
        reallocation=reallocation or [],
        experiments=experiments or [],
        model_notes=model_notes or {},
    )

    deck.caveats = list(data_caveats or [])
    if tier.upper() == "FAIL":
        deck.caveats.insert(
            0,
            "This model did not pass validation. Nothing in this deck should drive a budget "
            "decision until the failures are resolved.",
        )
    if not experiments:
        deck.caveats.append(
            "No incrementality experiment has been run. Every number here is estimated from "
            "observed correlation, and the split of credit between correlated channels is the "
            "model's assumption rather than a measurement."
        )
    if baseline_share is not None and baseline_share < 0.3:
        deck.caveats.append(
            f"The baseline accounts for only {_pct(baseline_share)} of {target_label}. A low "
            "baseline usually means media is being credited with demand it did not create."
        )

    deck.open_questions = list(roadmap_questions or [])
    return deck


def _build_cmo(deck: DeckSpec, **k: Any) -> None:
    ranked = k["ranked"]
    target = k["target_label"]
    conf = k["confidence"]
    baseline = k["baseline_share"]

    deck.slides.append(Slide(
        headline=(
            f"Media drives {_pct(1 - baseline)} of {target}; the rest would happen anyway"
            if baseline is not None
            else f"What media contributes to {target}, and what we still cannot separate"
        ),
        body=[
            "The baseline is everything that would have happened with no advertising at all — "
            "brand equity, distribution, price, season, and demand you already had.",
            "Every channel number in this deck is a share of what is left over that baseline.",
            "Read the baseline first. If it is wrong, every channel number below it is wrong too.",
        ],
        chart=ChartSpec(
            kind="waterfall",
            title=f"{target} decomposition",
            takeaway="Most of the business is not media, and that is normal.",
            x="component",
            y=target,
            series=["baseline"] + [c for c, _ in ranked],
            annotations=["Label the baseline explicitly — audiences assume the whole bar is media."],
        ),
        speaker_notes=(
            "Spend a full minute here. Every downstream disagreement traces back to someone "
            "assuming the baseline is zero. If the baseline is under 30% say so out loud rather "
            "than hoping nobody asks."
        ),
        anticipated_question="Why is the baseline so large? What are we paying the agency for?",
        answer=(
            "The baseline is the business you have built, not wasted spend. Media's job is the "
            "increment on top of it, and an increment on a large base is still a large number in "
            "absolute terms."
        ),
        confidence=conf,
    ))

    if ranked:
        top = ranked[:5]
        deck.slides.append(Slide(
            headline=(
                f"{top[0][0]} is the largest contributor at "
                f"{_pct(top[0][1].get('contribution_share'))} of media-driven {target}"
            ),
            body=[
                f"{name}: {_pct(v.get('contribution_share'))} of media contribution"
                + (f", {v.get('roas_label', 'ROAS')} {_num(v.get('roas'))}" if v.get("roas") else "")
                for name, v in top
            ],
            chart=ChartSpec(
                kind="bar",
                title="Contribution by channel",
                takeaway="Contribution is about size, not efficiency — the next slide covers that.",
                x="channel",
                y="contribution share",
                annotations=["Order by contribution, not alphabetically."],
            ),
            speaker_notes=(
                "Contribution and efficiency get confused constantly. Say explicitly that a big "
                "channel is not necessarily a good one, and that the small ones are not "
                "necessarily bad."
            ),
            anticipated_question="Does this mean we should put everything into the top channel?",
            answer=(
                "No. Every channel saturates, so the tenth pound into the top channel returns less "
                "than the first. The reallocation slide is about where the *next* pound goes, "
                "which is a different question from where the current money sits."
            ),
            confidence=conf,
        ))

        vague = [(n, v) for n, v in ranked if _wide_interval(v)]
        if vague:
            deck.slides.append(Slide(
                headline=(
                    f"{len(vague)} channel{'s' if len(vague) > 1 else ''} "
                    f"({', '.join(n for n, _ in vague[:3])}) "
                    f"{'have' if len(vague) > 1 else 'has'} not really been measured"
                ),
                body=[
                    f"{n}: plausible range {_num(v.get('roas_low'))} to {_num(v.get('roas_high'))}"
                    for n, v in vague[:5]
                ],
                chart=ChartSpec(
                    kind="interval",
                    title="Efficiency with uncertainty",
                    takeaway="A range spanning a factor of three is not a measurement.",
                    x="channel",
                    y="ROAS",
                    annotations=["Show the full interval. Never plot the midpoint alone."],
                ),
                speaker_notes=(
                    "This is the most valuable slide in the deck and the one most often cut. "
                    "Presenting a wide interval as a point estimate is how a model loses its "
                    "credibility six months later when the number moves."
                ),
                anticipated_question="So the model cannot tell us about these channels?",
                answer=(
                    "Not from historical data alone — these channels did not vary enough, or moved "
                    "together with something else. An experiment resolves it, and that is what the "
                    "roadmap slide proposes."
                ),
                confidence="low",
            ))

    if k["reallocation"]:
        moves = k["reallocation"][:6]
        deck.slides.append(Slide(
            headline="Moving budget between these channels is the highest-confidence change available",
            body=[
                f"{m.get('channel')}: {_pct(m.get('current_share'))} → {_pct(m.get('proposed_share'))}"
                + (f" ({m.get('rationale')})" if m.get("rationale") else "")
                for m in moves
            ],
            chart=ChartSpec(
                kind="bar",
                title="Current vs proposed allocation",
                takeaway="The change is a shift at the margin, not a rebuild of the plan.",
                x="channel",
                y="share of budget",
                series=["current", "proposed"],
            ),
            speaker_notes=(
                "Frame every move as marginal. Recommending a channel go to zero on observational "
                "evidence alone is how MMMs get discredited — propose a step, measure, and step "
                "again."
            ),
            anticipated_question="How confident are you? Can we do this in one go?",
            answer=(
                "Move part of the way, hold for a full purchase cycle, and check the result "
                "against what the model predicted. If it holds, move the rest. That sequence is "
                "also the cheapest experiment available."
            ),
            confidence=conf,
        ))

    _append_experiment_slide(deck, k, tone="cmo")
    _append_next_steps(deck, k, tone="cmo")


def _build_cfo(deck: DeckSpec, **k: Any) -> None:
    ranked = k["ranked"]
    target = k["target_label"]
    conf = k["confidence"]

    deck.slides.append(Slide(
        headline="What the marketing budget returned, with the uncertainty attached",
        body=[
            "Every figure is a range, not a point. The range is the finding — a channel whose "
            "return could be 0.5x or 5x has not been measured, and averaging that to 2.75x would "
            "be a fabrication.",
            "Figures reconcile to the media spend in the ledger. Where they do not, the difference "
            "is stated.",
        ],
        chart=ChartSpec(
            kind="interval",
            title=f"Return by channel ({target})",
            takeaway="Rank by the lower bound, not the midpoint: that is what is defensible.",
            x="channel",
            y="return",
            annotations=["Mark the break-even line explicitly."],
        ),
        speaker_notes=(
            "Lead with the accounting basis: gross or net of fees, and whether the target is gross "
            "or net revenue. Getting asked this after presenting a number is much worse than "
            "stating it first."
        ),
        anticipated_question="Which single number do I put in the plan?",
        answer=(
            "The lower bound, if the plan has to be defensible. The midpoint, if it is a forecast "
            "you will revise. Say which you used, and never mix the two across channels."
        ),
        confidence=conf,
    ))

    if ranked:
        deck.slides.append(Slide(
            headline="Efficiency at the margin differs from average efficiency, and only the margin is decision-relevant",
            body=[
                "Average return divides all the credit by all the spend. It answers 'was this "
                "worth doing', which is a question about the past.",
                "Marginal return is what the next pound produces. It is always lower, because "
                "channels saturate, and it is the only one that informs the next budget.",
                "Reallocating on average return systematically overfunds saturated channels.",
            ],
            chart=ChartSpec(
                kind="response_curve",
                title="Response curves with current spend marked",
                takeaway="Where the curve flattens, further spend buys little.",
                x="spend",
                y=target,
                annotations=[
                    "Mark current spend on each curve.",
                    "Shade beyond the observed spend range — the model is extrapolating there and "
                    "should not be trusted for a decision.",
                ],
            ),
            speaker_notes=(
                "If the CFO takes one thing away, make it this. It is also the strongest defence "
                "against 'the model said channel X has 8x return so give it everything'."
            ),
            anticipated_question="Why has this number changed since the last model?",
            answer=(
                "Media mix models are re-estimated on new data and the estimates move. Track the "
                "range, not the point — if the new range overlaps the old one, nothing has "
                "actually changed."
            ),
            confidence=conf,
        ))

    deck.slides.append(Slide(
        headline="What this model cannot tell you",
        body=[
            "It cannot separate channels that always moved together. Where two channels correlate "
            "closely, the split between them is the model's assumption.",
            "It cannot price long-term brand effects; the window is what it is.",
            "It cannot extrapolate beyond spend levels ever observed.",
        ],
        chart=ChartSpec(
            kind="none",
            title="",
            takeaway="Stating the limits up front is what makes the rest credible.",
        ),
        speaker_notes=(
            "Finance audiences trust models that declare their own limits and distrust ones that "
            "do not. This slide buys credibility for everything before it."
        ),
        anticipated_question="Then why should we act on any of it?",
        answer=(
            "Because it is a better basis than last-click attribution or the previous plan, and "
            "because the recommendations are marginal moves that we measure and correct."
        ),
        confidence="high",
    ))

    _append_experiment_slide(deck, k, tone="cfo")
    _append_next_steps(deck, k, tone="cfo")


def _build_mops(deck: DeckSpec, **k: Any) -> None:
    conf = k["confidence"]

    deck.slides.append(Slide(
        headline="What changes next cycle, line by line",
        body=[
            f"{m.get('channel')}: {_pct(m.get('current_share'))} → {_pct(m.get('proposed_share'))}"
            + (f" — {m.get('rationale')}" if m.get("rationale") else "")
            for m in k["reallocation"][:10]
        ] or ["No reallocation has been produced yet — the optimiser has not been run."],
        chart=ChartSpec(
            kind="table",
            title="Proposed changes by channel",
            takeaway="Each row is an executable change, not a direction of travel.",
            annotations=["Include the current commitment and notice period per channel."],
        ),
        speaker_notes=(
            "Ask about contracted commitments before promising any of this. An upfront TV "
            "commitment or a minimum agency retainer can make a recommendation undeliverable, and "
            "it is better to find that out here than in the next planning meeting."
        ),
        anticipated_question="We are contractually committed on some of this. What then?",
        answer=(
            "Then we apply the change to the uncommitted portion and phase the rest. Tell us the "
            "constraints and we will re-run the optimiser with them as bounds."
        ),
        confidence=conf,
    ))

    deck.slides.append(Slide(
        headline="What we need from you to make the next model better",
        body=[
            "Consistent campaign naming — the mapping from campaign names to channels is rebuilt "
            "by hand every refresh, and every rename silently drops spend out of the model.",
            "Flight dates and delivery dates, not billing dates.",
            "A log of anything that changed and is not in the data: site migrations, tracking "
            "changes, stockouts, promotions, competitor launches.",
            "Advance notice of any planned dark period — an unplanned gap is a wasted natural "
            "experiment, and a planned one is a free measurement.",
        ],
        chart=ChartSpec(kind="none", title="", takeaway="Data quality is an operational habit."),
        speaker_notes=(
            "This is the highest-leverage slide for future accuracy and the one most likely to be "
            "skipped for time. Do not skip it."
        ),
        anticipated_question="How much work is this?",
        answer=(
            "The naming convention is a one-off. The change log is five minutes a week and it is "
            "the single cheapest thing anyone can do to improve next year's model."
        ),
        confidence="high",
    ))

    _append_experiment_slide(deck, k, tone="mops")
    _append_next_steps(deck, k, tone="mops")


def _build_ds(deck: DeckSpec, **k: Any) -> None:
    notes = k["model_notes"]
    tier = k["tier"]

    deck.slides.append(Slide(
        headline=f"Model validation: {tier} — leading with what is weakest",
        body=[
            f"Framework: {notes.get('framework', 'not stated')}",
            f"Granularity: {notes.get('granularity', 'not stated')}, "
            f"{notes.get('n_periods', '?')} periods",
            f"Adstock: {notes.get('adstock', 'not stated')}; "
            f"saturation: {notes.get('saturation', 'not stated')}",
            f"Convergence: {notes.get('convergence', 'not reported')}",
        ],
        chart=ChartSpec(
            kind="table",
            title="Diagnostics summary",
            takeaway="Convergence first — nothing downstream is readable without it.",
            annotations=["r-hat, ESS, divergences, per-parameter."],
        ),
        speaker_notes=(
            "Order matters: convergence, then fit, then generalisation, then baseline, then "
            "learning, then plausibility. Stop at the first failure and say the rest cannot be "
            "read yet."
        ),
        anticipated_question="What is the biggest risk of the model being wrong?",
        answer=(
            "Identifiability, essentially always. Fit statistics do not detect it, and two models "
            "with the same fit can recommend opposite budgets."
        ),
        confidence=k["confidence"],
    ))

    deck.slides.append(Slide(
        headline="Where the model is under-identified, and what we did about it",
        body=[
            "Collinear channel pairs and the grouping decisions taken.",
            "Parameters whose posterior barely moved from the prior — these are assumptions "
            "being reported as findings.",
            "Always-on channels with insufficient spend variation to be measured.",
        ],
        chart=ChartSpec(
            kind="scatter",
            title="Prior vs posterior by parameter",
            takeaway="Points near the diagonal are priors, not results.",
            x="prior mean",
            y="posterior mean",
        ),
        speaker_notes=(
            "Invite attack here explicitly. A reviewer who finds a structural problem in this room "
            "is much cheaper than one who finds it after the CFO has seen the numbers."
        ),
        anticipated_question="How do you know the collinear split is right?",
        answer=(
            "We do not, and we do not claim to. Where two channels correlate above 0.9 we report "
            "the pair as one number, and the experiment roadmap proposes the test that would "
            "separate them."
        ),
        confidence="medium",
    ))

    deck.slides.append(Slide(
        headline="Validation and refutation results",
        body=[
            "Time-series cross-validation on rolling origins.",
            "Holdout on the final periods.",
            "Refutation: placebo channel, random common cause, subset stability.",
            "Parameter recovery on simulated data with known ground truth.",
        ],
        chart=ChartSpec(
            kind="line",
            title="Out-of-sample predictions vs actual",
            takeaway="In-sample fit is not evidence; this is.",
            x="date",
            y=k["target_label"],
        ),
        speaker_notes=(
            "If any of these were not run, say which and why. A validation section that lists only "
            "the tests that passed is worse than none."
        ),
        anticipated_question="Which validation would most change your confidence?",
        answer=(
            "An incrementality experiment on the largest uncertain channel. Everything else tests "
            "internal consistency; only an experiment tests the causal claim."
        ),
        confidence="high",
    ))

    _append_experiment_slide(deck, k, tone="ds")
    _append_next_steps(deck, k, tone="ds")


def _append_experiment_slide(deck: DeckSpec, k: dict[str, Any], tone: str) -> None:
    experiments = k["experiments"]
    if experiments:
        deck.slides.append(Slide(
            headline=f"{len(experiments)} experiment(s) anchor this model to measured reality",
            body=[
                f"{e.get('channel')}: {e.get('design', 'test')} — "
                f"measured {e.get('result', 'lift')} ({e.get('source', 'internal')})"
                for e in experiments[:5]
            ],
            chart=ChartSpec(
                kind="interval",
                title="Model estimate vs experimental measurement",
                takeaway="Where they agree, confidence rises for the whole decomposition.",
                x="channel",
                y="effect",
                series=["model", "experiment"],
            ),
            speaker_notes=(
                "A calibrated channel constrains its neighbours too, because the total has to add "
                "up. One good test improves every number in the deck, not just its own."
            ),
            anticipated_question="What if the test and the model disagree?",
            answer=(
                "The test wins on that channel, and the disagreement tells us something structural "
                "is wrong — usually a confounder we did not control for."
            ),
            confidence="high",
        ))
        return

    deck.slides.append(Slide(
        headline="Nothing here has been tested; this is the plan to fix that",
        body=[
            "Every number in this deck comes from observed correlation in a media plan that was "
            "never randomised.",
            "One well-powered experiment converts the largest uncertainty into a measurement and "
            "constrains every other channel at the same time.",
            "The roadmap ranks tests by money at risk times how little we currently know.",
        ],
        chart=ChartSpec(
            kind="table",
            title="Proposed experiment roadmap",
            takeaway="Ordered by value of information, not by ease.",
            annotations=[
                "Per test: channel, design, duration, detectable effect, expected effect, "
                "revenue at risk."
            ],
        ),
        speaker_notes=(
            "Do not let this become a vague 'we should test more'. Bring the specific first test, "
            "its cost, its duration, and what result would change the recommendation."
        ),
        anticipated_question="Experiments cost money and revenue. Why now?",
        answer=(
            "Because without one, the split of credit between correlated channels is an "
            "assumption, and we are about to move budget on it. The cost of one test is small "
            "against the cost of reallocating wrongly for a year."
        ),
        confidence="high",
    ))


def _append_next_steps(deck: DeckSpec, k: dict[str, Any], tone: str) -> None:
    steps = {
        "cmo": [
            "Agree the reallocation in principle, at partial size.",
            "Approve the first experiment and its revenue at risk.",
            "Set the date we check the prediction against what happened.",
        ],
        "cfo": [
            "Agree which figure — lower bound or midpoint — goes into the plan.",
            "Confirm the accounting basis: gross or net of fees and returns.",
            "Approve the experiment budget as measurement, not media.",
        ],
        "mops": [
            "Confirm which changes are executable within current commitments.",
            "Agree the campaign naming convention and who owns it.",
            "Start the change log this week.",
        ],
        "ds": [
            "Circulate the spec and diagnostics for adversarial review.",
            "Run the refutation tests not yet completed.",
            "Design the first experiment properly, including its power calculation.",
        ],
    }[tone]

    deck.slides.append(Slide(
        headline="What we are asking for",
        body=steps,
        chart=ChartSpec(kind="none", title="", takeaway="Leave with decisions, not impressions."),
        speaker_notes=(
            "Name an owner and a date against each item before the meeting ends. A deck that "
            "closes without owners produces a follow-up meeting instead of a decision."
        ),
        anticipated_question="What happens if we do nothing?",
        answer=(
            "The current allocation continues on the assumption that it is right, and we learn "
            "nothing this cycle that would tell us whether it is."
        ),
        confidence="high",
    ))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_deck_markdown(deck: DeckSpec) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# {deck.title}")
    a("")
    a(f"**{deck.subtitle}**  ")
    a(f"Audience: {deck.framing.get('audience', deck.audience)}  ")
    a(f"Prepared: {deck.prepared}  ")
    a(f"Format: {deck.framing.get('duration', '')}")
    a("")
    a(f"* **They want:** {deck.framing.get('wants', '')}")
    a(f"* **They fear:** {deck.framing.get('fears', '')}")
    a(f"* **Register:** {deck.framing.get('register', '')}")
    a("")
    a("---")
    a("")

    for i, slide in enumerate(deck.slides, start=1):
        a(f"## Slide {i} — {slide.headline}")
        a("")
        if slide.confidence:
            a(f"`confidence: {slide.confidence}`")
            a("")
        for b in slide.body:
            a(f"* {b}")
        if slide.body:
            a("")
        if slide.chart and slide.chart.kind != "none":
            c = slide.chart
            a(f"**Chart** — `{c.kind}`: {c.title}")
            a("")
            if c.x or c.y:
                a(f"* Axes: {c.x or '—'} × {c.y or '—'}")
            if c.series:
                a(f"* Series: {', '.join(c.series)}")
            a(f"* Takeaway: {c.takeaway}")
            for note in c.annotations:
                a(f"* Annotation: {note}")
            a("")
        elif slide.chart:
            a(f"*No chart — {slide.chart.takeaway}*")
            a("")
        if slide.speaker_notes:
            a(f"> **Speaker notes.** {slide.speaker_notes}")
            a("")
        if slide.anticipated_question:
            a(f"**They will ask:** {slide.anticipated_question}")
            a("")
            a(f"**Answer:** {slide.answer}")
            a("")
        a("---")
        a("")

    if deck.caveats:
        a("## Caveats to state, not bury")
        a("")
        for c in deck.caveats:
            a(f"* {c}")
        a("")

    if deck.open_questions:
        a("## Open questions for the room")
        a("")
        for q in deck.open_questions:
            a(f"* {q}")
        a("")

    return "\n".join(lines)


def write_deck(deck: DeckSpec, base: str | Path = ".") -> Path:
    """Write the deck spec to ``mmm-workspace/reports/deck_<audience>.md``."""
    from agent_mmm.workspace import ensure_workspace

    ensure_workspace(base)
    out_dir = Path(base) / "mmm-workspace" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"deck_{deck.audience}.md"
    path.write_text(render_deck_markdown(deck), encoding="utf-8")
    return path
