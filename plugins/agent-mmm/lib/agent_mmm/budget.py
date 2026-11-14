"""Budget scenarios with a guardrail ladder.

One "optimal" allocation is the least useful thing an MMM can hand over: it
depends entirely on how far the optimiser was allowed to move, and the further
it moves the more it relies on response curves outside the spend the data has
seen. So this module answers the question at several levels of ambition:

* **Guardrail ladder** — small (±10%), moderate (±25%), large (±50%) and
  aggressive (0–2×, extrapolates). The same budget is optimised under each.
* **Budget levels** — total spend at −20%, −10%, same, +10%, +20% (or custom).
* **Uncertainty on every answer** — each allocation is scored on every
  posterior draw: expected gain, 90% interval, probability it beats today.
* **Disputed channels** (where frameworks disagree) stay within the small band
  at every level unless explicitly released.
* **Efficiency frontier** — best achievable outcome as total budget changes,
  with the marginal return of the last unit, per guardrail level.
* **Recommendation** — the largest level that still clears the confidence bar,
  does not extrapolate, and is not into diminishing returns; with the reason.

Works on any framework's ``results.json``: response curves are interpolated per
posterior draw from the curve grid (total incremental outcome over the training
window vs a spend multiplier). Optimisation is greedy marginal allocation on the
mean curve, which is optimal for concave curves and needs no solver.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agent_mmm.results_schema import ModelResults, summarize


@dataclass(frozen=True)
class Guardrail:
    key: str
    label: str
    lo: float          # min multiple of current spend per channel
    hi: float          # max multiple
    extrapolates: bool = False


GUARDRAILS: tuple[Guardrail, ...] = (
    Guardrail("small", "Small adjustments (±10%)", 0.9, 1.1),
    Guardrail("moderate", "Moderate (±25%)", 0.75, 1.25),
    Guardrail("large", "Large (±50%)", 0.5, 1.5),
    Guardrail("aggressive", "Aggressive (0–2×, extrapolates)", 0.0, 2.0, extrapolates=True),
)
BUDGET_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
CONFIDENCE_BAR = 0.80         # P(better than today) needed to recommend a level
DIMINISHING = 0.25            # next level must add ≥25% more gain to be worth its risk
DATA_SUPPORTED_MAX = 1.5      # beyond this multiple of current spend, curves extrapolate
STEPS = 400


# --------------------------------------------------------------------------- #
# Response curves per channel
# --------------------------------------------------------------------------- #
@dataclass
class ChannelCurve:
    name: str
    current: float
    grid: np.ndarray       # (M,) spend
    draws: np.ndarray      # (S, M) total incremental outcome
    mean: np.ndarray       # (M,)

    def value_mean(self, x: float) -> float:
        return float(np.interp(x, self.grid, self.mean))

    def value_draws(self, x: float) -> np.ndarray:
        # Linear interpolation per draw on the shared spend grid.
        g = self.grid
        i = int(np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2))
        w = 0.0 if g[i + 1] == g[i] else (x - g[i]) / (g[i + 1] - g[i])
        w = float(np.clip(w, 0.0, 1.0)) if x <= g[-1] else w   # linear extension past the grid
        return self.draws[:, i] * (1 - w) + self.draws[:, i + 1] * w


def curves_from_results(r: ModelResults) -> dict[str, ChannelCurve]:
    out = {}
    for ch in r.channels:
        c = r.curves.get("channels", {}).get(ch)
        spend = r.channel_summary.get(ch, {}).get("spend")
        if not c or not spend or c["spend"][0] is None:
            continue
        out[ch] = ChannelCurve(ch, float(spend), np.asarray(c["spend"], float),
                               np.asarray(c["draws"], float), np.asarray(c["mean"], float))
    return out


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def allocate(curves: dict[str, ChannelCurve], total: float,
             bounds: dict[str, tuple[float, float]]) -> tuple[dict[str, float], bool, float]:
    """Greedy marginal allocation on the mean curves within per-channel bounds.

    Returns (allocation, feasible, total_allocated). If the bounds cannot absorb
    the total, spend is capped at the bounds and ``feasible`` is False.
    """
    lo = {c: bounds[c][0] for c in curves}
    hi = {c: bounds[c][1] for c in curves}
    if sum(lo.values()) > total + 1e-9:
        scale = total / sum(lo.values())
        return {c: lo[c] * scale for c in curves}, False, total
    alloc = dict(lo)
    remaining = total - sum(lo.values())
    room = sum(hi[c] - lo[c] for c in curves)
    feasible = room >= remaining - 1e-9
    remaining = min(remaining, room)
    step = remaining / STEPS if remaining > 0 else 0
    for _ in range(STEPS if step else 0):
        best, gain = None, -np.inf
        for c, cv in curves.items():
            if alloc[c] + step > hi[c] + 1e-9:
                continue
            g = cv.value_mean(alloc[c] + step) - cv.value_mean(alloc[c])
            if g > gain:
                best, gain = c, g
        if best is None:
            break
        alloc[best] += step
    return alloc, feasible, sum(alloc.values())


def evaluate(curves: dict[str, ChannelCurve], alloc: dict[str, float]) -> np.ndarray:
    """Per-draw outcome change vs current spend (draws aligned across channels)."""
    return sum(cv.value_draws(alloc[c]) - cv.value_draws(cv.current) for c, cv in curves.items())


def _bounds(curves, g: Guardrail, disputed: set[str], locked: dict[str, float],
            custom: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    small = GUARDRAILS[0]   # disputed channels never move more than ±10%
    b = {}
    for c, cv in curves.items():
        if c in locked:
            b[c] = (locked[c], locked[c])
        elif c in custom:
            b[c] = custom[c]
        elif c in disputed:
            lo_m, hi_m = max(g.lo, small.lo), min(g.hi, small.hi)
            b[c] = (cv.current * lo_m, cv.current * hi_m)
        else:
            b[c] = (cv.current * g.lo, cv.current * g.hi)
    return b


@dataclass
class Scenario:
    budget_factor: float
    guardrail: str
    guardrail_label: str
    total_requested: float
    total_allocated: float
    feasible: bool
    allocation: dict[str, dict[str, Any]]
    gain: dict[str, Any]
    gain_pct_of_outcome: dict[str, Any]
    prob_better: float
    gain_vs_mix: dict[str, Any]          # vs today's mix scaled to the same total
    prob_better_than_mix: float
    return_on_change: Optional[dict[str, Any]]
    extrapolated: list[str]
    disputed_held: list[str]


def run_scenario(curves, total_outcome: float, factor: float, g: Guardrail,
                 disputed: set[str], locked: dict[str, float],
                 custom: dict[str, tuple[float, float]]) -> Scenario:
    current_total = sum(cv.current for cv in curves.values())
    total = current_total * factor
    bounds = _bounds(curves, g, disputed, locked, custom)
    alloc, feasible, allocated = allocate(curves, total, bounds)
    d = evaluate(curves, alloc)
    d_mix = d - evaluate(curves, {c: cv.current * allocated / current_total for c, cv in curves.items()})
    delta_spend = allocated - current_total
    rows = {}
    extrap = []
    for c, cv in curves.items():
        new = alloc[c]
        h = max(new * 0.01, 1e-9)
        mr = (cv.value_draws(new + h) - cv.value_draws(new)) / h
        mult = new / cv.current if cv.current else None
        if mult is not None and mult > DATA_SUPPORTED_MAX + 1e-9:
            extrap.append(c)
        rows[c] = {"current": cv.current, "new": new,
                   "change_pct": (new / cv.current - 1) if cv.current else None,
                   "marginal_return": summarize(mr)}
    return Scenario(
        budget_factor=factor, guardrail=g.key, guardrail_label=g.label,
        total_requested=total, total_allocated=allocated, feasible=feasible, allocation=rows,
        gain=summarize(d), gain_pct_of_outcome=summarize(d / total_outcome) if total_outcome else {},
        prob_better=float(np.mean(d > 0)),
        gain_vs_mix=summarize(d_mix), prob_better_than_mix=float(np.mean(d_mix > 0)),
        return_on_change=summarize(d / delta_spend) if abs(delta_spend) > 1e-6 * current_total else None,
        extrapolated=extrap, disputed_held=sorted(disputed & set(curves)),
    )


def frontier(curves, g: Guardrail, disputed, locked, custom,
             factors=tuple(np.round(np.arange(0.6, 1.41, 0.05), 2))) -> list[dict[str, Any]]:
    """Best mean outcome change vs total budget, plus the proportional-scaling baseline."""
    out = []
    current_total = sum(cv.current for cv in curves.values())
    prev = None
    for f in factors:
        alloc, feasible, allocated = allocate(curves, current_total * f, _bounds(curves, g, disputed, locked, custom))
        opt = float(np.mean(evaluate(curves, alloc)))
        prop = float(np.mean(evaluate(curves, {c: cv.current * f for c, cv in curves.items()})))
        mr = None if prev is None or allocated == prev[0] else (opt - prev[1]) / (allocated - prev[0])
        out.append({"factor": float(f), "spend": allocated, "feasible": feasible,
                    "proportional_spend": current_total * float(f),
                    "optimised_gain": opt, "proportional_gain": prop, "marginal_return": mr})
        prev = (allocated, opt)
    return out


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def recommend(scenarios: list[Scenario], factor: float = 1.0,
              guardrails: tuple[Guardrail, ...] = GUARDRAILS) -> dict[str, Any]:
    """At today's budget, compare with today. At another budget, compare with today's
    channel mix scaled to that budget — spending less always loses outcome, so 'better
    than today' would reject every plan for a cut."""
    ladder = [s for s in scenarios if s.budget_factor == factor and s.feasible]
    same = factor == 1.0
    base_name = "today" if same else "scaling today's mix"
    prob = (lambda s: s.prob_better) if same else (lambda s: s.prob_better_than_mix)
    gain_of = (lambda s: s.gain) if same else (lambda s: s.gain_vs_mix)
    order = [g.key for g in guardrails]
    ladder.sort(key=lambda s: order.index(s.guardrail))
    chosen, reasons = None, []
    for s in ladder:
        g = next(x for x in guardrails if x.key == s.guardrail)
        if not same:
            scale = s.total_allocated / sum(a["current"] for a in s.allocation.values())
            spread = max(abs(a["new"] / a["current"] - scale) for a in s.allocation.values() if a["current"])
            if spread < 0.02:
                reasons.append(f"{s.guardrail_label}: no room to re-allocate at this budget — every channel "
                               "has to move by about the same amount.")
                continue
        if g.extrapolates or s.extrapolated:
            reasons.append(f"{s.guardrail_label}: not recommended — relies on response curves beyond "
                           f"{DATA_SUPPORTED_MAX:.1f}× historical spend ({', '.join(s.extrapolated) or 'by design'}). "
                           "Test before acting.")
            continue
        if prob(s) < CONFIDENCE_BAR:
            reasons.append(f"{s.guardrail_label}: stops here — only {prob(s):.0%} chance of beating "
                           f"{base_name} (bar is {CONFIDENCE_BAR:.0%}).")
            break
        if chosen is not None:
            prev_gain, gain = gain_of(chosen)["mean"] or 0, gain_of(s)["mean"] or 0
            if prev_gain > 0 and gain < prev_gain * (1 + DIMINISHING):
                reasons.append(f"{s.guardrail_label}: adds only {gain / prev_gain - 1:+.0%} over "
                               f"{chosen.guardrail_label} for larger moves — diminishing returns.")
                break
        chosen = s
    if chosen is None:
        return {"guardrail": None, "reasons": reasons or ["No level clears the confidence bar."],
                "summary": f"No reallocation is confidently better than {base_name}"
                           + ("" if same else " (scale every channel by the same factor)")
                           + ". Run experiments on the largest channels to sharpen the curves."}
    moves = sorted(chosen.allocation.items(), key=lambda kv: (kv[1]["new"] - kv[1]["current"]))
    cut = [c for c, a in moves if (a["change_pct"] or 0) < -0.02]
    add = [c for c, a in reversed(moves) if (a["change_pct"] or 0) > 0.02]
    g = gain_of(chosen)
    return {
        "guardrail": chosen.guardrail, "label": chosen.guardrail_label, "reasons": reasons,
        "increase": add, "decrease": cut, "baseline": base_name,
        "summary": f"{chosen.guardrail_label}: expected gain vs {base_name} {g['mean']:+,.0f} "
                   f"[{g['p05']:+,.0f}, {g['p95']:+,.0f}], {prob(chosen):.0%} chance of beating {base_name}.",
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
@dataclass
class BudgetPlan:
    run_id: str
    framework: str
    target_label: str
    kpi_kind: str
    period_note: str
    current_total: float
    channels: list[str]
    disputed: list[str]
    locked: dict[str, float]
    guardrails: list[Guardrail] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    frontiers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    recommendations: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BudgetPlan":
        d = dict(d)
        d["guardrails"] = [Guardrail(**g) for g in d.get("guardrails", [])]
        d["scenarios"] = [Scenario(**s) for s in d.get("scenarios", [])]
        return cls(**d)


def load_plans(base: str | Path = ".") -> list[BudgetPlan]:
    """Saved plans (as the user ran them, custom guardrails and locks included), oldest first."""
    out = []
    for p in sorted((Path(base) / "mmm-workspace" / "budget").glob("*/scenarios.json")):
        try:
            out.append(BudgetPlan.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def plan_budget(
    r: ModelResults,
    comparison: Optional[dict] = None,
    budget_factors: tuple[float, ...] = BUDGET_FACTORS,
    guardrails: tuple[Guardrail, ...] = GUARDRAILS,
    locked: Optional[dict[str, float]] = None,
    custom_bounds: Optional[dict[str, tuple[float, float]]] = None,
    release_disputed: bool = False,
) -> BudgetPlan:
    curves = curves_from_results(r)
    if not curves:
        raise ValueError("no channels with spend and response curves to optimise")
    locked = {k: float(v) for k, v in (locked or {}).items() if k in curves}
    custom = {k: (float(a), float(b)) for k, (a, b) in (custom_bounds or {}).items() if k in curves}
    disputed: set[str] = set()
    if comparison and not release_disputed:
        disputed = {ch for ch, e in comparison.get("channels", {}).items() if e.get("agreement") == "disagree"}
    total_outcome = float(np.sum([v for v, h in zip(r.series["expected_mean"], r.series["holdout"]) if not h]))

    plan = BudgetPlan(
        run_id=r.run_id, framework=r.framework, target_label=r.target_label, kpi_kind=r.kpi_kind,
        period_note=f"Amounts cover the training window ({r.window['n_train']} periods, "
                    f"{r.window['train_start']} → {r.window['train_end']}).",
        current_total=sum(cv.current for cv in curves.values()), channels=list(curves),
        disputed=sorted(disputed & set(curves)), locked=locked, guardrails=list(guardrails),
    )
    skipped = [c for c in r.channels if c not in curves]
    if skipped:
        plan.notes.append(f"Not optimised (no spend or curve): {skipped}.")
    if disputed:
        plan.notes.append(f"Held to ±10% at every level because the frameworks disagree: {sorted(disputed)}.")
    diag = r.diagnostics or {}
    if (diag.get("max_rhat") or 1) > 1.05 or diag.get("divergences") or (diag.get("min_ess") or 1e9) < 400:
        plan.notes.append("This model did not fully converge; treat every scenario as indicative only.")
    for f in budget_factors:
        for g in guardrails:
            plan.scenarios.append(run_scenario(curves, total_outcome, f, g, disputed, locked, custom))
    for g in guardrails:
        plan.frontiers[g.key] = frontier(curves, g, disputed, locked, custom)
    for f in budget_factors:
        plan.recommendations[f"{f:g}"] = recommend(plan.scenarios, f, tuple(guardrails))
    return plan


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _n(x: Optional[float]) -> str:
    return "—" if x is None else (f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.3g}")


def _ratio(s: Optional[dict]) -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{s['mean']:.3g} [{s['p05']:.3g}, {s['p95']:.3g}]"


def _iv(s: Optional[dict], pct: bool = False) -> str:
    if not s or s.get("mean") is None:
        return "—"
    if pct:
        return f"{s['mean']:+.1%} [{s['p05']:+.1%}, {s['p95']:+.1%}]"
    return f"{s['mean']:+,.0f} [{s['p05']:+,.0f}, {s['p95']:+,.0f}]"


def render_plan(p: BudgetPlan) -> str:
    lines = [f"# Budget scenarios — {p.framework} ({p.run_id})", "",
             f"Current spend {_n(p.current_total)} across {len(p.channels)} channels. {p.period_note}",
             f"Gains are in {p.target_label}, vs today's allocation, with 90% intervals.", ""]
    rec = p.recommendations.get("1")
    if rec:
        lines += ["## Recommendation at today's budget", "", f"**{rec['summary']}**", ""]
        if rec.get("increase"):
            lines.append(f"- Increase: {', '.join(rec['increase'])}")
        if rec.get("decrease"):
            lines.append(f"- Decrease: {', '.join(rec['decrease'])}")
        lines += [f"- {x}" for x in rec.get("reasons", [])] + [""]
    lines += ["## The guardrail ladder at today's budget", "",
              "| Guardrail | Expected gain | Gain as % of outcome | P(better than today) | Extrapolates |",
              "|---|---|---|---|---|"]
    for s in [s for s in p.scenarios if s.budget_factor == 1.0]:
        lines.append(f"| {s.guardrail_label} | {_iv(s.gain)} | {_iv(s.gain_pct_of_outcome, pct=True)} | "
                     f"{s.prob_better:.0%} | {', '.join(s.extrapolated) or 'no'} |")
    lines += ["", "### Allocation by guardrail (change vs today)", "",
              "| Channel | Today | " + " | ".join(g.label for g in p.guardrails) + " |",
              "|---|---|" + "---|" * len(p.guardrails)]
    base = {s.guardrail: s for s in p.scenarios if s.budget_factor == 1.0}
    for c in p.channels:
        cells = []
        for g in p.guardrails:
            a = base[g.key].allocation[c] if g.key in base else None
            cells.append("—" if a is None else f"{a['change_pct']:+.0%}")
        flag = " (held)" if c in p.disputed else " (locked)" if c in p.locked else ""
        lines.append(f"| {c}{flag} | {_n(base[p.guardrails[0].key].allocation[c]['current'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Budget levels", "",
              "| Total budget | Guardrail | Change vs today | Gain vs scaling today's mix | P(beats scaled mix) | Δ outcome per Δ budget |",
              "|---|---|---|---|---|---|"]
    for s in p.scenarios:
        if s.budget_factor == 1.0:
            continue
        if not s.feasible:
            lines.append(f"| {s.budget_factor - 1:+.0%} | {s.guardrail_label} | guardrail too tight for this budget | — | — | — |")
            continue
        lines.append(f"| {s.budget_factor - 1:+.0%} | {s.guardrail_label} | {_iv(s.gain)} | {_iv(s.gain_vs_mix)} | "
                     f"{s.prob_better_than_mix:.0%} | {_ratio(s.return_on_change)} |")
    lines += ["", "*Change vs today* is the total effect of the new budget and allocation. *Gain vs scaling "
              "today's mix* isolates the value of re-allocating. *Δ outcome per Δ budget* is outcome gained per "
              "unit of budget added; for a cut, a negative value means outcome went up while spend went down.", ""]
    for f, rec in p.recommendations.items():
        if f == "1":
            continue
        lines.append(f"- **Budget {float(f) - 1:+.0%}:** {rec['summary']}")
    if p.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in p.notes]
    return "\n".join(lines) + "\n"


def png_ladder(p: BudgetPlan, path: Path) -> Path:
    from agent_mmm.charts import NEUTRAL, PALETTE, SURFACE, _legend_below, _plt, _save

    plt = _plt()
    ss = [s for s in p.scenarios if s.budget_factor == 1.0]
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(ss) + 1.6))
    for i, s in enumerate(ss):
        col = NEUTRAL if s.extrapolated or s.guardrail == "aggressive" else PALETTE[0]
        ax.hlines(i, s.gain["p05"], s.gain["p95"], color=col, lw=2)
        ax.plot(s.gain["mean"], i, "o", color=col, ms=9, mec=SURFACE, mew=1.5)
        ax.text(s.gain["p95"], i, f"  {s.prob_better:.0%} better", va="center", color="#52514e", fontsize=9)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(range(len(ss)), [s.guardrail_label for s in ss])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Gain vs today by guardrail, same budget — {p.framework}")
    ax.set_xlabel(f"Change in {p.target_label} (90% interval)")
    from agent_mmm.charts import _human
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    return _save(fig, path)


def png_allocation(p: BudgetPlan, path: Path) -> Path:
    from agent_mmm.charts import PALETTE, _legend_below, _plt, _save

    plt = _plt()
    base = {s.guardrail: s for s in p.scenarios if s.budget_factor == 1.0}
    keys = [g.key for g in p.guardrails if g.key in base]
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(p.channels) + 1.8))
    k = len(keys)
    for j, key in enumerate(keys):
        vals = [base[key].allocation[c]["change_pct"] * 100 for c in p.channels]
        y = np.arange(len(p.channels)) + (j - (k - 1) / 2) * 0.18
        ax.barh(y, vals, height=0.16, color=PALETTE[j], label=base[key].guardrail_label)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(range(len(p.channels)), p.channels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Recommended spend change by guardrail, same budget — {p.framework}")
    _legend_below(ax, k)
    return _save(fig, path)


def png_frontier(p: BudgetPlan, path: Path) -> Path:
    from agent_mmm.charts import NEUTRAL, PALETTE, _human, _legend_below, _plt, _save

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, g in enumerate(p.guardrails):
        fr = p.frontiers.get(g.key)
        if not fr:
            continue
        ok = [x for x in fr if x["feasible"]]   # only budgets this guardrail can actually reach
        ax.plot([x["spend"] for x in ok], [x["optimised_gain"] for x in ok], color=PALETTE[j],
                ls="--" if g.extrapolates else "-", label=g.label)
    fr = p.frontiers[p.guardrails[0].key]
    ax.plot([x["proportional_spend"] for x in fr], [x["proportional_gain"] for x in fr], color=NEUTRAL, lw=1.5,
            label="Scale today's mix")
    ax.axvline(p.current_total, color="#52514e", lw=0.8)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.annotate("today", (p.current_total, 0), textcoords="offset points", xytext=(4, 6), color="#52514e", fontsize=8)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    ax.set_xlabel("Total spend over training window")
    ax.set_ylabel(f"Change in {p.target_label} vs today")
    ax.set_title(f"Efficiency frontier — {p.framework}")
    _legend_below(ax, 3)
    return _save(fig, path)


def run_budget(results: list[ModelResults], comparison: Optional[dict] = None, base: str | Path = ".",
               **kwargs) -> dict[str, Any]:
    out_dir = Path(base) / "mmm-workspace" / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    plans, written = [], {}
    for r in results:
        p = plan_budget(r, comparison, **kwargs)
        plans.append(p)
        d = out_dir / r.run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "scenarios.json").write_text(json.dumps(p.to_dict(), indent=1, default=str), encoding="utf-8")
        (d / "budget_report.md").write_text(render_plan(p), encoding="utf-8")
        figs = [png_ladder(p, d / "guardrail_ladder.png"), png_allocation(p, d / "allocation_by_guardrail.png"),
                png_frontier(p, d / "efficiency_frontier.png")]
        written[r.run_id] = {"report": str(d / "budget_report.md"), "json": str(d / "scenarios.json"),
                             "figures": [str(f) for f in figs]}
    if len(plans) > 1:
        (out_dir / "agreement.md").write_text(render_agreement(plans), encoding="utf-8")
        written["agreement"] = str(out_dir / "agreement.md")
    return {"plans": plans, "written": written}


def render_agreement(plans: list[BudgetPlan]) -> str:
    """Do the frameworks point the same way at their recommended guardrail?"""
    lines = ["# Do the models agree on the direction?", "",
             "Direction of each channel's change at each model's recommended guardrail, today's budget.", "",
             "| Channel | " + " | ".join(f"{p.framework}" for p in plans) + " | Agreement |",
             "|---|" + "---|" * len(plans) + "---|"]
    chans = [c for c in plans[0].channels if all(c in p.channels for p in plans)]
    for c in chans:
        dirs = []
        for p in plans:
            rec = p.recommendations.get("1", {})
            dirs.append("up" if c in rec.get("increase", []) else "down" if c in rec.get("decrease", []) else "hold")
        agree = "agree" if len(set(dirs)) == 1 else ("conflict" if {"up", "down"} <= set(dirs) else "partial")
        lines.append(f"| {c} | " + " | ".join(dirs) + f" | {agree} |")
    lines += ["", "Act on channels where the models agree. Where they conflict, hold spend and test."]
    return "\n".join(lines) + "\n"
