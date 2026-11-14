"""Compare models from different frameworks on the same data.

The point is not to crown a winner by R². Two frameworks that disagree about a
channel are telling you the data cannot pin that channel down; two that agree
with tight intervals are giving you a robust number. So the comparison reports:

* **Comparability** — same training and holdout window, or it refuses.
* **Trust** — convergence (R-hat, ESS, divergences). A model that did not
  converge is excluded from recommendations, however well it fits.
* **Forecast quality** — holdout error on periods neither model saw.
  Differences under 2 percentage points of MAPE are treated as a tie.
* **Agreement per channel** — do the ROAS intervals overlap? How far apart are
  the means? Do the models rank channels the same way?
* **Findings** — plain-language conclusions and what to do about each.

Inputs are ``ModelResults`` (``results.json``), so any framework can take part.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np

from agent_mmm.results_schema import ModelResults

RHAT_MAX = 1.05
ESS_MIN = 400
MAPE_TIE = 0.02
RATIO_DISAGREE = 2.0


def _converged(r: ModelResults) -> tuple[bool, list[str]]:
    d, issues = r.diagnostics or {}, []
    if d.get("max_rhat") is not None and d["max_rhat"] > RHAT_MAX:
        issues.append(f"R-hat {d['max_rhat']:.3f} > {RHAT_MAX}")
    if d.get("min_ess") is not None and d["min_ess"] < ESS_MIN:
        issues.append(f"min ESS {d['min_ess']:.0f} < {ESS_MIN}")
    if d.get("divergences"):
        issues.append(f"{d['divergences']} divergences")
    return not issues, issues


def _rank_corr(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _label(r: ModelResults) -> str:
    return f"{r.framework} ({r.run_id})"


def compare(results: list[ModelResults]) -> dict[str, Any]:
    if len(results) < 2:
        raise ValueError("need at least two results to compare")

    windows = {(r.window.get("train_start"), r.window.get("train_end"),
                r.window.get("holdout_start"), r.window.get("holdout_end")) for r in results}
    comparable = len(windows) == 1
    targets = {r.target for r in results}
    if len(targets) > 1:
        comparable = False

    models = []
    for r in results:
        ok, issues = _converged(r)
        models.append({
            "label": _label(r), "framework": r.framework, "run_id": r.run_id,
            "fit": r.fit, "diagnostics": r.diagnostics, "converged": ok,
            "convergence_issues": issues, "baseline_share": r.baseline_share,
            "notes": r.notes,
        })

    common = [c for c in results[0].channels if all(c in r.channels for r in results)]
    only = {_label(r): sorted(set(r.channels) - set(common)) for r in results}

    channels: dict[str, Any] = {}
    for ch in common:
        per = {}
        for r in results:
            s = r.channel_summary.get(ch, {})
            per[_label(r)] = {k: s.get(k) for k in ("spend", "share", "roas", "mroas")}
        roas = [p["roas"] for p in per.values() if p.get("roas") and p["roas"].get("mean") is not None]
        entry: dict[str, Any] = {"by_model": per}
        if len(roas) == len(results):
            lo = max(x["p05"] for x in roas)
            hi = min(x["p95"] for x in roas)
            means = [x["mean"] for x in roas]
            pos = [m for m in means if m > 0]
            ratio = (max(pos) / min(pos)) if len(pos) == len(means) else None
            overlap = lo <= hi
            widths = [(x["p95"] - x["p05"]) / abs(x["mean"]) if x["mean"] else None for x in roas]
            entry.update({
                "intervals_overlap": overlap,
                "mean_ratio": ratio,
                "relative_width": widths,
                "agreement": (
                    "disagree" if not overlap or (ratio is not None and ratio > RATIO_DISAGREE)
                    else "agree-precise" if all(w is not None and w < 1.0 for w in widths)
                    else "agree-uncertain"
                ),
            })
        channels[ch] = entry

    rank = {}
    for a, b in combinations(results, 2):
        ra = [a.channel_summary[c].get("roas", {}).get("mean") for c in common]
        rb = [b.channel_summary[c].get("roas", {}).get("mean") for c in common]
        if all(v is not None for v in ra + rb):
            rank[f"{_label(a)} vs {_label(b)}"] = _rank_corr(ra, rb)

    out = {
        "comparable": comparable,
        "window": results[0].window if comparable else [r.window for r in results],
        "kpi_kind": results[0].kpi_kind,
        "return_label": results[0].return_label,
        "models": models,
        "common_channels": common,
        "channels_only_in": {k: v for k, v in only.items() if v},
        "channels": channels,
        "rank_correlation": rank,
    }
    out["findings"] = _findings(out)
    out["recommendation"] = _recommend(out)
    return out


def _findings(c: dict) -> list[dict[str, str]]:
    f: list[dict[str, str]] = []

    def add(kind: str, text: str, action: str) -> None:
        f.append({"kind": kind, "finding": text, "action": action})

    if not c["comparable"]:
        add("blocking", "The runs use different windows or targets, so their numbers are not comparable.",
            "Refit every framework with the same spec and holdout before comparing.")
        return f
    for m in c["models"]:
        if not m["converged"]:
            add("trust", f"{m['label']} did not converge ({'; '.join(m['convergence_issues'])}).",
                "Do not use its numbers. Refit with more draws or fix identifiability (see /mmm-diagnose).")
    mapes = [(m["fit"].get("mape_holdout"), m["label"]) for m in c["models"] if m["fit"].get("mape_holdout") is not None]
    if len(mapes) >= 2:
        mapes.sort()
        (best, bl), (second, _) = mapes[0], mapes[1]
        if second - best < MAPE_TIE:
            add("forecast", f"Holdout error is a tie (best {best:.1%}, next {second:.1%}).",
                "Forecast accuracy does not separate the models; decide on agreement and plausibility instead.")
        else:
            add("forecast", f"{bl} forecasts the holdout best ({best:.1%} MAPE vs {second:.1%}).",
                "Prefer it for decisions, provided it converged and its attribution is plausible.")
    dis = [ch for ch, e in c["channels"].items() if e.get("agreement") == "disagree"]
    prec = [ch for ch, e in c["channels"].items() if e.get("agreement") == "agree-precise"]
    unc = [ch for ch, e in c["channels"].items() if e.get("agreement") == "agree-uncertain"]
    if dis:
        add("identification", f"The models disagree on {dis}.",
            "The data cannot pin these channels down. Do not move large budgets on them; test them first "
            "(/mmm-experiment-plan).")
    if prec:
        add("robust", f"The models agree, with reasonably tight intervals, on {prec}.",
            "These are the most decision-ready numbers.")
    if unc:
        add("uncertain", f"The models agree on {unc}, but the intervals are wide.",
            "Directionally consistent, not precise. Treat budget changes as tests.")
    bases = [m["baseline_share"].get("mean") for m in c["models"] if m["baseline_share"].get("mean") is not None]
    if len(bases) >= 2 and max(bases) - min(bases) > 0.15:
        add("baseline", f"Baseline share differs by {max(bases) - min(bases):.0%} between models.",
            "The split between media and baseline is not identified; check controls and trend before "
            "trusting any channel number.")
    if any(b is not None and b < 0.2 for b in bases):
        add("baseline", "At least one model attributes more than 80% of the outcome to media.",
            "That is rarely plausible; review priors and missing controls.")
    for pair, rho in c["rank_correlation"].items():
        if rho is not None and rho < 0.5:
            add("ranking", f"{pair}: channel ROAS rankings barely agree (rank correlation {rho:.2f}).",
                "Do not reallocate on the ranking alone.")
    return f


def _recommend(c: dict) -> dict[str, Any]:
    if not c["comparable"]:
        return {"model": None, "reason": "Runs are not comparable."}
    ok = [m for m in c["models"] if m["converged"]]
    if not ok:
        return {"model": None, "reason": "No model converged. Fix convergence before using any result."}
    with_mape = [m for m in ok if m["fit"].get("mape_holdout") is not None]
    if len(with_mape) >= 2:
        with_mape.sort(key=lambda m: m["fit"]["mape_holdout"])
        if with_mape[1]["fit"]["mape_holdout"] - with_mape[0]["fit"]["mape_holdout"] < MAPE_TIE:
            return {"model": None, "reason": "Converged models tie on holdout error. Report both; use channels "
                    "where they agree for decisions."}
        return {"model": with_mape[0]["label"], "reason": "Converged, with the lowest holdout error."}
    return {"model": ok[0]["label"], "reason": "Only converged model with a holdout score."}


# --------------------------------------------------------------------------- #
def _fmt(x: Optional[dict], pct: bool = False, nd: int = 3) -> str:
    if not x or x.get("mean") is None:
        return "—"
    if pct:
        return f"{x['mean']:.0%} [{x['p05']:.0%}, {x['p95']:.0%}]"
    return f"{x['mean']:.{nd}g} [{x['p05']:.{nd}g}, {x['p95']:.{nd}g}]"


def render_comparison(c: dict) -> str:
    lines = ["# Framework comparison", ""]
    if c["comparable"]:
        w = c["window"]
        lines += [f"Training {w['train_start']} → {w['train_end']} ({w['n_train']} periods); "
                  f"holdout {w.get('holdout_start')} → {w.get('holdout_end')} ({w.get('n_holdout')} periods).", ""]
    rec = c["recommendation"]
    lines += [f"**Recommendation:** {rec['model'] or 'no single model'} — {rec['reason']}", ""]
    lines += ["## Findings", ""]
    for x in c["findings"]:
        lines += [f"- **{x['kind']}** — {x['finding']} *{x['action']}*"]
    lines += ["", "## Fit and trust", "",
              "| Model | Holdout MAPE | Holdout wMAPE | Train MAPE | Max R-hat | Min ESS | Divergences | Baseline share |",
              "|---|---|---|---|---|---|---|---|"]
    for m in c["models"]:
        f, d = m["fit"], m["diagnostics"]
        mape = f"{f['mape_holdout']:.1%}" if f.get("mape_holdout") is not None else "—"
        r2 = f"{f['wmape_holdout']:.1%}" if f.get("wmape_holdout") is not None else "—"
        tr = f"{f['mape_train']:.1%}" if f.get("mape_train") is not None else "—"
        rh = f"{d['max_rhat']:.3f}" if d.get("max_rhat") is not None else "—"
        es = f"{d['min_ess']:.0f}" if d.get("min_ess") is not None else "—"
        lines.append(f"| {m['label']}{'' if m['converged'] else ' ⚠️'} | {mape} | {r2} | {tr} | {rh} | {es} | "
                     f"{d.get('divergences', '—')} | {_fmt(m['baseline_share'], pct=True)} |")
    labels = [m["label"] for m in c["models"]]
    lines += ["", f"## {c['return_label']} by channel (mean [90% interval])", "",
              "| Channel | " + " | ".join(labels) + " | Agreement |",
              "|---|" + "---|" * len(labels) + "---|"]
    for ch, e in c["channels"].items():
        cells = [_fmt(e["by_model"][l].get("roas")) for l in labels]
        lines.append(f"| {ch} | " + " | ".join(cells) + f" | {e.get('agreement', '—')} |")
    lines += ["", "## Contribution share by channel", "",
              "| Channel | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for ch, e in c["channels"].items():
        lines.append(f"| {ch} | " + " | ".join(_fmt(e["by_model"][l].get("share"), pct=True) for l in labels) + " |")
    if c["channels_only_in"]:
        lines += ["", "Channels not in every model (excluded above): " +
                  "; ".join(f"{k}: {v}" for k, v in c["channels_only_in"].items())]
    notes = sorted({n for m in c["models"] for n in m["notes"]})
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def run_comparison(results: list[ModelResults], base: str | Path = ".") -> tuple[dict, Path]:
    c = compare(results)
    out = Path(base) / "mmm-workspace" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(c, indent=1, default=str), encoding="utf-8")
    md = out / "comparison.md"
    md.write_text(render_comparison(c), encoding="utf-8")
    return c, md
