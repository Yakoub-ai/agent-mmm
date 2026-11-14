"""Charts and reports from framework-neutral results.

Three outputs, all from ``ModelResults`` (and optionally a comparison):

* **PNG charts** (matplotlib) — for decks and documents.
* **Interactive HTML report** (plotly) — one self-contained file. plotly.js is
  inlined, nothing loads from a CDN, and every piece of text is HTML-escaped
  (channel names, notes and findings come from user data).
* **Markdown summary** — the numbers and findings in plain text.

Visual rules: a fixed categorical order where colour follows the entity
(frameworks and channels always keep their colour), a neutral grey for the
baseline, intervals on every estimate, one y-axis per chart, recessive grid.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from agent_mmm.compare import _converged
from agent_mmm.results_schema import ModelResults

# Validated categorical order (light surface).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9a9893"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"
FRAMEWORK_SLOT = {"pymc-marketing": 0, "meridian": 1, "robyn": 2}
MAX_STACK = 7


# --------------------------------------------------------------------------- #
# Colour assignment: follows the entity, never the rank
# --------------------------------------------------------------------------- #
def model_colors(results: list[ModelResults]) -> dict[str, str]:
    """Each framework keeps its slot; a second run of the same framework takes the next free slot."""
    used: set[int] = set()
    out: dict[str, str] = {}
    for r in results:
        slot = FRAMEWORK_SLOT.get(r.framework)
        if slot is None or slot in used:
            free = [i for i in range(len(PALETTE)) if i not in used]
            slot = free[0] if free else len(used) % len(PALETTE)
        used.add(slot)
        out[r.run_id] = PALETTE[slot]
    return out


def channel_colors(channels: list[str]) -> dict[str, str]:
    return {ch: (PALETTE[i] if i < len(PALETTE) else NEUTRAL) for i, ch in enumerate(channels)}


def _label(r: ModelResults) -> str:
    return f"{r.framework} · {r.run_id}"


def _fmt(x: Optional[float], kind: str = "num") -> str:
    if x is None:
        return "—"
    if kind == "pct":
        return f"{x:.1%}"
    return f"{x:,.3g}" if abs(x) < 1000 else f"{x:,.0f}"


# --------------------------------------------------------------------------- #
# PNG (matplotlib)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
    })
    return plt


def _human(x: float, _pos=None) -> str:
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.1f}B".replace(".0B", "B")
    if a >= 1e6:
        return f"{x / 1e6:.1f}M".replace(".0M", "M")
    if a >= 1e5:
        return f"{x / 1e3:.0f}K"
    if a >= 1e4:
        return f"{x / 1e3:.1f}K".replace(".0K", "K")
    return f"{x:,.0f}" if a >= 10 else f"{x:g}"


def _legend_below(ax, n: int) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncols=max(1, min(n, 3)))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def png_fit(r: ModelResults, path: Path, color: str) -> Path:
    plt = _plt()
    s = r.series
    d = pd.to_datetime(s["dates"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(d, s["expected_p05"], s["expected_p95"], color=color, alpha=0.18, lw=0, label="90% interval")
    ax.plot(d, s["actual"], color=TEXT, lw=1.5, label="Actual")
    ax.plot(d, s["expected_mean"], color=color, label="Model")
    if any(s["holdout"]):
        h = [x for x, f in zip(d, s["holdout"]) if f]
        ax.axvspan(h[0], h[-1], color=NEUTRAL, alpha=0.12, lw=0, label="Holdout")
    mape = r.fit.get("mape_holdout")
    ax.set_title(f"Actual vs model — {_label(r)}" + (f" (holdout MAPE {mape:.1%})" if mape is not None else ""))
    ax.set_ylabel(r.target_label)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    _legend_below(ax, 4)
    return _save(fig, path)


def _stack_parts(r: ModelResults) -> tuple[list[str], dict[str, np.ndarray]]:
    contrib = {ch: np.asarray(v, float) for ch, v in r.series["contributions"].items()}
    order = sorted(contrib, key=lambda c: -contrib[c].sum())
    keep, rest = order[:MAX_STACK], order[MAX_STACK:]
    parts = {ch: contrib[ch] for ch in keep}
    if rest:
        parts["Other channels"] = sum(contrib[c] for c in rest)
    return [c for c in r.channels if c in keep] + (["Other channels"] if rest else []), parts


def png_decomposition(r: ModelResults, path: Path) -> Path:
    plt = _plt()
    d = pd.to_datetime(r.series["train_dates"])
    names, parts = _stack_parts(r)
    colors = channel_colors(r.channels)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    base = np.asarray(r.series["baseline"], float)
    ys = [base] + [parts[n] for n in names]
    cs = [NEUTRAL] + [colors.get(n, NEUTRAL) for n in names]
    ax.stackplot(d, *ys, labels=["Baseline (non-media)"] + names, colors=cs, alpha=0.9,
                 edgecolor=SURFACE, linewidth=0.6)
    ax.set_title(f"Where the outcome comes from — {_label(r)}")
    ax.set_ylabel(r.target_label)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def _dot_whisker(ax, labels, stats, color, offset=0.0, label=None, marker="o"):
    y = np.arange(len(labels)) + offset
    m = [s["mean"] if s and s.get("mean") is not None else np.nan for s in stats]
    lo = [s["p05"] if s and s.get("p05") is not None else np.nan for s in stats]
    hi = [s["p95"] if s and s.get("p95") is not None else np.nan for s in stats]
    ax.hlines(y, lo, hi, color=color, lw=2)
    ax.plot(m, y, marker, color=color, ms=8, mec=SURFACE, mew=1.5, ls="none", label=label)


def png_roas(r: ModelResults, path: Path, color: str) -> Path:
    plt = _plt()
    chans = [c for c in r.channels if r.channel_summary[c].get("roas")]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(chans) + 1.5))
    _dot_whisker(ax, chans, [r.channel_summary[c]["roas"] for c in chans], color, label="Average")
    if any(r.channel_summary[c].get("mroas") for c in chans):
        _dot_whisker(ax, chans, [r.channel_summary[c].get("mroas") for c in chans], NEUTRAL,
                     offset=-0.25, label="Marginal (next unit)", marker="D")
    ax.set_yticks(range(len(chans)), chans)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.set_title(f"{r.return_label}, 90% interval — {_label(r)}")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    _legend_below(ax, 2)
    return _save(fig, path)


def png_share(r: ModelResults, path: Path) -> Path:
    plt = _plt()
    colors = channel_colors(r.channels)
    rows = [("Baseline (non-media)", r.baseline_share, NEUTRAL)] + \
           [(c, r.channel_summary[c]["share"], colors[c]) for c in r.channels]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    for i, (name, s, col) in enumerate(rows):
        if s.get("mean") is None:
            continue
        ax.barh(i, s["mean"], color=col, height=0.6)
        ax.hlines(i, s["p05"], s["p95"], color=TEXT, lw=1.2)
        ax.text(max(s["p95"], s["mean"]) + 0.01, i, f"{s['mean']:.0%}", va="center", color=TEXT2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r_[0] for r_ in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"Share of outcome, 90% interval — {_label(r)}")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def png_curves(r: ModelResults, path: Path, color: str) -> Path:
    plt = _plt()
    chans = [c for c in r.channels if c in r.curves["channels"] and r.curves["channels"][c]["spend"][0] is not None]
    if not chans:
        return path
    n = len(chans)
    cols = min(3, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4 * cols, 3.2 * rows_), squeeze=False)
    mult = r.curves["multipliers"]
    i1 = mult.index(1.0) if 1.0 in mult else None
    for ax, ch in zip(axes.flat, chans):
        c = r.curves["channels"][ch]
        ax.fill_between(c["spend"], c["p05"], c["p95"], color=color, alpha=0.18, lw=0)
        ax.plot(c["spend"], c["mean"], color=color)
        if i1 is not None:
            ax.plot(c["spend"][i1], c["mean"][i1], "o", color=color, ms=8, mec=SURFACE, mew=1.5)
            ax.annotate("current", (c["spend"][i1], c["mean"][i1]), textcoords="offset points",
                        xytext=(6, -12), color=TEXT2, fontsize=8)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("Spend over training window")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Response curves (incremental {r.target_label}) — {_label(r)}", x=0.01, ha="left",
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)


def png_compare_metric(results: list[ModelResults], metric: str, path: Path, title: str) -> Path:
    plt = _plt()
    colors = model_colors(results)
    chans = [c for c in results[0].channels if all(c in r.channels for r in results)]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(chans) + 1.5))
    k = len(results)
    for i, r in enumerate(results):
        off = (i - (k - 1) / 2) * 0.22
        _dot_whisker(ax, chans, [r.channel_summary[c].get(metric) for c in chans], colors[r.run_id],
                     offset=off, label=_label(r))
    ax.set_yticks(range(len(chans)), chans)
    ax.invert_yaxis()
    if metric == "share":
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _legend_below(ax, len(results))
    return _save(fig, path)


def png_compare_fit(results: list[ModelResults], path: Path) -> Path:
    plt = _plt()
    colors = model_colors(results)
    s0 = results[0].series
    d = pd.to_datetime(s0["dates"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d, s0["actual"], color=TEXT, lw=1.5, label="Actual")
    for r in results:
        m = r.fit.get("mape_holdout")
        ax.plot(pd.to_datetime(r.series["dates"]), r.series["expected_mean"], color=colors[r.run_id],
                label=_label(r) + (f" (holdout MAPE {m:.1%})" if m is not None else ""))
    if any(s0["holdout"]):
        h = [x for x, f in zip(d, s0["holdout"]) if f]
        ax.axvspan(h[0], h[-1], color=NEUTRAL, alpha=0.12, lw=0, label="Holdout")
    ax.set_title("Actual vs each model")
    ax.set_ylabel(results[0].target_label)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_human))
    _legend_below(ax, len(results) + 2)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Interactive HTML (plotly)
# --------------------------------------------------------------------------- #
def _layout(fig, title: str, ytitle: str = "", height: int = 380):
    fig.update_layout(
        title=dict(text=html.escape(title), x=0, font=dict(size=15, color=TEXT)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=TEXT2, size=12),
        margin=dict(l=60, r=20, t=50, b=40), hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_text=html.escape(ytitle))
    return fig


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def fig_fit(results: list[ModelResults]):
    import plotly.graph_objects as go

    colors = model_colors(results)
    s0 = results[0].series
    fig = go.Figure()
    for r in results:
        s, c, name = r.series, colors[r.run_id], html.escape(_label(r))
        fig.add_trace(go.Scatter(x=s["dates"] + s["dates"][::-1], y=s["expected_p95"] + s["expected_p05"][::-1],
                                 fill="toself", fillcolor=_rgba(c, 0.15), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, legendgroup=name))
        fig.add_trace(go.Scatter(x=s["dates"], y=s["expected_mean"], name=name, line=dict(color=c, width=2),
                                 legendgroup=name))
    fig.add_trace(go.Scatter(x=s0["dates"], y=s0["actual"], name="Actual", line=dict(color=TEXT, width=1.5)))
    if any(s0["holdout"]):
        h = [x for x, f in zip(s0["dates"], s0["holdout"]) if f]
        fig.add_vrect(x0=h[0], x1=h[-1], fillcolor=NEUTRAL, opacity=0.12, line_width=0,
                      annotation_text="holdout", annotation_position="top left")
    return _layout(fig, "Actual vs model (90% interval)", results[0].target_label)


def fig_decomposition(r: ModelResults):
    import plotly.graph_objects as go

    names, parts = _stack_parts(r)
    colors = channel_colors(r.channels)
    d = r.series["train_dates"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=r.series["baseline"], name="Baseline (non-media)", stackgroup="one",
                             line=dict(width=0.5, color=SURFACE), fillcolor=NEUTRAL))
    for n in names:
        fig.add_trace(go.Scatter(x=d, y=parts[n].tolist(), name=html.escape(n), stackgroup="one",
                                 line=dict(width=0.5, color=SURFACE), fillcolor=colors.get(n, NEUTRAL)))
    return _layout(fig, f"Where the outcome comes from — {_label(r)}", r.target_label, 420)


def fig_compare(results: list[ModelResults], metric: str, title: str):
    import plotly.graph_objects as go

    colors = model_colors(results)
    chans = [c for c in results[0].channels if all(c in r.channels for r in results)]
    fig = go.Figure()
    k = len(results)
    for i, r in enumerate(results):
        st = [r.channel_summary[c].get(metric) or {} for c in chans]
        m = [s.get("mean") for s in st]
        off = (i - (k - 1) / 2) * 0.22
        fig.add_trace(go.Scatter(
            x=m, y=[j + off for j in range(len(chans))], mode="markers", name=html.escape(_label(r)),
            customdata=[html.escape(c) for c in chans],
            marker=dict(color=colors[r.run_id], size=10, line=dict(color=SURFACE, width=1.5)),
            error_x=dict(type="data", symmetric=False, thickness=2, width=0, color=colors[r.run_id],
                         array=[(s.get("p95") or 0) - (s.get("mean") or 0) for s in st],
                         arrayminus=[(s.get("mean") or 0) - (s.get("p05") or 0) for s in st]),
            hovertemplate="%{customdata}: %{x:.3g}<extra>" + html.escape(_label(r)) + "</extra>",
        ))
    fig = _layout(fig, title, "", 100 + 60 * len(chans))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=list(range(len(chans))),
                     ticktext=[html.escape(c) for c in chans], showgrid=False)
    if metric == "share":
        fig.update_xaxes(tickformat=".0%")
    return fig


def fig_curves(results: list[ModelResults]):
    import plotly.graph_objects as go

    colors = model_colors(results)
    chans = [c for c in results[0].channels if any(c in r.curves["channels"] for r in results)]
    fig = go.Figure()
    buttons = []
    per_channel = []
    for ch in chans:
        idx = []
        for r in results:
            c = r.curves["channels"].get(ch)
            if not c or c["spend"][0] is None:
                continue
            col, name = colors[r.run_id], html.escape(_label(r))
            fig.add_trace(go.Scatter(x=c["spend"] + c["spend"][::-1], y=c["p95"] + c["p05"][::-1], fill="toself",
                                     fillcolor=_rgba(col, 0.15), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, visible=False))
            fig.add_trace(go.Scatter(x=c["spend"], y=c["mean"], name=name, line=dict(color=col, width=2),
                                     visible=False, hovertemplate="spend %{x:,.0f}: %{y:,.0f}<extra>" + name + "</extra>"))
            idx += [len(fig.data) - 2, len(fig.data) - 1]
        per_channel.append(idx)
    for i, ch in enumerate(chans):
        vis = [False] * len(fig.data)
        for j in per_channel[i]:
            vis[j] = True
        buttons.append(dict(label=html.escape(ch), method="update", args=[{"visible": vis}]))
    if per_channel and per_channel[0]:
        for j in per_channel[0]:
            fig.data[j].visible = True
    fig = _layout(fig, "Response curves — incremental outcome vs spend (90% interval)", results[0].target_label, 420)
    fig.update_layout(hovermode="closest", updatemenus=[dict(buttons=buttons, x=1, xanchor="right", y=1.18)])
    fig.update_xaxes(title_text="Spend over training window")
    return fig


_CSS = f"""
:root {{ color-scheme: light; }}
body {{ margin: 0; background: {SURFACE}; color: {TEXT}; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }} h2 {{ font-size: 19px; margin: 36px 0 8px; }}
.muted {{ color: {TEXT2}; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }}
.tile {{ border: 1px solid {GRID}; border-radius: 10px; padding: 12px 14px; background: #fff; }}
.tile b {{ display: block; font-size: 13px; color: {TEXT2}; font-weight: 600; }}
.tile .v {{ font-size: 22px; font-weight: 700; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }}
.findings li {{ margin: 6px 0; }} .kind {{ font-weight: 700; text-transform: capitalize; }}
.warn {{ color: #a33b16; font-weight: 600; }}
.chart {{ border: 1px solid {GRID}; border-radius: 10px; background: {SURFACE}; padding: 4px; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 8px 0; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid {GRID}; }}
th {{ color: {TEXT2}; font-weight: 600; }} td.n {{ font-variant-numeric: tabular-nums; }}
details {{ margin: 8px 0; }} summary {{ cursor: pointer; color: {TEXT2}; }}
.scroll {{ overflow-x: auto; }}
"""


def _iv(s: Optional[dict], kind: str = "num") -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{_fmt(s['mean'], kind)} <span class='muted'>[{_fmt(s['p05'], kind)}, {_fmt(s['p95'], kind)}]</span>"


def _channel_table(r: ModelResults) -> str:
    rows = []
    for c in r.channels:
        s = r.channel_summary[c]
        rows.append(f"<tr><td>{html.escape(c)}</td><td class='n'>{_fmt(s.get('spend'))}</td>"
                    f"<td class='n'>{_iv(s.get('share'), 'pct')}</td><td class='n'>{_iv(s.get('roas'))}</td>"
                    f"<td class='n'>{_iv(s.get('mroas'))}</td></tr>")
    return ("<div class='scroll'><table><tr><th>Channel</th><th>Spend</th><th>Share</th>"
            f"<th>{html.escape(r.return_label)}</th><th>Marginal</th></tr>" + "".join(rows) + "</table></div>")


def build_html(results: list[ModelResults], comparison: Optional[dict] = None, title: str = "MMM results") -> str:
    from plotly.offline import get_plotlyjs

    cfg = {"displaylogo": False, "responsive": True}
    colors = model_colors(results)
    figs = [fig_fit(results)]
    if len(results) > 1:
        figs += [fig_compare(results, "roas", f"{results[0].return_label} by channel (90% interval)"),
                 fig_compare(results, "share", "Share of outcome by channel (90% interval)")]
    figs.append(fig_curves(results))
    figs += [fig_decomposition(r) for r in results]
    w = results[0].window

    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p class='muted'>Target: {html.escape(results[0].target_label)} · training "
             f"{html.escape(str(w.get('train_start')))} → {html.escape(str(w.get('train_end')))}"
             + (f" · holdout {w.get('n_holdout')} periods" if w.get("n_holdout") else "") + "</p>",
             "<div class='tiles'>"]
    for r in results:
        conv, issues = _converged(r)   # same rule as the comparison, so tiles and findings agree
        parts.append(
            f"<div class='tile'><b><span class='swatch' style='background:{colors[r.run_id]}'></span>"
            f"{html.escape(_label(r))}</b>"
            f"<div class='v'>{_fmt(r.fit.get('mape_holdout'), 'pct')}</div><div class='muted'>holdout MAPE</div>"
            f"<div>Baseline share {_iv(r.baseline_share, 'pct')}</div>"
            + ("" if conv else f"<div class='warn'>⚠ not converged: {html.escape('; '.join(issues))}</div>")
            + "</div>")
    parts.append("</div>")

    if comparison:
        rec = comparison.get("recommendation", {})
        parts += ["<h2>Findings</h2>",
                  f"<p><b>Recommendation:</b> {html.escape(str(rec.get('model') or 'no single model'))} — "
                  f"{html.escape(str(rec.get('reason', '')))}</p><ul class='findings'>"]
        parts += [f"<li><span class='kind'>{html.escape(f['kind'])}</span> — {html.escape(f['finding'])} "
                  f"<span class='muted'>{html.escape(f['action'])}</span></li>" for f in comparison.get("findings", [])]
        parts.append("</ul>")

    parts.append("<h2>Charts</h2>")
    for f in figs:
        parts.append(f"<div class='chart'>{f.to_html(full_html=False, include_plotlyjs=False, config=cfg)}</div>")

    parts.append("<h2>Channel tables</h2>")
    for r in results:
        parts += [f"<h3>{html.escape(_label(r))}</h3>", _channel_table(r)]
    notes = sorted({n for r in results for n in r.notes})
    if notes:
        parts += ["<h2>Notes</h2><ul>"] + [f"<li>{html.escape(n)}</li>" for n in notes] + ["</ul>"]
    parts.append("<p class='muted'>Intervals are 90% posterior intervals. Marginal = return on the next unit of "
                 "spend at current levels. Contribution and ROAS are over the training window.</p>")

    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{_CSS}</style>"
            f"<script>{get_plotlyjs()}</script></head><body><main>" + "".join(parts) + "</main></body></html>")


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #
def _miv(s: Optional[dict], kind: str = "num") -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{_fmt(s['mean'], kind)} [{_fmt(s['p05'], kind)}, {_fmt(s['p95'], kind)}]"


def build_markdown(results: list[ModelResults], figures: dict[str, list[str]],
                   comparison: Optional[dict] = None) -> str:
    w = results[0].window
    lines = ["# MMM results summary", "",
             f"Target **{results[0].target_label}** · training {w.get('train_start')} → {w.get('train_end')}"
             + (f" · holdout {w.get('n_holdout')} periods" if w.get("n_holdout") else ""), ""]
    if comparison:
        rec = comparison.get("recommendation", {})
        lines += ["## Findings", "", f"**Recommendation:** {rec.get('model') or 'no single model'} — {rec.get('reason')}", ""]
        lines += [f"- **{f['kind']}** — {f['finding']} *{f['action']}*" for f in comparison.get("findings", [])]
        lines.append("")
    for r in results:
        d = r.diagnostics
        lines += [f"## {_label(r)}", "",
                  f"- Holdout MAPE {_fmt(r.fit.get('mape_holdout'), 'pct')} · train MAPE {_fmt(r.fit.get('mape_train'), 'pct')}",
                  f"- Baseline share {_miv(r.baseline_share, 'pct')}",
                  f"- Max R-hat {_fmt(d.get('max_rhat'))} · min ESS {_fmt(d.get('min_ess'))} · divergences {d.get('divergences', '—')}",
                  "", f"| Channel | Spend | Share | {r.return_label} | Marginal |", "|---|---|---|---|---|"]
        for c in r.channels:
            s = r.channel_summary[c]
            lines.append(f"| {c} | {_fmt(s.get('spend'))} | {_miv(s.get('share'), 'pct')} | {_miv(s.get('roas'))} | {_miv(s.get('mroas'))} |")
        lines += [""] + [f"![{Path(p).stem}]({p})" for p in figures.get(r.run_id, [])] + [""]
    if figures.get("comparison"):
        lines += ["## Comparison charts", ""] + [f"![{Path(p).stem}]({p})" for p in figures["comparison"]] + [""]
    lines += ["Intervals are 90% posterior intervals; contribution and ROAS cover the training window.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_report(results: list[ModelResults], comparison: Optional[dict] = None, base: str | Path = ".",
                 title: str = "MMM results") -> dict[str, Any]:
    if not results:
        raise ValueError("no results to report")
    out = Path(base) / "mmm-workspace" / "reports"
    fig_dir = out / "figures"
    colors = model_colors(results)
    figures: dict[str, list[str]] = {}
    for r in results:
        d = fig_dir / r.run_id
        c = colors[r.run_id]
        paths = [png_fit(r, d / "fit_holdout.png", c), png_decomposition(r, d / "decomposition.png"),
                 png_share(r, d / "contribution_share.png"), png_roas(r, d / "roas.png", c),
                 png_curves(r, d / "response_curves.png", c)]
        figures[r.run_id] = [str(p.relative_to(out)) for p in paths if p.exists()]
    if len(results) > 1:
        d = fig_dir / "comparison"
        paths = [png_compare_fit(results, d / "fit_compare.png"),
                 png_compare_metric(results, "roas", d / "roas_compare.png",
                                    f"{results[0].return_label} by channel, 90% interval"),
                 png_compare_metric(results, "share", d / "share_compare.png", "Share of outcome by channel, 90% interval")]
        figures["comparison"] = [str(p.relative_to(out)) for p in paths]
    html_path = out / "mmm_report.html"
    html_path.write_text(build_html(results, comparison, title), encoding="utf-8")
    md_path = out / "summary.md"
    md_path.write_text(build_markdown(results, figures, comparison), encoding="utf-8")
    return {"html": str(html_path), "markdown": str(md_path),
            "figures": {k: [str(out / p) for p in v] for k, v in figures.items()}}
