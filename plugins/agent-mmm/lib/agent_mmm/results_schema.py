"""Framework-neutral model results.

Every framework run ends in one ``results.json`` with the same shape, so the
comparison, the charts and the budget optimiser never need to know which
framework produced it.

**All metric arithmetic lives in** :func:`build_results`. Extractors only hand
over raw posterior arrays in original units (expected outcome, channel
contributions, response-curve draws). That way ROAS, shares, holdout error and
marginal ROAS are computed identically for every framework — a comparison is
only fair if the scoring is the same code.

Definitions (all over the **training window**, per posterior draw, then
summarised as mean and 90% interval):

* contribution   — incremental outcome of a channel vs zero spend.
* share          — contribution / total expected outcome.
* ROAS           — contribution / spend. For a non-revenue target this is
                   outcome units per unit of spend (e.g. sales per £).
* marginal ROAS  — (response at 1.1x spend − response at 1.0x) / (0.1 × spend).
* baseline share — 1 − sum of media shares.
* response curve — total incremental outcome when every period's spend in that
                   channel is multiplied by m, other channels unchanged.

Holdout metrics use the posterior-mean expected outcome on periods the model
never saw. Only numpy/pandas are imported, so this runs in every framework env.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCHEMA_VERSION = "1"
DEFAULT_MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0)
MAX_CURVE_DRAWS = 100
LO, HI = 5, 95


def summarize(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "p05": None, "p95": None}
    return {
        "mean": float(np.mean(a)),
        "p05": float(np.percentile(a, LO)),
        "p95": float(np.percentile(a, HI)),
    }


def fit_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, Optional[float]]:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    if actual.size == 0:
        return {"r2": None, "mape": None, "wmape": None}
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    nz = actual != 0
    return {
        "r2": float(1 - np.sum((actual - pred) ** 2) / ss_tot) if ss_tot > 0 else None,
        "mape": float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz]))) if nz.any() else None,
        "wmape": float(np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual))) if np.any(actual) else None,
    }


def holdout_mask(n_periods: int, periods_per_year: int, holdout: Any = "auto",
                 dates: Optional[list] = None, start: Any = None, end: Any = None) -> np.ndarray:
    """Boolean mask, True for holdout periods. Identical for every framework.

    ``holdout``: "auto" (a quarter of a year, at most 15% of the data, at least
    4 periods), an int number of trailing periods, or 0/None for no holdout.
    Explicit ``start``/``end`` dates win when given.
    """
    mask = np.zeros(n_periods, dtype=bool)
    if start is not None and dates is not None:
        d = np.array([str(x)[:10] for x in dates])
        s, e = str(start)[:10], str(end)[:10] if end is not None else "9999-12-31"
        mask = (d >= s) & (d <= e)
        return mask
    if not holdout:
        return mask
    if holdout == "auto":
        k = min(max(4, periods_per_year // 4), int(n_periods * 0.15))
    else:
        k = int(holdout)
    if k <= 0 or k >= n_periods - 8:
        return mask
    mask[-k:] = True
    return mask


def thin(n_draws: int, keep: int = MAX_CURVE_DRAWS) -> np.ndarray:
    return np.unique(np.linspace(0, n_draws - 1, min(keep, n_draws)).round().astype(int))


@dataclass
class ModelResults:
    framework: str
    run_id: str
    target: str
    target_label: str
    kpi_kind: str                       # revenue | non_revenue
    channels: list[str]
    window: dict[str, Any]
    fit: dict[str, Any]
    diagnostics: dict[str, Any]
    channel_summary: dict[str, dict[str, Any]]
    baseline_share: dict[str, Any]
    series: dict[str, Any]
    curves: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def return_label(self) -> str:
        return "ROAS" if self.kpi_kind == "revenue" else f"{self.target_label} per unit spend"

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelResults":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(data.get("schema_version")) != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema {data.get('schema_version')} != {SCHEMA_VERSION}")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _clean(o: Any) -> Any:
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def build_results(
    *,
    framework: str,
    run_id: str,
    target: str,
    target_label: str,
    kpi_kind: str,
    channels: list[str],
    dates: list[str],
    actual: np.ndarray,                 # (T,)
    expected: np.ndarray,               # (S, T) expected outcome per draw, all periods
    contributions: np.ndarray,          # (S, T_train, C) training window only
    spend: np.ndarray,                  # (T, C), NaN for channels without spend
    holdout: np.ndarray,                # (T,) bool
    curve_draws: dict[str, np.ndarray], # ch -> (S', M) total response over training window
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS,
    diagnostics: Optional[dict] = None,
    provenance: Optional[dict] = None,
    notes: Optional[list[str]] = None,
) -> ModelResults:
    actual = np.asarray(actual, float)
    expected = np.asarray(expected, float)
    contributions = np.asarray(contributions, float)
    spend = np.asarray(spend, float)
    holdout = np.asarray(holdout, bool)
    train = ~holdout
    T, C = len(dates), len(channels)
    n_train = int(train.sum())
    if expected.shape[1] != T or actual.shape != (T,) or spend.shape != (T, C):
        raise ValueError("expected/actual/spend do not match the number of periods and channels")
    if contributions.shape[1:] != (n_train, C):
        raise ValueError(f"contributions must be (draws, {n_train}, {C}); got {contributions.shape}")

    exp_mean = expected.mean(axis=0)
    total_expected_train = expected[:, train].sum(axis=1)               # (S,)
    contrib_total = contributions.sum(axis=1)                           # (S, C)
    spend_train = np.nansum(spend[train], axis=0)                       # (C,)
    has_spend = ~np.all(np.isnan(spend), axis=0)

    m = list(multipliers)
    i1 = m.index(1.0) if 1.0 in m else None
    i11 = m.index(1.1) if 1.1 in m else None

    summary: dict[str, dict[str, Any]] = {}
    for j, ch in enumerate(channels):
        c = contrib_total[:, j]
        entry: dict[str, Any] = {
            "spend": float(spend_train[j]) if has_spend[j] else None,
            "contribution": summarize(c),
            "share": summarize(c / total_expected_train),
        }
        if has_spend[j] and spend_train[j] > 0:
            entry["roas"] = summarize(c / spend_train[j])
            cd = curve_draws.get(ch)
            if cd is not None and i1 is not None and i11 is not None:
                entry["mroas"] = summarize((cd[:, i11] - cd[:, i1]) / (0.1 * spend_train[j]))
        summary[ch] = entry

    media_share = contrib_total.sum(axis=1) / total_expected_train
    train_fit = fit_metrics(actual[train], exp_mean[train])
    hold_fit = fit_metrics(actual[holdout], exp_mean[holdout]) if holdout.any() else {}

    curves: dict[str, Any] = {"multipliers": m, "channels": {}}
    for j, ch in enumerate(channels):
        cd = curve_draws.get(ch)
        if cd is None:
            continue
        cd = np.asarray(cd, float)[thin(len(cd))]
        curves["channels"][ch] = {
            "spend": [float(spend_train[j] * x) if has_spend[j] else None for x in m],
            "mean": cd.mean(axis=0).tolist(),
            "p05": np.percentile(cd, LO, axis=0).tolist(),
            "p95": np.percentile(cd, HI, axis=0).tolist(),
            "draws": cd.tolist(),
        }

    cmean = contributions.mean(axis=0)                                  # (T_train, C)
    tdates = [d for d, h in zip(dates, holdout) if not h]
    return ModelResults(
        framework=framework,
        run_id=run_id,
        target=target,
        target_label=target_label,
        kpi_kind=kpi_kind,
        channels=list(channels),
        window={
            "train_start": tdates[0], "train_end": tdates[-1], "n_train": n_train,
            "holdout_start": next((d for d, h in zip(dates, holdout) if h), None),
            "holdout_end": next((d for d, h in zip(reversed(dates), holdout[::-1]) if h), None),
            "n_holdout": int(holdout.sum()),
        },
        fit={
            "r2_train": train_fit["r2"], "mape_train": train_fit["mape"],
            "r2_holdout": hold_fit.get("r2"), "mape_holdout": hold_fit.get("mape"),
            "wmape_holdout": hold_fit.get("wmape"),
        },
        diagnostics=diagnostics or {},
        channel_summary=summary,
        baseline_share=summarize(1 - media_share),
        series={
            "dates": list(dates),
            "holdout": holdout.tolist(),
            "actual": actual.tolist(),
            "expected_mean": exp_mean.tolist(),
            "expected_p05": np.percentile(expected, LO, axis=0).tolist(),
            "expected_p95": np.percentile(expected, HI, axis=0).tolist(),
            "train_dates": tdates,
            "contributions": {ch: cmean[:, j].tolist() for j, ch in enumerate(channels)},
            "baseline": (exp_mean[train] - cmean.sum(axis=1)).tolist(),
        },
        curves=curves,
        provenance=provenance or {},
        notes=list(notes or []),
    )


def load_all(runs_dir: str | Path) -> list[ModelResults]:
    """Every results.json under runs/, oldest first. Invalid files are skipped."""
    out = []
    for p in sorted(Path(runs_dir).glob("*/results.json")):
        try:
            out.append(ModelResults.load(p))
        except (ValueError, OSError, TypeError):
            continue
    return out
