"""Utilities for handling target-unit semantics across monetary and non-monetary targets."""
from __future__ import annotations
from typing import Optional
from agent_mmm.spec import TargetUnit, TargetUnitKind


def is_monetary(unit: TargetUnit) -> bool:
    return unit.kind == TargetUnitKind.monetary


def cpa_label(unit: TargetUnit) -> str:
    return f"Cost per {unit.label}"


def roas_label(unit: TargetUnit) -> str:
    if is_monetary(unit):
        code = unit.currency_code or "unit"
        return f"ROAS ({code} return per {code} spent)"
    return cpa_label(unit)


def spend_to_return_ratio(
    spend: float,
    incremental_units: float,
    unit: TargetUnit,
) -> dict:
    """Compute cost efficiency metrics respecting the target unit."""
    if incremental_units == 0:
        return {"cpa": None, "roas": None, "unit_label": unit.label}
    cpa = spend / incremental_units
    roas = None
    if is_monetary(unit):
        roas = incremental_units / spend if spend > 0 else None
    elif unit.value_per_unit is not None:
        roas = (incremental_units * unit.value_per_unit) / spend if spend > 0 else None
    return {"cpa": cpa, "roas": roas, "unit_label": unit.label}
