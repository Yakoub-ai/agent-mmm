"""Controls / external-factors recommender for MMM."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agent_mmm.spec import MMMSpec
from agent_mmm.workspace import ensure_workspace

# Path to the controls catalog (relative to this file's package root)
_CATALOG_PATH = Path(__file__).parent.parent.parent / "references" / "controls_catalog.yaml"


def _load_catalog() -> dict:
    if not _CATALOG_PATH.exists():
        return {}
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f) or {}


def recommend_controls(
    spec: MMMSpec,
    audit_findings: dict | None = None,
    base: str | Path = ".",
) -> dict[str, Any]:
    """Generate control variable recommendations based on spec + audit findings.

    Returns a dict of recommendations, also writes:
    - ./mmm-workspace/controls/recommended.json
    - ./mmm-workspace/controls/recommendations_report.md
    """
    catalog = _load_catalog()
    industry_lower = spec.industry.lower()
    region_lower = spec.region.lower()

    recommendations: list[dict] = []

    def _add(rec: dict, category: str, reason: str = "") -> None:
        rec = dict(rec)
        rec["category"] = category
        rec["reason"] = reason
        recommendations.append(rec)

    # --- Calendar controls (always) ---
    for item in catalog.get("calendar", []):
        applies = item.get("applies_to", [])
        if "all" in applies or any(ind in industry_lower for ind in [a.lower() for a in applies]):
            _add(item, "calendar", "standard calendar control")

    # --- Events ---
    # COVID: always recommend if data spans 2020-2021
    _add({
        "name": "is_covid_lockdown",
        "description": "Dummy = 1 for COVID-restriction weeks (approx Mar 2020 – Jun 2021)",
        "source": "user_provided_or_generated",
        "importance": "high",
    }, "events", "COVID period often creates a structural break in sales")

    # Structural break from audit
    if audit_findings:
        break_check = audit_findings.get("checks", {}).get("structural_break", {})
        if break_check.get("break_detected"):
            _add({
                "name": "structural_break_dummy",
                "description": "Binary dummy variable marking weeks before/after the detected structural break",
                "source": "generated_from_audit",
                "importance": "high",
            }, "events", f"Structural break detected in target series (p={break_check.get('p_value', '?')})")

    # --- Macro controls by industry ---
    for item in catalog.get("macro", []):
        applies = item.get("applies_to", [])
        if "all" in applies or any(ind in industry_lower for ind in [a.lower() for a in applies]):
            _add(item, "macro", f"macro indicator relevant to {spec.industry}")

    # --- Search trends ---
    for item in catalog.get("search", []):
        _add(item, "search", "organic search interest captures brand-driven demand")

    # --- Industry-specific ---
    for ind_key, items in catalog.get("industry_specific", {}).items():
        if ind_key.lower() in industry_lower or industry_lower in ind_key.lower():
            for item in items:
                # Check region filter
                if "regions" in item:
                    item_regions = [r.lower() for r in item["regions"]]
                    if not any(r in region_lower for r in item_regions):
                        continue
                _add(item, f"industry:{ind_key}", f"specific to {ind_key} industry")

    # Deduplicate by name
    seen: set[str] = set()
    unique_recs = []
    for r in recommendations:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_recs.append(r)

    # Sort: high importance first
    importance_order = {"high": 0, "medium": 1, "low": 2}
    unique_recs.sort(key=lambda x: importance_order.get(x.get("importance", "low"), 2))

    result = {
        "generated_at": datetime.now().isoformat(),
        "industry": spec.industry,
        "region": spec.region,
        "total_recommendations": len(unique_recs),
        "recommendations": unique_recs,
        "existing_controls": spec.control_columns(),
        "note": "Review recommendations and add chosen columns to spec.yaml controls[] then re-run /mmm-analyze-data",
    }

    # Save artifacts
    ws = ensure_workspace(base)
    controls_dir = Path(ws) / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)

    with open(controls_dir / "recommended.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    report = _render_controls_report(result)
    with open(controls_dir / "recommendations_report.md", "w") as f:
        f.write(report)

    return result


def _render_controls_report(result: dict) -> str:
    lines = [
        "# MMM External Factors & Controls Recommendations",
        "",
        f"**Industry**: {result['industry']}  ",
        f"**Region**: {result['region']}  ",
        f"**Generated**: {result['generated_at'][:19]}  ",
        "",
        f"Found **{result['total_recommendations']}** recommended control variables.",
        "",
    ]

    if result["existing_controls"]:
        lines += [
            "## Already in Spec",
            "",
            "The following controls are already configured in your spec.yaml:",
            "",
        ]
        for c in result["existing_controls"]:
            lines.append(f"- `{c}`")
        lines.append("")

    # Group by category
    by_category: dict[str, list] = {}
    for r in result["recommendations"]:
        cat = r.get("category", "other")
        by_category.setdefault(cat, []).append(r)

    category_labels = {
        "calendar": "Calendar Controls",
        "events": "Event Dummies",
        "macro": "Macroeconomic Indicators",
        "search": "Search & Organic Interest",
    }

    for cat, items in by_category.items():
        label = category_labels.get(cat, f"{cat.replace('industry:', 'Industry: ').title()}")
        lines += [f"## {label}", ""]
        lines += ["| Variable | Importance | Source | Reason |",
                  "|----------|------------|--------|--------|"]
        for item in items:
            name = f"`{item['name']}`"
            imp = item.get("importance", "medium")
            imp_label = {"high": "High", "medium": "Medium", "low": "Low"}.get(imp, imp)
            source = item.get("source", "").replace("_", " ")
            reason = item.get("reason", item.get("description", ""))[:80]
            lines.append(f"| {name} | {imp_label} | {source} | {reason} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## How to Add Controls",
        "",
        "1. Generate or obtain the control column data for your date range",
        "2. Add it as a new column in your marketing data CSV",
        "3. Add it to `controls:` in `./mmm-workspace/spec.yaml`:  ",
        "   ```yaml",
        "   controls:",
        "     - column: is_christmas_week",
        "       label: Christmas week",
        "       source: generated",
        "   ```",
        "4. Re-run `/mmm-analyze-data` to verify the updated dataset",
        "",
        "*Generated by agent-mmm controls engine*",
        "",
    ]
    return "\n".join(lines)
