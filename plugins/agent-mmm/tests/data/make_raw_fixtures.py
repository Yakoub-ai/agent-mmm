"""Regenerate the deliberately messy fixtures in tests/data/raw/.

Each file reproduces a defect that shows up constantly in real marketing data and
that is invisible until it has already corrupted a model. Kept as a script so the
intent is recorded and the set can be extended, but the CSVs are committed so the
tests do not depend on a generator or on a random seed.

    python tests/data/make_raw_fixtures.py

| File | What it is deliberately wrong about |
|---|---|
| `meta_ads_export.csv`   | Long format (one row per day per campaign); money as `£1,234.56` text; a CPC rate column that must never be modelled as spend |
| `google_ads_weekly.csv` | One duplicated week; an unexplained free-text column |
| `finance_sales.csv`     | `dd/mm/yyyy` dates that pandas parses inconsistently by default; Sunday weeks against everything else's Monday; starts five months later than the media |
| `tv_plan_monthly.csv`   | Monthly grain among weekly sources; a negative make-good spend |
| `ooh_geo_panel.csv`     | A geo panel joined to national files; geo codes that need a crosswalk |
| `geo_lookup.csv`        | An undated lookup table, not a time series |
| `readme.md`             | A non-data file that must be skipped |

Together they produce a discovery report with roughly a dozen blocking questions,
which is the realistic outcome of pointing the tool at a real data drop.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "raw"


def main() -> None:
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)

    # Meta-style long export: money as text with a currency symbol, plus a rate column.
    days = pd.date_range("2022-01-03", "2023-12-31", freq="D")
    campaigns = [
        "BR_Prospecting_Q1", "BR_Retargeting_Evergreen",
        "PERF_DPA_Catalog", "BRAND_Awareness_Video",
    ]
    rows = []
    for d in days:
        for c in campaigns:
            spend = max(0.0, rng.normal(800 if "BRAND" in c else 400, 150))
            rows.append({
                "Reporting starts": d.strftime("%Y-%m-%d"),
                "Campaign name": c,
                "Ad Set Name": c + "_set",
                "Amount spent (GBP)": f"£{spend:,.2f}",
                "Impressions": int(spend * rng.uniform(60, 90)),
                "Link clicks": int(spend * rng.uniform(0.8, 1.4)),
                "CPC": round(rng.uniform(0.4, 1.2), 3),
                "Results": int(spend * rng.uniform(0.01, 0.05)),
            })
    pd.DataFrame(rows).to_csv(OUT / "meta_ads_export.csv", index=False)

    # Google Ads weekly: one duplicated week, one unexplained column.
    weeks = pd.date_range("2022-01-03", "2023-12-25", freq="W-MON")
    g = pd.DataFrame({
        "week": weeks.strftime("%Y-%m-%d"),
        "campaign_type": rng.choice(["Search Brand", "Search Generic", "PMax"], len(weeks)),
        "cost": np.round(rng.normal(5200, 900, len(weeks)), 2),
        "impressions": rng.integers(200_000, 400_000, len(weeks)),
        "conversions": rng.integers(300, 900, len(weeks)),
        "search_impr_share": np.round(rng.uniform(0.4, 0.9, len(weeks)), 3),
        "notes_field": rng.choice(["", "budget capped", "creative refresh"], len(weeks)),
    })
    pd.concat([g, g.iloc[[10]]], ignore_index=True).to_csv(
        OUT / "google_ads_weekly.csv", index=False
    )

    # Finance: day-first dates, Sunday weeks, and a later start than the media.
    fw = pd.date_range("2022-06-05", "2023-12-31", freq="W-SUN")
    pd.DataFrame({
        "Week Ending": fw.strftime("%d/%m/%Y"),
        "Net Revenue": np.round(rng.normal(240_000, 30_000, len(fw)), 2),
        "Orders": rng.integers(2000, 4000, len(fw)),
        "Average Selling Price": np.round(rng.normal(72, 6, len(fw)), 2),
        "Returns Value": np.round(rng.normal(9000, 2000, len(fw)), 2),
        "currency": "GBP",
    }).to_csv(OUT / "finance_sales.csv", index=False)

    # TV: monthly among weekly sources, with a negative make-good.
    months = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
    tv = pd.DataFrame({
        "month": months.strftime("%Y-%m"),
        "grp": np.round(rng.uniform(0, 400, len(months)), 1),
        "tv_spend": np.round(rng.normal(120_000, 40_000, len(months)), 2),
    })
    tv.loc[3, "tv_spend"] = -15_000.0
    tv.to_csv(OUT / "tv_plan_monthly.csv", index=False)

    # A geo panel, with codes that do not match the lookup's market names.
    gw = pd.date_range("2022-01-03", "2023-12-25", freq="W-MON")
    regions = ["GB-LON", "GB-MAN", "GB-BIR", "GB-GLA"]
    populations = {"GB-LON": 9_000_000, "GB-MAN": 2_800_000,
                   "GB-BIR": 2_600_000, "GB-GLA": 1_700_000}
    pd.DataFrame([
        {"date": d.strftime("%Y-%m-%d"), "region": r,
         "ooh_spend": round(max(0.0, rng.normal(4000, 1500)), 2),
         "population": populations[r]}
        for d in gw for r in regions
    ]).to_csv(OUT / "ooh_geo_panel.csv", index=False)

    # An undated lookup table.
    pd.DataFrame({
        "market": ["London", "Manchester", "Birmingham", "Glasgow"],
        "region_code": regions,
        "tier": [1, 2, 2, 3],
    }).to_csv(OUT / "geo_lookup.csv", index=False)

    # A non-data file that discovery must skip.
    (OUT / "readme.md").write_text("# notes\nthese are the exports\n")

    print("wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
