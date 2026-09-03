"""End-to-end: the messy fixtures must flow through discovery, taxonomy, pivoting,
reconciliation, experiment design and deck generation without manual intervention
between the stages.

Each module is unit-tested elsewhere. This file exists because modules that pass
individually routinely fail to compose — a column name that changes shape between
stages, a date that survives one parse and not the next, a frame that is wide where
the next stage expects long.
"""
from pathlib import Path

import pandas as pd
import pytest

from agent_mmm.discovery import discover
from agent_mmm.experiments import (
    ChannelTestCandidate,
    build_roadmap,
    estimate_geo_cv,
    estimate_pre_period_correlation,
    render_roadmap,
)
from agent_mmm.reconciliation import reconcile_spend
from agent_mmm.reports.deck import AUDIENCES, build_deck, render_deck_markdown
from agent_mmm.taxonomy import apply_rules, build_taxonomy, pivot_to_channels, suggest_rules

RAW = Path(__file__).parent / "data" / "raw"


@pytest.fixture(scope="module")
def discovery_report():
    return discover(RAW)


@pytest.fixture(scope="module")
def meta_long() -> pd.DataFrame:
    df = pd.read_csv(RAW / "meta_ads_export.csv")
    # The discovery report flags this column as text-that-is-really-money.
    df["spend"] = df["Amount spent (GBP)"].str.replace(r"[^\d.\-]", "", regex=True).astype(float)
    return df


class TestDiscoveryToTaxonomy:
    def test_discovery_identifies_the_long_file_and_its_name_column(self, discovery_report):
        meta = next(f for f in discovery_report.files if f.name == "meta_ads_export.csv")
        assert meta.shape_kind == "long_transactional"
        entity_cols = [c.name for c in meta.columns_with_role("entity")]
        assert "Campaign name" in entity_cols

    def test_discovery_flags_the_currency_column_taxonomy_then_needs(self, discovery_report):
        meta = next(f for f in discovery_report.files if f.name == "meta_ads_export.csv")
        spend = meta.column("Amount spent (GBP)")
        assert spend.numeric_after_cleaning
        assert spend.detected_currency == "GBP"

    def test_taxonomy_maps_the_discovered_campaign_column(self, meta_long):
        result = build_taxonomy(meta_long, name_column="Campaign name", spend_column="spend")
        assert result.ok
        assert result.coverage_spend == pytest.approx(1.0)
        assert len(result.channel_spend) >= 3


class TestTaxonomyToDataset:
    def test_pivot_produces_a_weekly_frame_ready_to_join(self, meta_long):
        rules = suggest_rules(meta_long["Campaign name"].unique())
        df = meta_long.assign(
            channel=apply_rules(meta_long, rules, name_column="Campaign name")
        )
        wide = pivot_to_channels(
            df,
            date_column="Reporting starts",
            channel_column="channel",
            value_columns={"spend": "sum", "Impressions": "sum"},
            freq="W-MON",
        )
        assert wide["Reporting starts"].is_unique
        assert len(wide) == pytest.approx(104, abs=3)      # two years of weeks
        spend_cols = [c for c in wide.columns if c.endswith("_spend")]
        assert spend_cols
        # Daily spend must survive aggregation to weekly exactly.
        assert wide[spend_cols].to_numpy().sum() == pytest.approx(df["spend"].sum())

    def test_pivoted_frame_reconciles_against_a_matching_ledger(self, meta_long):
        rules = suggest_rules(meta_long["Campaign name"].unique())
        df = meta_long.assign(
            channel=apply_rules(meta_long, rules, name_column="Campaign name")
        )
        wide = pivot_to_channels(
            df, date_column="Reporting starts", channel_column="channel",
            value_columns={"spend": "sum"}, freq="W-MON",
        ).rename(columns={"Reporting starts": "date"})

        # A ledger carrying a 12% agency fee on top of platform-reported spend.
        spend_cols = [c for c in wide.columns if c.endswith("_spend")]
        ledger = pd.DataFrame({
            "date": wide["date"],
            "media_cost": wide[spend_cols].sum(axis=1) / 0.88,
        })

        rec = reconcile_spend(wide, ledger, modelled_spend_columns=spend_cols,
                              finance_spend_column="media_cost")
        assert rec.pattern == "constant_shortfall"
        assert not rec.ok
        assert any("agency fees" in w for w in rec.warnings)


class TestDatasetToExperiments:
    def test_geo_panel_supports_a_powered_roadmap(self):
        panel = pd.read_csv(RAW / "ooh_geo_panel.csv")
        panel["revenue"] = panel["ooh_spend"] * 12 + panel["population"] / 1000

        cv = estimate_geo_cv(panel, geo_column="region", target_column="revenue")
        rho = estimate_pre_period_correlation(
            panel, geo_column="region", target_column="revenue", date_column="date"
        )
        assert cv > 0
        assert rho > 0.5

        roadmap = build_roadmap(
            [
                ChannelTestCandidate("search_brand", spend=900_000, contribution_share=0.12,
                                     roas_point=8.2, roas_ci_low=3.0, roas_ci_high=14.0,
                                     always_on=True),
                ChannelTestCandidate("ooh", spend=400_000, contribution_share=0.05,
                                     roas_point=1.1, roas_ci_low=0.3, roas_ci_high=3.0),
            ],
            n_geos_available=int(panel["region"].nunique()),
            geo_cv=cv,
            pre_period_correlation=rho,
            duration_periods=8,
        )
        assert roadmap.plans
        # Four geos is far too few; the roadmap must say so rather than proceed.
        assert any("geos available" in w for w in roadmap.warnings)
        assert render_roadmap(roadmap).startswith("# Experiment Roadmap")

    def test_roadmap_questions_flow_into_the_deck(self):
        roadmap = build_roadmap(
            [ChannelTestCandidate("tv", spend=1_000_000, contribution_share=0.1)],
            n_geos_available=40, geo_cv=0.3,
        )
        d = build_deck("cmo", company="Acme", roadmap_questions=roadmap.questions)
        md = render_deck_markdown(d)
        assert "Open questions for the room" in md
        assert any(q in md for q in roadmap.questions)


class TestFullChain:
    def test_discovery_warnings_become_deck_caveats(self, discovery_report):
        """Data problems found at the start must still be visible at the end.
        A caveat that gets lost between discovery and the deck is how a known
        limitation becomes an unqualified slide."""
        d = build_deck(
            "cfo",
            company="Acme",
            tier="WARN",
            data_caveats=discovery_report.warnings,
        )
        md = render_deck_markdown(d)
        assert "Caveats to state, not bury" in md
        for warning in discovery_report.warnings:
            assert warning in md

    def test_no_experiments_survives_all_the_way_to_every_deck(self):
        for audience in AUDIENCES:
            d = build_deck(audience, company="Acme", tier="PASS", experiments=[])
            assert any("No incrementality experiment" in c for c in d.caveats), audience

    def test_a_failed_model_cannot_produce_a_confident_deck(self):
        d = build_deck("cmo", company="Acme", tier="FAIL", contributions={
            "tv": {"contribution_share": 0.9, "roas": 5.0, "roas_low": 4.9, "roas_high": 5.1}
        })
        assert "did not pass validation" in d.caveats[0]
        # Slides that present the model's results inherit the tier. Slides making
        # meta-claims ("nothing has been tested", "here is what we are asking for")
        # stay high-confidence because they are true independently of the fit.
        result_slides = [
            s for s in d.slides
            if "decomposition" in (s.chart.kind if s.chart else "")
            or "contributor" in s.headline
            or "would happen anyway" in s.headline
        ]
        assert result_slides
        assert all(s.confidence == "low" for s in result_slides)

    def test_whole_chain_runs_without_writing_to_the_fixtures(self, discovery_report, meta_long):
        before = sorted((p.name, p.stat().st_mtime_ns) for p in RAW.iterdir())

        rules = suggest_rules(meta_long["Campaign name"].unique())
        df = meta_long.assign(channel=apply_rules(meta_long, rules, name_column="Campaign name"))
        wide = pivot_to_channels(
            df, date_column="Reporting starts", channel_column="channel",
            value_columns={"spend": "sum"}, freq="W-MON",
        )
        contributions = {
            c.replace("_spend", ""): {"contribution_share": 0.1, "roas": 2.0,
                                      "roas_low": 0.7, "roas_high": 5.0}
            for c in wide.columns if c.endswith("_spend")
        }
        for audience in AUDIENCES:
            assert render_deck_markdown(
                build_deck(audience, company="Acme", tier="WARN", contributions=contributions)
            )

        assert sorted((p.name, p.stat().st_mtime_ns) for p in RAW.iterdir()) == before
