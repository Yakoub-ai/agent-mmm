"""Campaign-name mapping must be auditable and must refuse to be quietly incomplete."""
import pandas as pd
import pytest

from agent_mmm.taxonomy import (
    MappingRule,
    apply_rules,
    apply_rules_with_provenance,
    build_taxonomy,
    pivot_to_channels,
    render_taxonomy_report,
    suggest_rules,
)


@pytest.fixture
def names() -> list[str]:
    return [
        "UK | Search | Brand | Exact",
        "UK | Search | Generic | Broad",
        "FB_Prospecting_Cold_Q1",
        "FB_Retargeting_DPA",
        "TV_Q4_Burst_2023",
        "OOH_London_Billboards",
        "Email_Newsletter_Weekly",
    ]


@pytest.fixture
def long_df(names) -> pd.DataFrame:
    rows = []
    for week in pd.date_range("2023-01-02", periods=8, freq="W-MON"):
        for i, n in enumerate(names):
            rows.append({"date": week, "campaign": n, "spend": 100.0 * (i + 1), "impressions": 1000 * (i + 1)})
    return pd.DataFrame(rows)


class TestSuggestRules:
    def test_only_returns_rules_that_match(self, names):
        rules = suggest_rules(names)
        assert rules
        channels = {r.channel for r in rules}
        assert "search_brand" in channels
        assert "search_generic" in channels
        assert "tv" in channels
        assert "cinema" not in channels     # nothing in the names is cinema

    def test_brand_beats_generic_search_on_priority(self, names):
        rules = {r.channel: r.priority for r in suggest_rules(names)}
        assert rules["search_brand"] < rules["search_generic"]
        assert rules["search_brand"] < rules["search_other"] if "search_other" in rules else True

    def test_rules_are_returned_in_priority_order(self, names):
        rules = suggest_rules(names)
        assert [r.priority for r in rules] == sorted(r.priority for r in rules)

    def test_no_names_yields_no_rules(self):
        assert suggest_rules(["zzz_unknown_thing", "another_mystery"]) == []

    def test_separator_agnostic(self):
        variants = ["UK|Search|Brand", "uk-search-brand", "UK_SEARCH_BRAND", "UK Search Brand"]
        for v in variants:
            assert any(r.channel == "search_brand" for r in suggest_rules([v])), v


class TestApplyRules:
    def test_first_match_wins_by_priority(self):
        df = pd.DataFrame({"campaign": ["brand_search_uk"]})
        rules = [
            MappingRule(pattern=r"\bsearch\b", channel="search_other", priority=90),
            MappingRule(pattern=r"\bbrand\b", channel="search_brand", priority=10),
        ]
        assert apply_rules(df, rules, name_column="campaign").iloc[0] == "search_brand"

    def test_unmatched_rows_are_na_not_invented(self):
        df = pd.DataFrame({"campaign": ["something_odd"]})
        rules = [MappingRule(pattern=r"\btv\b", channel="tv")]
        assert pd.isna(apply_rules(df, rules, name_column="campaign").iloc[0])

    def test_fallback_is_explicit(self):
        df = pd.DataFrame({"campaign": ["something_odd"]})
        rules = [MappingRule(pattern=r"\btv\b", channel="tv")]
        out = apply_rules(df, rules, name_column="campaign", fallback="unclassified")
        assert out.iloc[0] == "unclassified"

    def test_provenance_records_the_winning_rule(self):
        df = pd.DataFrame({"campaign": ["brand_search_uk", "tv_burst"]})
        rules = [
            MappingRule(pattern=r"\bbrand\b", channel="search_brand", priority=10),
            MappingRule(pattern=r"\btv\b", channel="tv", priority=35),
        ]
        channel, via = apply_rules_with_provenance(df, rules, name_column="campaign")
        assert list(channel) == ["search_brand", "tv"]
        assert via.iloc[0] == r"\bbrand\b"


class TestCoverage:
    def test_full_coverage_passes(self, long_df):
        rep = build_taxonomy(long_df, name_column="campaign", spend_column="spend")
        assert rep.ok
        assert rep.coverage_spend == pytest.approx(1.0)

    def test_unmapped_spend_is_an_error_not_a_warning(self, long_df):
        """A channel assembled from 90% of its spend produces a ROAS that is
        wrong by construction and looks entirely normal."""
        extra = long_df.iloc[:8].copy()
        extra["campaign"] = "MYSTERY_LINE_ITEM"
        extra["spend"] = 5000.0
        df = pd.concat([long_df, extra], ignore_index=True)
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert not rep.ok
        assert any("unmapped" in e for e in rep.errors)

    def test_unmapped_names_ranked_by_spend(self, long_df):
        small = long_df.iloc[:1].copy()
        small["campaign"] = "SMALL_MYSTERY"
        small["spend"] = 10.0
        big = long_df.iloc[:1].copy()
        big["campaign"] = "BIG_MYSTERY"
        big["spend"] = 90_000.0
        df = pd.concat([long_df, small, big], ignore_index=True)
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert [u["name"] for u in rep.unmapped][:2] == ["BIG_MYSTERY", "SMALL_MYSTERY"]

    def test_no_rules_match_is_an_error(self):
        df = pd.DataFrame({"campaign": ["xx_1", "xx_2"], "spend": [1.0, 2.0]})
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert not rep.ok
        assert any("not one we recognise" in e for e in rep.errors)

    def test_missing_column_raises(self, long_df):
        with pytest.raises(KeyError):
            build_taxonomy(long_df, name_column="nope")


class TestFindings:
    def test_dead_rule_is_warned_about(self, long_df):
        rules = suggest_rules(long_df["campaign"].unique()) + [
            MappingRule(pattern=r"\bcinema\b", channel="cinema", priority=200)
        ]
        rep = build_taxonomy(long_df, name_column="campaign", rules=rules, spend_column="spend")
        assert any("matched nothing" in w for w in rep.warnings)

    def test_conflicting_rules_are_surfaced(self):
        df = pd.DataFrame({"campaign": ["DPA_Catalog_Retargeting"] * 4, "spend": [100.0] * 4})
        rules = [
            MappingRule(pattern=r"\bdpa\b", channel="social_retargeting", priority=25),
            MappingRule(pattern=r"\bcatalog\b", channel="direct_mail", priority=80),
        ]
        rep = build_taxonomy(df, name_column="campaign", rules=rules, spend_column="spend")
        assert rep.conflicts
        c = rep.conflicts[0]
        assert c["resolved_to"] == "social_retargeting"
        assert "direct_mail" in c["channels"]

    def test_concentration_is_warned_about(self):
        df = pd.DataFrame({
            "campaign": ["TV_Burst"] * 90 + ["OOH_Sites"] * 10,
            "spend": [1000.0] * 90 + [10.0] * 10,
        })
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert any("of mapped spend" in w for w in rep.warnings)

    def test_email_prompts_the_organic_question(self, long_df):
        rep = build_taxonomy(long_df, name_column="campaign", spend_column="spend")
        assert any("never be given a ROAS" in q for q in rep.questions)

    def test_brand_without_generic_is_questioned(self):
        df = pd.DataFrame({"campaign": ["Search_Brand_Exact"] * 5, "spend": [10.0] * 5})
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert any("generic search" in q for q in rep.questions)

    def test_pmax_is_flagged_as_mixed_inventory(self):
        df = pd.DataFrame({"campaign": ["PMax_All"] * 5, "spend": [10.0] * 5})
        rep = build_taxonomy(df, name_column="campaign", spend_column="spend")
        assert any("PMax" in q for q in rep.questions)


class TestPivot:
    def test_pivot_produces_one_row_per_period(self, long_df):
        wide = pivot_to_channels(
            long_df.assign(channel=apply_rules(long_df, suggest_rules(long_df["campaign"].unique()),
                                               name_column="campaign")),
            date_column="date",
            channel_column="channel",
            value_columns={"spend": "sum", "impressions": "sum"},
        )
        assert len(wide) == 8
        assert wide["date"].is_unique
        assert any(c.endswith("_spend") for c in wide.columns)

    def test_pivot_preserves_total_spend(self, long_df):
        channel = apply_rules(long_df, suggest_rules(long_df["campaign"].unique()), name_column="campaign")
        df = long_df.assign(channel=channel)
        wide = pivot_to_channels(
            df, date_column="date", channel_column="channel", value_columns={"spend": "sum"}
        )
        spend_cols = [c for c in wide.columns if c.endswith("_spend")]
        assert wide[spend_cols].to_numpy().sum() == pytest.approx(
            df.loc[channel.notna(), "spend"].sum()
        )

    def test_absent_channel_periods_are_zero_not_missing(self, long_df):
        df = long_df.copy()
        channel = apply_rules(df, suggest_rules(df["campaign"].unique()), name_column="campaign")
        df = df.assign(channel=channel)
        df = df[~((df["channel"] == "tv") & (df["date"] < "2023-02-01"))]
        wide = pivot_to_channels(
            df, date_column="date", channel_column="channel", value_columns={"spend": "sum"}
        )
        assert wide["tv_spend"].isna().sum() == 0
        assert (wide["tv_spend"] == 0).any()

    def test_geo_panel_pivot_is_rectangular(self, long_df):
        df = pd.concat([long_df.assign(geo="north"), long_df.assign(geo="south")], ignore_index=True)
        df["channel"] = apply_rules(df, suggest_rules(df["campaign"].unique()), name_column="campaign")
        wide = pivot_to_channels(
            df, date_column="date", channel_column="channel",
            value_columns={"spend": "sum"}, geo_column="geo",
        )
        assert len(wide) == 8 * 2


class TestRendering:
    def test_renders_markdown(self, long_df):
        rep = build_taxonomy(long_df, name_column="campaign", spend_column="spend")
        md = render_taxonomy_report(rep)
        assert md.startswith("# Channel Taxonomy Mapping")
        assert "## Rules applied" in md
        assert "## Resulting channels" in md

    def test_json_serialisable(self, long_df):
        import json

        json.dumps(build_taxonomy(long_df, name_column="campaign", spend_column="spend").to_dict())
