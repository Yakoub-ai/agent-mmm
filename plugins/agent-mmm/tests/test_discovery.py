"""Discovery must read messy real-world exports without deciding anything silently."""
import warnings
from pathlib import Path

import pandas as pd
import pytest

from agent_mmm.discovery import (
    DiscoveryReport,
    discover,
    parse_date_column,
    profile_dataframe,
    profile_file,
    propose_joins,
    render_discovery_report,
)

RAW = Path(__file__).parent / "data" / "raw"


@pytest.fixture(scope="module")
def report() -> DiscoveryReport:
    return discover(RAW)


# --------------------------------------------------------------------------- #
# Date parsing — one format for the whole column, or an explicit question
# --------------------------------------------------------------------------- #
class TestDateParsing:
    def test_day_first_column_parses_consistently(self):
        """The bug this guards: pandas parsing 05/06 as May and 19/06 as June
        in the same column, scrambling the calendar without erroring."""
        s = pd.Series(["05/06/2022", "12/06/2022", "19/06/2022", "26/06/2022"])
        parsed, fmt, ambiguous = parse_date_column(s)
        assert fmt == "%d/%m/%Y"
        assert ambiguous == []
        assert parsed.min() == pd.Timestamp("2022-06-05")
        assert parsed.max() == pd.Timestamp("2022-06-26")
        assert (parsed.dt.month == 6).all()

    def test_ambiguous_dates_are_reported_not_guessed(self):
        s = pd.Series(["01/02/2023", "08/02/2023", "03/04/2023", "10/05/2023"])
        _, fmt, ambiguous = parse_date_column(s)
        assert set(ambiguous) == {"%d/%m/%Y", "%m/%d/%Y"}
        assert fmt in ambiguous

    def test_iso_dates_are_unambiguous(self):
        _, fmt, ambiguous = parse_date_column(pd.Series(["2023-01-02", "2023-01-09"]))
        assert fmt == "%Y-%m-%d"
        assert ambiguous == []

    def test_year_month_strings(self):
        parsed, fmt, _ = parse_date_column(pd.Series(["2022-01", "2022-02", "2022-03"]))
        assert fmt == "%Y-%m"
        assert parsed.iloc[0] == pd.Timestamp("2022-01-01")

    def test_campaign_names_are_not_dates(self):
        s = pd.Series(["BR_Prospecting_Q1", "PERF_DPA_Catalog", "BRAND_Awareness"])
        parsed, fmt, _ = parse_date_column(s)
        assert fmt is None
        assert parsed.empty

    def test_bare_integers_are_not_dates(self):
        _, fmt, _ = parse_date_column(pd.Series([1, 2, 3, 4, 5]))
        assert fmt is None

    def test_yyyymmdd_integers_are_dates(self):
        parsed, fmt, _ = parse_date_column(pd.Series([20220103, 20220110]))
        assert parsed.iloc[0] == pd.Timestamp("2022-01-03")

    def test_empty_series(self):
        parsed, fmt, ambiguous = parse_date_column(pd.Series([], dtype="object"))
        assert parsed.empty and fmt is None and ambiguous == []

    def test_no_dateutil_warnings_on_a_messy_folder(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            discover(RAW)


# --------------------------------------------------------------------------- #
# Column semantics
# --------------------------------------------------------------------------- #
class TestColumnRoles:
    def test_currency_text_is_recognised_as_numeric_spend(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-02", periods=6, freq="W-MON"),
            "Amount spent (GBP)": ["£1,200.00", "£980.50", "£1,010.10", "£2,300.00", "£75.00", "£640.20"],
        })
        prof = profile_dataframe(df, name="meta.csv")
        col = prof.column("Amount spent (GBP)")
        assert col.numeric_after_cleaning is True
        assert col.detected_currency == "GBP"
        assert col.role == "spend"
        assert col.max_value == pytest.approx(2300.0)

    def test_accounting_negatives_are_parsed(self):
        df = pd.DataFrame({"cost": ["1,000.00", "(2,500.00)", "3,000.00", "4,000.00"]})
        col = profile_dataframe(df, name="f.csv").column("cost")
        assert col.numeric_after_cleaning
        assert col.min_value == pytest.approx(-2500.0)

    @pytest.mark.parametrize(
        "column,expected",
        [
            ("week_start", "date"),
            ("region", "geo"),
            ("tv_spend", "spend"),
            ("impressions", "exposure"),
            ("net_revenue", "target"),
            ("average_selling_price", "price"),
            ("store_count", "distribution"),
            ("campaign_name", "entity"),
        ],
    )
    def test_role_vocabulary(self, column, expected):
        n = 40
        if expected == "date":
            values = pd.date_range("2023-01-02", periods=n, freq="W-MON")
        elif expected in {"geo", "entity"}:
            values = [f"v{i % 4}" for i in range(n)]
        else:
            values = range(1, n + 1)
        prof = profile_dataframe(pd.DataFrame({column: values}), name="t.csv")
        assert prof.column(column).role == expected

    def test_rates_are_never_spend(self):
        """CPC/CPM/ROAS look like money but are ratios; a ratio cannot carry
        adstock and must never become a channel input."""
        df = pd.DataFrame({"cpc": [0.4, 0.5, 0.6], "cost_per_acquisition": [12.0, 14.0, 11.0]})
        prof = profile_dataframe(df, name="t.csv")
        for name in ("cpc", "cost_per_acquisition"):
            assert prof.column(name).role == "control"
            assert "rate" in prof.column(name).role_reason

    def test_name_says_spend_but_values_are_text(self):
        df = pd.DataFrame({"spend": ["not available", "pending", "tbc", "unknown"]})
        col = profile_dataframe(df, name="t.csv").column("spend")
        assert col.role == "unknown"
        assert "not numeric" in col.role_reason

    def test_constant_column_flagged(self):
        df = pd.DataFrame({"currency": ["GBP"] * 20})
        col = profile_dataframe(df, name="t.csv").column("currency")
        assert col.is_constant
        assert any("constant" in n for n in col.notes)

    def test_negative_spend_is_noted(self):
        df = pd.DataFrame({"tv_spend": [100.0, 200.0, -150.0, 300.0]})
        col = profile_dataframe(df, name="t.csv").column("tv_spend")
        assert col.negative_fraction == pytest.approx(0.25)
        assert any("negative" in n for n in col.notes)


# --------------------------------------------------------------------------- #
# File shape and grain
# --------------------------------------------------------------------------- #
class TestFileShape:
    def test_long_export_detected(self, report):
        meta = next(f for f in report.files if f.name == "meta_ads_export.csv")
        assert meta.shape_kind == "long_transactional"
        assert meta.inferred_grain == "daily"
        assert meta.likely_source == "meta_ads"
        assert "Campaign name" in meta.grain_keys

    def test_day_first_finance_file_reads_correctly(self, report):
        """Regression: this file was previously mis-parsed into February and
        misclassified as long format as a result."""
        fin = next(f for f in report.files if f.name == "finance_sales.csv")
        assert fin.date_format == "%d/%m/%Y"
        assert fin.date_min == "2022-06-05"
        assert fin.shape_kind == "wide_timeseries"
        assert fin.inferred_grain == "weekly"

    def test_geo_panel_is_wide_not_long(self, report):
        geo = next(f for f in report.files if f.name == "ooh_geo_panel.csv")
        assert geo.shape_kind == "wide_timeseries"
        assert "region" in geo.grain_keys

    def test_lookup_table_has_no_date(self, report):
        lookup = next(f for f in report.files if f.name == "geo_lookup.csv")
        assert lookup.shape_kind == "lookup"
        assert lookup.date_column is None

    def test_duplicate_rows_detected(self, report):
        g = next(f for f in report.files if f.name == "google_ads_weekly.csv")
        assert g.duplicate_key_rows == 1

    def test_non_data_files_skipped(self, report):
        assert any(s["path"].endswith("readme.md") for s in report.skipped)
        assert all(f.name != "readme.md" for f in report.files)

    def test_unreadable_file_does_not_raise(self, tmp_path):
        bad = tmp_path / "broken.csv"
        bad.write_bytes(b"\xff\xfe\x00garbage,,,\n\x00\x01")
        prof = profile_file(bad)
        assert prof.read_error is not None or prof.n_rows >= 0


# --------------------------------------------------------------------------- #
# Coverage and joins
# --------------------------------------------------------------------------- #
class TestCoverage:
    def test_modal_grain_wins_over_coarsest(self, report):
        """One monthly TV plan must not drag a weekly model down to 24 rows."""
        assert report.recommended_grain == "weekly"

    def test_coarser_source_raises_a_blocking_question(self, report):
        grain_qs = [q for q in report.open_questions if q["topic"] == "grain"]
        assert grain_qs
        assert any("tv_plan_monthly.csv" in q["question"] for q in grain_qs)
        assert all(q["blocking"] == "yes" for q in grain_qs)

    def test_usable_window_is_the_intersection(self, report):
        start, end = report.recommended_date_range
        assert start == "2022-06-05"     # finance starts latest
        assert end == "2023-12-01"       # tv plan ends earliest

    def test_short_window_warns_and_blocks(self, report):
        assert any("practical minimum" in w for w in report.warnings)
        assert any(q["topic"] == "coverage" for q in report.blocking_questions)

    def test_joins_flag_geo_vs_national_mismatch(self, report):
        geo_joins = [j for j in report.joins if "ooh_geo_panel.csv" in (j.left, j.right)]
        assert geo_joins
        assert any(
            any("geo panel" in n for n in j.notes) for j in geo_joins
        ), "a national file joined to a geo panel must be flagged"

    def test_joins_flag_grain_mismatch(self, report):
        pairs = {
            (j.left, j.right)
            for j in report.joins
            if any("grains differ" in n for n in j.notes)
        }
        assert pairs

    def test_overlap_counts_are_reported(self, report):
        j = next(
            j for j in report.joins
            if {j.left, j.right} == {"google_ads_weekly.csv", "finance_sales.csv"}
        )
        assert j.overlap_periods > 0
        # Google Ads covers a wider window than finance, so one side has
        # periods the other does not, whichever way round the pair was built.
        assert j.left_only_periods + j.right_only_periods > 0
        assert j.confidence < 1.0


# --------------------------------------------------------------------------- #
# Questions are the deliverable
# --------------------------------------------------------------------------- #
class TestQuestions:
    def test_always_asks_about_experiments(self, report):
        qs = [q for q in report.open_questions if q["topic"] == "experiments"]
        assert len(qs) == 1
        assert qs[0]["blocking"] == "yes"
        assert "geo holdout" in qs[0]["question"]

    def test_always_asks_about_the_decision(self, report):
        assert any(q["topic"] == "decision" and q["blocking"] == "yes" for q in report.open_questions)

    def test_always_asks_about_channel_roles(self, report):
        assert any(
            q["topic"] == "channels" and "organic" in q["question"] for q in report.open_questions
        )

    def test_asks_about_delivery_vs_billing_dates(self, report):
        assert any("delivered" in q["question"] for q in report.open_questions)

    def test_asks_week_start_day_for_weekly_files(self, report):
        assert any("week start on" in q["question"] for q in report.open_questions)

    def test_asks_about_unknown_columns(self, report):
        assert any("notes_field" in q["question"] for q in report.open_questions)

    def test_asks_which_target_when_several_candidates(self, report):
        target_qs = [q for q in report.open_questions if q["topic"] == "target"]
        assert target_qs and target_qs[0]["blocking"] == "yes"

    def test_missing_price_or_distribution_is_asked_about(self, report):
        confounders = [q for q in report.open_questions if q["topic"] == "confounders"]
        assert any("distribution" in q["question"] for q in confounders)

    def test_every_question_explains_why(self, report):
        for q in report.open_questions:
            assert q["why"].strip(), f"question without a reason: {q['question']}"

    def test_no_target_at_all_is_blocking(self):
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-03", periods=60, freq="W-MON"),
            "tv_spend": range(60),
        })
        rep = DiscoveryReport(root="x")
        rep.files = [profile_dataframe(df, name="media.csv")]
        from agent_mmm.discovery import _ask_about_structure

        _ask_about_structure(rep)
        assert any(q["topic"] == "target" and q["blocking"] == "yes" for q in rep.open_questions)


# --------------------------------------------------------------------------- #
# Rendering and contracts
# --------------------------------------------------------------------------- #
class TestRendering:
    def test_report_renders_markdown(self, report):
        md = render_discovery_report(report)
        assert md.startswith("# Data Discovery Report")
        assert "## Blocking — answer these first" in md
        assert "## Files" in md
        assert "meta_ads_export.csv" in md

    def test_report_is_json_serialisable(self, report):
        import json

        json.dumps(report.to_dict())

    def test_discovery_changes_nothing_on_disk(self, report):
        before = sorted((p.name, p.stat().st_mtime_ns) for p in RAW.iterdir())
        discover(RAW)
        after = sorted((p.name, p.stat().st_mtime_ns) for p in RAW.iterdir())
        assert before == after

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError):
            discover("does/not/exist")

    def test_empty_folder_warns(self, tmp_path):
        rep = discover(tmp_path)
        assert rep.files == []
        assert any("No readable data files" in w for w in rep.warnings)

    def test_single_file_root(self):
        rep = discover(RAW / "google_ads_weekly.csv")
        assert len(rep.files) == 1


class TestContainment:
    def test_symlink_out_of_the_root_is_not_profiled(self, tmp_path):
        """Silently profiling files from elsewhere on the machine is never what
        was wanted, and a symlink out of a data drop is almost always an accident."""
        import os

        drop = tmp_path / "drop"
        drop.mkdir()
        (drop / "ok.csv").write_text("date,spend\n2022-01-03,100\n2022-01-10,120\n")

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "secret.csv").write_text("date,secret\n2022-01-03,42\n")
        os.symlink(elsewhere / "secret.csv", drop / "link.csv")

        rep = discover(drop)
        assert [f.name for f in rep.files] == ["ok.csv"]
        assert any("outside the root" in s["reason"] for s in rep.skipped)

    def test_symlink_within_the_root_is_allowed(self, tmp_path):
        import os

        drop = tmp_path / "drop"
        (drop / "sub").mkdir(parents=True)
        (drop / "sub" / "real.csv").write_text("date,spend\n2022-01-03,100\n2022-01-10,120\n")
        os.symlink(drop / "sub" / "real.csv", drop / "alias.csv")

        rep = discover(drop)
        assert {f.name for f in rep.files} == {"real.csv", "alias.csv"}
