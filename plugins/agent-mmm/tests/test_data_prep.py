"""Tests for data_prep.py — the transformations that change what the data means."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.data_prep import (
    PrepReport, add_calendar_features, add_event_flags, aggregate_to_period,
    deflate, flag_outliers, impute, infer_aggregation, prepare_dataset,
    reindex_complete, standardise_dates, suggest_imputation,
)
from agent_mmm.spec import DataGranularity, MMMSpec

DATA = Path(__file__).parent / "data"


def _spec(**overrides) -> MMMSpec:
    payload = {
        "mmm_type": "greenfield",
        "company_name": "Test Co",
        "industry": "retail",
        "region": "SE",
        "data_path": str(DATA / "synthetic_weekly.csv"),
        "target_unit": {"kind": "acquisition", "label": "policy"},
        "channels": [{"column": "spend_sem"}, {"column": "spend_tv"}],
        "controls": [{"column": "spend_display", "category": "pricing"}],
    }
    payload.update(overrides)
    return MMMSpec.model_validate(payload)


def test_aggregation_rule_follows_meaning_not_dtype():
    spec = _spec()
    # Spend is a flow, so it sums; a price control is a state, so it averages.
    assert infer_aggregation("spend_sem", spec) == "sum"
    assert infer_aggregation("spend_display", spec) == "mean"
    assert infer_aggregation("y", spec) == "sum"


def test_daily_to_weekly_sums_flows_and_averages_states():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=28, freq="D")
    df = pd.DataFrame({
        "date": days,
        "y": np.ones(28) * 10,
        "spend_sem": np.ones(28) * 100,
        "spend_tv": np.ones(28) * 50,
        "spend_display": np.ones(28) * 5,   # price-like control
    })
    spec = _spec(granularity="daily")
    weekly = aggregate_to_period(df, spec, target_granularity=DataGranularity.weekly)
    assert weekly["spend_sem"].iloc[1] == pytest.approx(700)
    assert weekly["spend_display"].iloc[1] == pytest.approx(5)


def test_reindex_inserts_gaps_as_nan_not_zero():
    df = pd.read_csv(DATA / "synthetic_weekly.csv")
    df["date"] = pd.to_datetime(df["date"])
    gapped = df.drop(index=[10, 11, 12]).reset_index(drop=True)
    report = PrepReport()
    out = reindex_complete(gapped, _spec(), report=report)
    assert len(out) == len(df)
    assert out["spend_sem"].isna().sum() == 3
    assert any("not zero" in a for a in report.actions)


def test_panel_reindex_makes_the_grid_rectangular():
    df = pd.read_csv(DATA / "synthetic_panel.csv")
    df["date"] = pd.to_datetime(df["date"])
    ragged = df.drop(index=df.index[:3]).reset_index(drop=True)
    spec = _spec(
        data_path=str(DATA / "synthetic_panel.csv"),
        geo={"is_panel": True, "geo_column": "geo"},
    )
    out = reindex_complete(ragged, spec)
    per_geo = out.groupby("geo")["date"].nunique()
    assert per_geo.nunique() == 1


def test_suggested_imputation_matches_declared_roles():
    spec = _spec()
    df = pd.read_csv(DATA / "synthetic_weekly.csv")
    strategies = suggest_imputation(spec, df)
    assert strategies["spend_sem"] == "zero"        # no spend means zero spend
    assert strategies["spend_display"] == "ffill"   # a price persists until changed
    assert strategies["y"] == "drop"                # a missing target is unusable


def test_impute_records_every_change():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
    report = PrepReport()
    out = impute(df, {"a": "zero", "b": "ffill"}, report=report)
    assert out["a"].tolist() == [1.0, 0.0, 3.0]
    assert out["b"].tolist() == [2.0, 2.0, 3.0]
    assert len(report.actions) == 2


def test_median_imputation_is_flagged_as_variance_destroying():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    report = PrepReport()
    impute(df, {"a": "median"}, report=report)
    assert any("biases" in w for w in report.warnings)


def test_drop_removes_rows_and_warns_about_the_gap():
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [1.0, 2.0, 3.0]})
    report = PrepReport()
    out = impute(df, {"y": "drop"}, report=report)
    assert len(out) == 2
    assert any("breaks the date sequence" in w for w in report.warnings)


def test_deflate_puts_money_on_one_scale():
    df = pd.DataFrame({"spend": [100.0, 100.0, 100.0]})
    index = pd.Series([100.0, 105.0, 110.0])
    out = deflate(df, ["spend"], index, base_period=0)
    # Later nominal spend buys less, so in base-period terms it is worth less.
    assert out["spend"].iloc[0] == pytest.approx(100.0)
    assert out["spend"].iloc[2] < out["spend"].iloc[0]


def test_calendar_features_include_quarter_flags():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60, freq="W-MON")})
    out = add_calendar_features(df, "date")
    assert {"is_q1", "is_q2", "is_q3", "is_q4", "week_of_year"} <= set(out.columns)
    assert out[["is_q1", "is_q2", "is_q3", "is_q4"]].sum(axis=1).eq(1).all()


def test_event_flags_cover_the_declared_window():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="W-MON")})
    out = add_event_flags(df, "date", {"relaunch": ("2024-02-01", "2024-03-01")})
    assert out["relaunch"].sum() > 0
    assert set(out["relaunch"].unique()) <= {0, 1}


def test_outliers_are_flagged_not_deleted():
    df = pd.DataFrame({"y": [10.0] * 30 + [1000.0]})
    report = PrepReport()
    out = flag_outliers(df, ["y"], z_threshold=3.0, report=report)
    assert len(out) == len(df)
    assert out["outlier_y"].sum() == 1


def test_granularity_mismatch_is_surfaced():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30, freq="D")})
    report = PrepReport()
    standardise_dates(df, "date", DataGranularity.weekly, report)
    assert any("Declared weekly" in w for w in report.warnings)


def test_full_pipeline_leaves_no_missing_modelling_columns():
    df = pd.read_csv(DATA / "synthetic_weekly.csv")
    df.loc[3:6, "spend_sem"] = np.nan
    out, report = prepare_dataset(_spec(), df=df)
    assert out[["spend_sem", "spend_tv", "y"]].isna().sum().sum() == 0
    assert report.rows_before and report.rows_after
    assert "Data Preparation Report" in report.to_markdown()
