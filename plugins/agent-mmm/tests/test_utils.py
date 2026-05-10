"""Tests for utils/moment_match.py and utils/io.py."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.utils.moment_match import beta_moment_match, gamma_moment_match
from agent_mmm.utils.io import load_data, validate_columns, parse_dates


def test_beta_basic():
    a, b = beta_moment_match(0.3, 0.1)
    assert a > 0 and b > 0
    assert abs(a / (a + b) - 0.3) < 0.001


def test_beta_sem():
    a, b = beta_moment_match(0.10, 0.07)
    assert a > 0 and b > 0


def test_beta_invalid_mu():
    with pytest.raises(ValueError):
        beta_moment_match(1.5, 0.1)


def test_beta_zero_sigma():
    with pytest.raises(ValueError):
        beta_moment_match(0.3, 0.0)


def test_gamma_basic():
    a, b = gamma_moment_match(3.5, 1.0)
    assert abs(a / b - 3.5) < 0.01


def test_gamma_tv():
    a, b = gamma_moment_match(1.5, 0.45)
    assert a > 0 and b > 0


def test_gamma_invalid():
    with pytest.raises(ValueError):
        gamma_moment_match(-1, 1)


def test_load_csv(tmp_path):
    df = pd.DataFrame({"date": ["2022-01-03"], "y": [1000.0]})
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    assert "y" in load_data(p).columns


def test_load_parquet(tmp_path):
    df = pd.DataFrame({"date": ["2022-01-03"], "y": [1000.0]})
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)
    assert "y" in load_data(p).columns


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/data.csv")


def test_unsupported_format(tmp_path):
    p = tmp_path / "data.xlsx"
    p.touch()
    with pytest.raises(ValueError):
        load_data(p)


def test_validate_columns_ok():
    df = pd.DataFrame({"date": [], "y": [], "spend_sem": []})
    assert validate_columns(df, ["date", "y", "spend_sem"]) == []


def test_validate_columns_missing():
    df = pd.DataFrame({"date": [], "y": []})
    assert validate_columns(df, ["date", "y", "spend_tv"]) == ["spend_tv"]


def test_parse_dates_sorts():
    df = pd.DataFrame({
        "date": ["2022-01-10", "2022-01-03", "2022-01-17"],
        "y": [1, 2, 3]
    })
    parsed = parse_dates(df, "date")
    assert pd.api.types.is_datetime64_any_dtype(parsed["date"])
    assert list(parsed["y"]) == [2, 1, 3]
