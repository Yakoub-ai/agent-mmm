"""Tests for target_units.py."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.spec import TargetUnit, TargetUnitKind
from agent_mmm.target_units import is_monetary, cpa_label, roas_label, spend_to_return_ratio


def monetary():
    return TargetUnit(kind=TargetUnitKind.monetary, label="SEK", currency_code="SEK")

def acquisition():
    return TargetUnit(kind=TargetUnitKind.acquisition, label="policy")

def acquisition_with_value():
    return TargetUnit(kind=TargetUnitKind.acquisition, label="policy", value_per_unit=1200.0)


def test_is_monetary():
    assert is_monetary(monetary()) is True
    assert is_monetary(acquisition()) is False


def test_cpa_label():
    assert cpa_label(acquisition()) == "Cost per policy"


def test_roas_label_monetary():
    label = roas_label(monetary())
    assert "ROAS" in label and "SEK" in label


def test_roas_label_acquisition():
    assert "Cost per" in roas_label(acquisition())


def test_ratio_monetary():
    r = spend_to_return_ratio(10_000, 25_000, monetary())
    assert r["roas"] == pytest.approx(2.5)
    assert r["cpa"] == pytest.approx(0.4)


def test_ratio_acquisition_no_value():
    r = spend_to_return_ratio(50_000, 100, acquisition())
    assert r["roas"] is None
    assert r["cpa"] == pytest.approx(500.0)


def test_ratio_acquisition_with_value():
    r = spend_to_return_ratio(50_000, 100, acquisition_with_value())
    assert r["roas"] == pytest.approx((100 * 1200) / 50_000)


def test_zero_units():
    r = spend_to_return_ratio(1000, 0, monetary())
    assert r["cpa"] is None and r["roas"] is None
