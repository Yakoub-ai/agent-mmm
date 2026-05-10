"""Tests for spec.py."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.spec import (
    MMMSpec, MMMType, TargetUnitKind, load_spec, save_spec,
)


def _minimal(**overrides) -> dict:
    base = {
        "mmm_type": "greenfield",
        "company_name": "Acme Insurance",
        "industry": "insurance",
        "region": "Sweden",
        "data_path": "/tmp/data.csv",
        "date_column": "date",
        "target_column": "y",
        "target_unit": {"kind": "acquisition", "label": "policy"},
        "channels": [{"column": "spend_sem", "label": "SEM"}],
        "controls": [],
    }
    base.update(overrides)
    return base


def test_minimal_greenfield():
    spec = MMMSpec.model_validate(_minimal())
    assert spec.mmm_type == MMMType.greenfield
    assert spec.target_unit.kind == TargetUnitKind.acquisition
    assert spec.channel_columns() == ["spend_sem"]


def test_monetary_requires_currency():
    with pytest.raises(Exception):
        MMMSpec.model_validate(_minimal(target_unit={"kind": "monetary", "label": "SEK"}))


def test_monetary_with_currency():
    spec = MMMSpec.model_validate(_minimal(
        target_unit={"kind": "monetary", "label": "SEK", "currency_code": "SEK"}
    ))
    assert spec.target_unit.currency_code == "SEK"


def test_brownfield_auto_creates_context():
    spec = MMMSpec.model_validate(_minimal(mmm_type="brownfield"))
    assert spec.brownfield is not None


def test_inactive_channels_excluded():
    spec = MMMSpec.model_validate(_minimal(channels=[
        {"column": "spend_sem", "is_active": True},
        {"column": "spend_tv", "is_active": False},
    ]))
    assert spec.channel_columns() == ["spend_sem"]


def test_save_load_roundtrip(tmp_path):
    spec = MMMSpec.model_validate(_minimal())
    p = tmp_path / "spec.yaml"
    save_spec(spec, p)
    loaded = load_spec(p)
    assert loaded.company_name == spec.company_name
    assert loaded.target_unit.label == spec.target_unit.label
