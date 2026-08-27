"""Tests for the backend layer: one spec, three frameworks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from agent_mmm.backends import available_backends, get_backend
from agent_mmm.backends.base import Capability
from agent_mmm.spec import MMMSpec

DATA = Path(__file__).parent / "data"


def _spec(framework="pymc-marketing", **overrides) -> MMMSpec:
    payload = {
        "mmm_type": "greenfield",
        "framework": framework,
        "company_name": "Test Co",
        "industry": "retail",
        "region": "UK",
        "data_path": str(DATA / "synthetic_weekly.csv"),
        "target_unit": {"kind": "monetary", "label": "GBP", "currency_code": "GBP"},
        "channels": [
            {"column": "spend_sem"},
            {"column": "spend_tv"},
            {"column": "spend_social", "role": "organic_media"},
            {"column": "spend_display", "role": "non_media_treatment",
             "channel_type": "price", "expected_sign": "negative"},
        ],
    }
    payload.update(overrides)
    return MMMSpec.model_validate(payload)


def test_registry_lists_capabilities():
    caps = available_backends()
    assert caps["pymc-marketing"] == Capability.executable.value
    assert caps["meridian"] == Capability.codegen.value
    assert caps["robyn"] == Capability.codegen.value


def test_unknown_framework_raises():
    with pytest.raises(ValueError, match="Unknown framework"):
        get_backend("lightweight_mmm")


def test_roles_route_columns_correctly():
    spec = _spec()
    # Organic media is still a media channel; a non-media treatment is a control.
    assert spec.channel_columns() == ["spend_sem", "spend_tv", "spend_social"]
    assert spec.paid_channel_columns() == ["spend_sem", "spend_tv"]
    assert "spend_display" in spec.control_columns()
    # An organic channel has no spend, so it cannot have a ROAS.
    assert "spend_social" not in spec.spend_columns()


@pytest.mark.parametrize("framework", ["pymc-marketing", "meridian", "robyn"])
def test_every_backend_produces_code_and_contract(framework):
    result = get_backend(framework).translate(_spec(framework), build=False)
    assert result.framework == framework
    assert len(result.code) > 200
    assert result.data_contract["required_columns"]


def test_pymc_backend_uses_the_1x_import_path():
    code = get_backend("pymc-marketing").translate(_spec(), build=False).code
    assert "from pymc_marketing.mmm import" in code
    # multidimensional was deprecated in 1.0 and the legacy MMM removed.
    assert "multidimensional" not in code


def test_pymc_panel_dims_are_a_tuple_not_a_string():
    spec = _spec(
        data_path=str(DATA / "synthetic_panel.csv"),
        geo={"is_panel": True, "geo_column": "geo"},
    )
    result = get_backend("pymc-marketing").translate(spec, build=False)
    assert result.structure["dims"] == ("geo",)
    assert "dims=('geo',)" in result.code


def test_pymc_reports_reach_frequency_as_unsupported():
    spec = _spec(channels=[
        {"column": "spend_tv", "role": "paid_reach_frequency",
         "reach_column": "spend_tv", "frequency_column": "spend_sem"},
    ])
    result = get_backend("pymc-marketing").translate(spec, build=False)
    assert any("Reach/frequency" in u for u in result.unsupported)


def test_pymc_widest_l_max_wins_so_no_carryover_is_truncated():
    priors = {"structure": {
        "spend_sem": {"adstock": "geometric", "saturation": "logistic", "l_max": 3},
        "spend_tv": {"adstock": "delayed", "saturation": "logistic", "l_max": 16},
        "spend_social": {"adstock": "geometric", "saturation": "logistic", "l_max": 7},
    }}
    result = get_backend("pymc-marketing").translate(_spec(), priors=priors, build=False)
    assert result.structure["l_max"] == 16
    assert any("l_max" in w for w in result.warnings)


def test_lift_tests_reach_the_generated_pymc_code():
    spec = _spec(experiments=[{
        "channel": "spend_tv", "design": "geo_holdout",
        "lift_absolute": 1500.0, "lift_se": 400.0, "spend_during_test": 60000.0,
    }])
    code = get_backend("pymc-marketing").translate(spec, build=False).code
    assert "add_lift_test_measurements" in code
    assert "60000.0" in code


def test_meridian_separates_organic_and_non_media():
    spec = _spec("meridian")
    result = get_backend("meridian").translate(spec, build=False)
    assert result.structure["organic_media"] == ["spend_social"]
    assert result.structure["non_media_treatments"] == ["spend_display"]
    assert "organic_media=" in result.code
    assert "non_media_treatments=" in result.code


def test_meridian_flags_a_national_spec():
    result = get_backend("meridian").translate(_spec("meridian"), build=False)
    assert any("national" in w.lower() for w in result.warnings)


def test_meridian_cannot_express_a_log_link():
    spec = _spec("meridian", architecture={"link": "log"})
    result = get_backend("meridian").translate(spec, build=False)
    assert any("log-link" in u or "multiplicative" in u for u in result.unsupported)


def test_robyn_rejects_geo_panels():
    spec = _spec(
        "robyn",
        data_path=str(DATA / "synthetic_panel.csv"),
        geo={"is_panel": True, "geo_column": "geo"},
    )
    result = get_backend("robyn").translate(spec, build=False)
    assert any("hierarchical geo" in u for u in result.unsupported)


def test_robyn_puts_organic_channels_in_organic_vars():
    code = get_backend("robyn").translate(_spec("robyn"), build=False).code
    assert 'organic_vars = c("spend_social")' in code
    assert 'paid_media_spends = c("spend_sem", "spend_tv")' in code


def test_robyn_says_priors_lose_information():
    priors = {"per_channel_audit": [{"column": "spend_sem", "alpha_mu": 0.25, "alpha_sigma": 0.1}]}
    result = get_backend("robyn").translate(_spec("robyn"), priors=priors, build=False)
    assert any("bounds" in w for w in result.warnings)


@pytest.mark.requires_pymc
def test_pymc_backend_builds_a_real_model():
    pytest.importorskip("pymc_marketing")
    result = get_backend("pymc-marketing").translate(_spec(), build=True)
    assert result.model is not None
    assert result.model.channel_columns == ["spend_sem", "spend_tv", "spend_social"]
