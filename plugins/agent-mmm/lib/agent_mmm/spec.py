"""Pydantic v2 models for spec.yaml — the single source of truth for an MMM project.

The spec is deliberately *framework-agnostic*. It describes the business problem,
the data, and the channel semantics; a backend (pymc-marketing, Meridian, Robyn)
then translates it into framework-specific model code. See
``agent_mmm.backends`` for the translation layer.

Schema version 2 is a strict superset of version 1: every v1 spec loads unchanged.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "2"


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class MMMType(str, Enum):
    greenfield = "greenfield"
    brownfield = "brownfield"


class Framework(str, Enum):
    """Modelling framework the spec should be compiled to."""

    pymc_marketing = "pymc-marketing"
    meridian = "meridian"
    robyn = "robyn"


class DataGranularity(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class TargetUnitKind(str, Enum):
    monetary = "monetary"
    acquisition = "acquisition"
    volume = "volume"


class ChannelRole(str, Enum):
    """How a variable enters the model.

    The role — not the column name — decides where a variable lands in every
    framework. Getting it wrong is the single most common structural MMM error:
    an organic channel modelled as paid media invites a meaningless ROAS, and a
    price variable modelled as media gets a non-negative coefficient it must not
    have.
    """

    paid_media = "paid_media"
    """Paid channel with spend. Gets adstock + saturation, a ROAS, and a place in
    the budget optimiser."""

    paid_reach_frequency = "paid_reach_frequency"
    """Paid channel measured as reach x frequency (typically CTV/YouTube/TV).
    Meridian models these natively; pymc-marketing needs reach as the channel
    input with frequency as a control or a separate effect."""

    organic_media = "organic_media"
    """Unpaid marketing activity with carryover — organic social, PR, email
    sends, owned content. Gets adstock + saturation but NO ROAS (no spend)."""

    non_media_treatment = "non_media_treatment"
    """Business lever that is not media — price, promo depth, distribution,
    product launches, store count. Gets a coefficient, no adstock/saturation,
    and often a sign constraint. Never appears in a budget optimiser."""

    control = "control"
    """Confounder we adjust for but do not act on — weather, macro, competitor
    activity, calendar. Never attributed, never optimised."""


class FunnelStage(str, Enum):
    upper = "upper"
    mid = "mid"
    lower = "lower"


class ExpectedSign(str, Enum):
    positive = "positive"
    negative = "negative"
    unconstrained = "unconstrained"


class ControlCategory(str, Enum):
    calendar = "calendar"
    macro = "macro"
    competitive = "competitive"
    pricing = "pricing"
    distribution = "distribution"
    operational = "operational"
    data_quality = "data_quality"
    other = "other"


class ExperimentDesign(str, Enum):
    geo_holdout = "geo_holdout"
    geo_split = "geo_split"
    ghost_ads = "ghost_ads"
    psa_control = "psa_control"
    switchback = "switchback"
    synthetic_control = "synthetic_control"
    matched_market = "matched_market"
    conversion_lift = "conversion_lift"
    other = "other"


class TrendKind(str, Enum):
    none = "none"
    linear_changepoints = "linear_changepoints"
    time_varying_intercept = "time_varying_intercept"
    knots = "knots"


class LinkFunction(str, Enum):
    identity = "identity"
    log = "log"


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
class TargetUnit(BaseModel):
    kind: TargetUnitKind
    label: str = Field(..., description="Human-readable unit label, e.g. 'policy', 'SEK', 'signup'")
    currency_code: Optional[str] = Field(None, description="ISO 4217 code if kind=monetary")
    value_per_unit: Optional[float] = Field(
        None, description="Monetary value of one unit; enables ROAS framing for non-monetary targets"
    )
    revenue_per_unit_column: Optional[str] = Field(
        None,
        description="Column holding a time-varying value per unit (Meridian's revenue_per_kpi). "
        "Takes precedence over the scalar value_per_unit when present.",
    )

    @model_validator(mode="after")
    def currency_required_for_monetary(self) -> "TargetUnit":
        if self.kind == TargetUnitKind.monetary and not self.currency_code:
            raise ValueError("currency_code required when kind=monetary")
        return self


class ChannelMeta(BaseModel):
    """One media (or media-like) variable.

    ``column`` is the column that enters the model as the channel input. For a
    spend-driven channel it is usually the same as ``spend_column``; when the
    model is driven by exposure (impressions, GRPs, clicks) set ``column`` to the
    exposure column and ``spend_column`` to the cost column so ROAS stays
    computable.
    """

    column: str
    label: str = ""
    channel_type: Optional[str] = Field(
        None, description="Taxonomy key, e.g. sem_brand, tv_linear. Auto-classified when omitted."
    )
    role: ChannelRole = ChannelRole.paid_media
    is_active: bool = True

    # Column wiring
    spend_column: Optional[str] = Field(
        None, description="Cost column. Defaults to `column` for paid channels driven by spend."
    )
    exposure_column: Optional[str] = Field(
        None, description="Impressions / GRPs / clicks column when the model is exposure-driven."
    )
    reach_column: Optional[str] = None
    frequency_column: Optional[str] = None

    # Semantics that drive priors and structure
    funnel_stage: Optional[FunnelStage] = None
    expected_sign: ExpectedSign = ExpectedSign.positive
    always_on: Optional[bool] = Field(
        None, description="True for channels that never go dark (brand search, always-on social). "
        "Always-on channels have little spend variation and are poorly identified without an experiment."
    )
    halflife_periods: Optional[float] = Field(
        None, gt=0, description="Prior belief about carryover half-life, in data periods."
    )
    adstock: Optional[str] = Field(None, description="Override adstock, e.g. geometric, delayed, weibull_cdf.")
    saturation: Optional[str] = Field(None, description="Override saturation, e.g. logistic, hill, michaelis_menten.")
    l_max: Optional[int] = Field(None, gt=0, description="Override max carryover lag in periods.")
    notes: str = ""

    @model_validator(mode="after")
    def default_spend_column(self) -> "ChannelMeta":
        if self.spend_column is None and self.role in (
            ChannelRole.paid_media,
            ChannelRole.paid_reach_frequency,
        ):
            # Exposure-driven channels must name their cost column explicitly.
            if self.exposure_column is None or self.exposure_column == self.column:
                self.spend_column = self.column
        return self

    @property
    def has_spend(self) -> bool:
        return self.spend_column is not None

    @property
    def model_input_column(self) -> str:
        """Column that actually feeds the media transformation."""
        return self.exposure_column or self.column


class ControlMeta(BaseModel):
    column: str
    label: str = ""
    category: ControlCategory = ControlCategory.other
    expected_sign: ExpectedSign = ExpectedSign.unconstrained
    scale_by_population: bool = Field(
        False, description="Meridian: divide by geo population before scaling (per-capita controls)."
    )
    source: str = "user"
    notes: str = ""


class Experiment(BaseModel):
    """A measured incrementality result used to calibrate the model.

    An MMM without at least one experiment is an extrapolation from observational
    correlation. Every framework has a way to fold these in: pymc-marketing's
    ``add_lift_test_measurements``, Meridian's ROI priors, Robyn's
    ``calibration_input``.
    """

    channel: str = Field(..., description="Channel name; must match a ChannelMeta.column or .label")
    design: ExperimentDesign = ExperimentDesign.geo_holdout
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    spend_during_test: Optional[float] = Field(
        None, ge=0, description="Incremental spend the test evaluated (the delta_x)."
    )
    baseline_spend: Optional[float] = Field(
        None, ge=0, description="Spend level the test started from (the x). Defaults to 0 for on/off tests."
    )
    lift_absolute: Optional[float] = Field(
        None, description="Measured incremental target units (the delta_y). Sign matters."
    )
    lift_se: Optional[float] = Field(
        None, gt=0, description="Standard error of the lift. Drives how hard the model is pulled."
    )
    roi_point: Optional[float] = Field(None, description="Measured ROI/ROAS, when reported instead of absolute lift.")
    roi_ci_low: Optional[float] = None
    roi_ci_high: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Reported confidence level, e.g. 0.9.")
    scope: str = Field("channel", description="channel | total_media")
    source: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def needs_a_measurement(self) -> "Experiment":
        if self.lift_absolute is None and self.roi_point is None:
            raise ValueError(
                f"Experiment for '{self.channel}' has neither lift_absolute nor roi_point — "
                "one is required to calibrate anything."
            )
        return self


class SeasonalityConfig(BaseModel):
    yearly_fourier_modes: int = Field(default=8, ge=1, le=52)
    weekly_fourier_modes: Optional[int] = Field(
        None, ge=1, le=3, description="Only meaningful for daily data."
    )
    explicit_holiday_column: Optional[str] = None
    holiday_country: Optional[str] = Field(None, description="ISO country code for automatic holiday flags.")
    expected_peaks: list[str] = Field(default_factory=list)


class TrendConfig(BaseModel):
    kind: TrendKind = TrendKind.none
    n_changepoints: int = Field(default=10, ge=1)
    n_knots: Optional[int] = Field(None, ge=1, description="Meridian knots; None = one per period (geo models).")
    notes: str = ""


class GeoConfig(BaseModel):
    is_panel: bool = False
    geo_column: Optional[str] = None
    geos: list[str] = Field(default_factory=list)
    population_column: Optional[str] = Field(
        None, description="Required by Meridian geo models; enables per-capita scaling."
    )
    baseline_geo: Optional[str] = Field(None, description="Reference geo for dummy encoding (Meridian).")


class ValidationConfig(BaseModel):
    holdout_start: Optional[date] = None
    holdout_end: Optional[date] = None
    cv_n_init: int = Field(default=80, ge=1)
    cv_forecast_horizon: int = Field(default=13, ge=1)
    cv_step_size: int = Field(default=13, ge=1)
    run_refutation_tests: bool = True


class ArchitectureConfig(BaseModel):
    """Structural modelling choices that are not per-channel."""

    link: LinkFunction = LinkFunction.identity
    likelihood: str = Field("StudentT", description="Normal | StudentT | LogNormal")
    student_t_nu: float = Field(5.0, gt=1)
    time_varying_media: bool = False
    adstock_first: bool = True
    default_adstock: str = "geometric"
    default_saturation: str = "logistic"
    default_l_max: int = Field(8, gt=0)
    use_r2d2_prior: bool = Field(
        False,
        description="Use an R2D2 variance-decomposition prior to control the baseline/media "
        "variance split explicitly instead of independent coefficient priors.",
    )
    expected_media_contribution: Optional[float] = Field(
        None, ge=0, le=1, description="Prior belief about total media share of the target, e.g. 0.25."
    )


class SamplerConfig(BaseModel):
    draws: int = Field(1000, gt=0)
    tune: int = Field(1500, gt=0)
    chains: int = Field(4, gt=0)
    target_accept: float = Field(0.9, gt=0, lt=1)
    random_seed: Optional[int] = 42


class BrownfieldContext(BaseModel):
    idata_path: Optional[str] = None
    prior_model_config_path: Optional[str] = None
    previous_framework: Optional[Framework] = None
    known_results: str = Field("", description="Prior ROAS/contribution figures stakeholders already believe.")
    notes: str = ""


# --------------------------------------------------------------------------- #
# Top-level spec
# --------------------------------------------------------------------------- #
class MMMSpec(BaseModel):
    """Complete spec for one MMM project."""

    version: str = SCHEMA_VERSION
    mmm_type: MMMType
    framework: Framework = Framework.pymc_marketing

    company_name: str
    industry: str
    region: str

    data_path: str
    date_column: str = "date"
    target_column: str = "y"
    target_unit: TargetUnit

    channels: list[ChannelMeta] = Field(default_factory=list)
    controls: list[ControlMeta] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)

    granularity: DataGranularity = DataGranularity.weekly
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)

    brownfield: Optional[BrownfieldContext] = None
    notes: str = ""

    # ---- validators -------------------------------------------------------- #
    @model_validator(mode="after")
    def brownfield_auto_context(self) -> "MMMSpec":
        if self.mmm_type == MMMType.brownfield and self.brownfield is None:
            self.brownfield = BrownfieldContext()
        return self

    @model_validator(mode="after")
    def panel_needs_geo_column(self) -> "MMMSpec":
        if self.geo.is_panel and not self.geo.geo_column:
            raise ValueError("geo.is_panel is true but geo.geo_column is not set")
        return self

    @model_validator(mode="after")
    def experiments_reference_known_channels(self) -> "MMMSpec":
        if not self.experiments or not self.channels:
            return self
        known = {c.column for c in self.channels} | {c.label for c in self.channels if c.label}
        unknown = sorted({e.channel for e in self.experiments if e.channel not in known})
        if unknown:
            raise ValueError(
                f"experiments reference unknown channels: {unknown}. "
                f"Known channels: {sorted(known)}"
            )
        return self

    # ---- accessors --------------------------------------------------------- #
    def active_channels(self) -> list[ChannelMeta]:
        return [c for c in self.channels if c.is_active]

    def channels_with_role(self, *roles: ChannelRole) -> list[ChannelMeta]:
        return [c for c in self.active_channels() if c.role in roles]

    def channel_columns(self) -> list[str]:
        """Columns that enter the media transformation (adstock + saturation)."""
        media_roles = (
            ChannelRole.paid_media,
            ChannelRole.paid_reach_frequency,
            ChannelRole.organic_media,
        )
        return [c.model_input_column for c in self.channels_with_role(*media_roles)]

    def paid_channel_columns(self) -> list[str]:
        return [
            c.model_input_column
            for c in self.channels_with_role(ChannelRole.paid_media, ChannelRole.paid_reach_frequency)
        ]

    def spend_columns(self) -> dict[str, str]:
        """Map model-input column -> spend column, for channels that have spend."""
        return {
            c.model_input_column: c.spend_column
            for c in self.active_channels()
            if c.spend_column is not None
        }

    def control_columns(self) -> list[str]:
        """Control columns, including non-media treatments declared as channels."""
        cols = [c.column for c in self.controls]
        cols += [c.column for c in self.channels_with_role(ChannelRole.non_media_treatment, ChannelRole.control)]
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        out: list[str] = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def periods_per_year(self) -> int:
        return {
            DataGranularity.daily: 365,
            DataGranularity.weekly: 52,
            DataGranularity.monthly: 12,
        }[self.granularity]

    def channel_by_name(self, name: str) -> Optional[ChannelMeta]:
        for c in self.channels:
            if name in (c.column, c.label, c.model_input_column):
                return c
        return None

    def required_columns(self) -> list[str]:
        """Every column the spec expects to find in the dataset."""
        cols = [self.date_column, self.target_column]
        for c in self.active_channels():
            cols += [c.column, c.spend_column, c.exposure_column, c.reach_column, c.frequency_column]
        cols += [c.column for c in self.controls]
        cols += [
            self.geo.geo_column,
            self.geo.population_column,
            self.seasonality.explicit_holiday_column,
            self.target_unit.revenue_per_unit_column,
        ]
        seen: set[str] = set()
        out: list[str] = []
        for c in cols:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out


def load_spec(path: str | Path) -> MMMSpec:
    with open(path) as f:
        data = yaml.safe_load(f)
    return MMMSpec.model_validate(data)


def save_spec(spec: MMMSpec, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(spec.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
