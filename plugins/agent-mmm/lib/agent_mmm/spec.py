"""Pydantic v2 models for spec.yaml — the single source of truth for an MMM project."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class MMMType(str, Enum):
    greenfield = "greenfield"
    brownfield = "brownfield"


class DataGranularity(str, Enum):
    weekly = "weekly"


class TargetUnitKind(str, Enum):
    monetary = "monetary"
    acquisition = "acquisition"
    volume = "volume"


class TargetUnit(BaseModel):
    kind: TargetUnitKind
    label: str = Field(..., description="Human-readable unit label, e.g. 'policy', 'SEK', 'signup'")
    currency_code: Optional[str] = Field(None, description="ISO 4217 code if kind=monetary")
    value_per_unit: Optional[float] = Field(None, description="Optional monetary value per unit")

    @model_validator(mode="after")
    def currency_required_for_monetary(self) -> "TargetUnit":
        if self.kind == TargetUnitKind.monetary and not self.currency_code:
            raise ValueError("currency_code required when kind=monetary")
        return self


class ChannelMeta(BaseModel):
    column: str
    label: str = ""
    channel_type: Optional[str] = None
    is_active: bool = True


class ControlMeta(BaseModel):
    column: str
    label: str = ""
    source: str = "user"


class SeasonalityConfig(BaseModel):
    yearly_fourier_modes: int = Field(default=8, ge=1, le=52)
    explicit_holiday_column: Optional[str] = None
    expected_peaks: list[str] = Field(default_factory=list)


class GeoConfig(BaseModel):
    is_panel: bool = False
    geo_column: Optional[str] = None
    geos: list[str] = Field(default_factory=list)


class BrownfieldContext(BaseModel):
    idata_path: Optional[str] = None
    prior_model_config_path: Optional[str] = None
    notes: str = ""


class MMMSpec(BaseModel):
    """Complete spec for one MMM project."""
    version: str = "1"
    mmm_type: MMMType
    company_name: str
    industry: str
    region: str
    data_path: str
    date_column: str = "date"
    target_column: str = "y"
    target_unit: TargetUnit
    channels: list[ChannelMeta] = Field(default_factory=list)
    controls: list[ControlMeta] = Field(default_factory=list)
    granularity: DataGranularity = DataGranularity.weekly
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    brownfield: Optional[BrownfieldContext] = None
    notes: str = ""

    @model_validator(mode="after")
    def brownfield_auto_context(self) -> "MMMSpec":
        if self.mmm_type == MMMType.brownfield and self.brownfield is None:
            self.brownfield = BrownfieldContext()
        return self

    def channel_columns(self) -> list[str]:
        return [c.column for c in self.channels if c.is_active]

    def control_columns(self) -> list[str]:
        return [c.column for c in self.controls]


def load_spec(path: str | Path) -> MMMSpec:
    with open(path) as f:
        data = yaml.safe_load(f)
    return MMMSpec.model_validate(data)


def save_spec(spec: MMMSpec, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(spec.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
