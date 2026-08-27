"""pymc-marketing backend (executable).

Targets pymc-marketing >= 1.0, where the multidimensional MMM *is* the MMM:

* ``from pymc_marketing.mmm import MMM`` — the ``multidimensional`` module is
  deprecated and the legacy 0.x ``MMM`` class no longer exists.
* ``idata`` is an ``xarray.DataTree``, not an ArviZ ``InferenceData``.
* ``dims`` is a tuple of extra panel dimensions, e.g. ``("geo",)``.
* Budget optimisation is ``mmm.budget_optimizer(start, end)``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agent_mmm.backends.base import Capability, TranslationResult
from agent_mmm.spec import ChannelRole, LinkFunction, MMMSpec, TrendKind
from agent_mmm.utils.io import load_data, parse_dates

ADSTOCK_CLASSES = {
    "geometric": "GeometricAdstock",
    "delayed": "DelayedAdstock",
    "weibull_cdf": "WeibullCDFAdstock",
    "weibull_pdf": "WeibullPDFAdstock",
    "binomial": "BinomialAdstock",
    "none": "NoAdstock",
}

SATURATION_CLASSES = {
    "logistic": "LogisticSaturation",
    "inverse_scaled_logistic": "InverseScaledLogisticSaturation",
    "hill": "HillSaturation",
    "hill_sigmoid": "HillSaturationSigmoid",
    "michaelis_menten": "MichaelisMentenSaturation",
    "tanh": "TanhSaturation",
    "tanh_baselined": "TanhSaturationBaselined",
    "root": "RootSaturation",
    "log": "LogSaturation",
    "none": "NoSaturation",
}


def _dict_to_prior(d: dict) -> Any:
    """Convert a plain dict from model_config.json into a ``Prior`` object."""
    from pymc_extras.prior import Prior

    dist = d["distribution"]
    dims = d.get("dims")
    kwargs = {k: v for k, v in d.items() if k not in ("distribution", "dims")}
    for key, val in list(kwargs.items()):
        if isinstance(val, list):
            kwargs[key] = np.array(val)
        elif isinstance(val, dict) and "distribution" in val:
            kwargs[key] = _dict_to_prior(val)
    return Prior(dist, dims=dims, **kwargs) if dims else Prior(dist, **kwargs)


def build_model_config_priors(model_config_dict: dict) -> dict:
    """Convert a JSON model_config to ``Prior`` objects, skipping ``_metadata``."""
    out = {}
    for key, value in model_config_dict.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and "distribution" in value:
            out[key] = _dict_to_prior(value)
    return out


def prepare_data(spec: MMMSpec, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for ``MMM.fit``.

    ``y`` is named to match ``spec.target_column``; pymc-marketing keys the
    likelihood off that name and a mismatch fails deep inside model building
    with an opaque error.
    """
    if df is None:
        df = parse_dates(load_data(spec.data_path), spec.date_column)

    feature_cols = [spec.date_column]
    if spec.geo.is_panel and spec.geo.geo_column:
        feature_cols.append(spec.geo.geo_column)
    feature_cols += spec.channel_columns()
    feature_cols += spec.control_columns()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns required by the spec are missing from the data: {missing}")
    if spec.target_column not in df.columns:
        raise KeyError(f"Target column '{spec.target_column}' not found in data")

    X = df[feature_cols].copy()
    y = pd.Series(df[spec.target_column].to_numpy(), name=spec.target_column)
    return X, y


def _resolve_structure(spec: MMMSpec, priors: dict | None) -> dict[str, Any]:
    """Decide the adstock/saturation/l_max the model will actually use."""
    structure = (priors or {}).get("structure", {})
    media = spec.channels_with_role(
        ChannelRole.paid_media, ChannelRole.paid_reach_frequency, ChannelRole.organic_media
    )
    resolved = {}
    for ch in media:
        col = ch.model_input_column
        s = structure.get(col, {})
        resolved[col] = {
            "adstock": ch.adstock or s.get("adstock") or spec.architecture.default_adstock,
            "saturation": ch.saturation or s.get("saturation") or spec.architecture.default_saturation,
            "l_max": ch.l_max or s.get("l_max") or spec.architecture.default_l_max,
        }
    return resolved


def _pick_shared_transforms(resolved: dict[str, Any]) -> tuple[str, str, int, list[str]]:
    """Collapse per-channel structure to the single adstock/saturation pair.

    ``MMM(adstock=..., saturation=...)`` applies one transformation to every
    channel. Per-channel transformations exist via ``MediaConfigList``, but they
    change the model's variable names and therefore every downstream helper, so
    the default path picks the majority adstock and the widest ``l_max`` and
    reports what that cost.
    """
    notes: list[str] = []
    adstocks = [v["adstock"] for v in resolved.values()]
    saturations = [v["saturation"] for v in resolved.values()]
    l_max = max((v["l_max"] for v in resolved.values()), default=8)

    adstock = max(set(adstocks), key=adstocks.count) if adstocks else "geometric"
    saturation = max(set(saturations), key=saturations.count) if saturations else "logistic"

    if len(set(adstocks)) > 1:
        odd = sorted({a for a in adstocks if a != adstock})
        notes.append(
            f"Channels wanted different adstocks ({odd} vs '{adstock}'); the shared model uses "
            f"'{adstock}'. Use MediaConfigList to give offline channels a delayed adstock "
            "while digital keeps geometric."
        )
    if len(set(saturations)) > 1:
        notes.append(
            f"Mixed saturation requests collapsed to '{saturation}'. Use MediaConfigList for "
            "per-channel saturation."
        )
    if len(set(v["l_max"] for v in resolved.values())) > 1:
        notes.append(
            f"l_max set to the maximum requested ({l_max}) so no channel's carryover is "
            "truncated. Short-carryover channels pay a small compute cost; a truncated "
            "long-carryover channel would pay with a biased decay estimate."
        )
    return adstock, saturation, l_max, notes


class PyMCMarketingBackend:
    name = "pymc-marketing"
    capability = Capability.executable

    def translate(
        self,
        spec: MMMSpec,
        priors: dict | None = None,
        build: bool = True,
        df: pd.DataFrame | None = None,
        **_: Any,
    ) -> TranslationResult:
        resolved = _resolve_structure(spec, priors)
        adstock, saturation, l_max, notes = _pick_shared_transforms(resolved)
        unsupported: list[str] = []
        warnings = list(notes)

        rf = spec.channels_with_role(ChannelRole.paid_reach_frequency)
        if rf:
            unsupported.append(
                "Reach/frequency channels ("
                + ", ".join(c.column for c in rf)
                + ") have no native reach x frequency likelihood in pymc-marketing. "
                "Model reach as the channel input and frequency as a control, or use "
                "Meridian if frequency optimisation matters."
            )

        if spec.trend.kind == TrendKind.knots:
            unsupported.append(
                "trend.kind='knots' is a Meridian concept. The pymc-marketing equivalent is "
                "time_varying_intercept=True (HSGP) or a LinearTrend additive effect."
            )

        code = self._render_code(spec, resolved, adstock, saturation, l_max, priors)
        structure = {
            "shared_adstock": adstock,
            "shared_saturation": saturation,
            "l_max": l_max,
            "per_channel": resolved,
            "link": spec.architecture.link.value,
            "dims": (spec.geo.geo_column,) if spec.geo.is_panel and spec.geo.geo_column else (),
        }

        model = None
        if build:
            try:
                model = self.build(spec, priors=priors, df=df)
            except ImportError as e:
                warnings.append(f"pymc-marketing not importable, returning code only: {e}")

        return TranslationResult(
            framework=self.name,
            capability=self.capability,
            code=code,
            data_contract=self.data_contract(spec),
            structure=structure,
            unsupported=unsupported,
            warnings=warnings,
            model=model,
        )

    # ------------------------------------------------------------------ #
    def build(self, spec: MMMSpec, priors: dict | None = None, df: pd.DataFrame | None = None) -> Any:
        """Instantiate an unfitted ``MMM``."""
        from pymc_marketing.mmm import MMM
        import pymc_marketing.mmm as pmm

        resolved = _resolve_structure(spec, priors)
        adstock_key, saturation_key, l_max, _ = _pick_shared_transforms(resolved)

        adstock_cls = getattr(pmm, ADSTOCK_CLASSES.get(adstock_key, "GeometricAdstock"))
        saturation_cls = getattr(pmm, SATURATION_CLASSES.get(saturation_key, "LogisticSaturation"))

        adstock_kwargs: dict[str, Any] = {"l_max": l_max}
        if adstock_key in ("geometric", "delayed"):
            # The half-life parameterisation makes the posterior directly readable
            # as "half the effect is gone after N weeks".
            adstock_kwargs["parametrization"] = "alpha"
        adstock = adstock_cls(**adstock_kwargs) if adstock_key != "none" else adstock_cls()
        saturation = saturation_cls()

        model_config = None
        if priors:
            cfg = priors.get("model_config", priors)
            model_config = build_model_config_priors(cfg)

        kwargs: dict[str, Any] = dict(
            date_column=spec.date_column,
            channel_columns=spec.channel_columns(),
            target_column=spec.target_column,
            adstock=adstock,
            saturation=saturation,
            adstock_first=spec.architecture.adstock_first,
            link=spec.architecture.link.value,
        )
        if model_config:
            kwargs["model_config"] = model_config
        if spec.control_columns():
            kwargs["control_columns"] = spec.control_columns()
        if spec.seasonality.yearly_fourier_modes:
            kwargs["yearly_seasonality"] = spec.seasonality.yearly_fourier_modes
        if spec.geo.is_panel and spec.geo.geo_column:
            # dims must be a TUPLE of extra panel dimensions; passing a bare
            # string silently iterates its characters.
            kwargs["dims"] = (spec.geo.geo_column,)
        if spec.trend.kind == TrendKind.time_varying_intercept:
            kwargs["time_varying_intercept"] = True
        if spec.architecture.time_varying_media:
            kwargs["time_varying_media"] = True

        # Non-spend channels need a cost per unit so ROAS and budget
        # optimisation stay in money rather than impressions.
        cost_per_unit = self._cost_per_unit_frame(spec, df)
        if cost_per_unit is not None:
            kwargs["cost_per_unit"] = cost_per_unit

        return MMM(**kwargs)

    @staticmethod
    def _cost_per_unit_frame(spec: MMMSpec, df: pd.DataFrame | None) -> pd.DataFrame | None:
        """Per-period cost per exposure unit, for exposure-driven channels."""
        exposure_channels = [
            c for c in spec.active_channels()
            if c.exposure_column and c.spend_column and c.exposure_column != c.spend_column
        ]
        if not exposure_channels:
            return None
        if df is None:
            df = parse_dates(load_data(spec.data_path), spec.date_column)

        frame = pd.DataFrame({spec.date_column: df[spec.date_column]})
        for c in exposure_channels:
            if c.spend_column not in df.columns or c.exposure_column not in df.columns:
                continue
            exposure = df[c.exposure_column].replace(0, np.nan)
            frame[c.exposure_column] = (df[c.spend_column] / exposure).fillna(1.0)
        return frame.set_index(spec.date_column) if frame.shape[1] > 1 else None

    # ------------------------------------------------------------------ #
    @staticmethod
    def data_contract(spec: MMMSpec) -> dict[str, Any]:
        return {
            "shape": "long, one row per date"
            + (f" x {spec.geo.geo_column}" if spec.geo.is_panel else ""),
            "required_columns": spec.required_columns(),
            "date_column": spec.date_column,
            "target_column": spec.target_column,
            "channel_columns": spec.channel_columns(),
            "control_columns": spec.control_columns(),
            "rectangular": bool(spec.geo.is_panel),
            "notes": [
                "Target and channel columns are max-abs scaled internally; controls are NOT. "
                "Scale or standardise controls yourself if their magnitudes differ wildly.",
                "The y Series name must equal target_column.",
                "Panel data must be rectangular: every geo needs a row for every date.",
            ],
        }

    def _render_code(
        self,
        spec: MMMSpec,
        resolved: dict[str, Any],
        adstock: str,
        saturation: str,
        l_max: int,
        priors: dict | None,
    ) -> str:
        adstock_cls = ADSTOCK_CLASSES.get(adstock, "GeometricAdstock")
        saturation_cls = SATURATION_CLASSES.get(saturation, "LogisticSaturation")
        controls = spec.control_columns()
        lines = [
            '"""Generated by agent-mmm for pymc-marketing >= 1.0."""',
            "import pandas as pd",
            "from pymc_marketing.mmm import (",
            f"    MMM, {adstock_cls}, {saturation_cls},",
            ")",
            "",
            f"df = pd.read_csv({spec.data_path!r}, parse_dates=[{spec.date_column!r}])",
            "",
            f"channel_columns = {spec.channel_columns()!r}",
        ]
        if controls:
            lines.append(f"control_columns = {controls!r}")
        feature_cols = "[{date!r}] + channel_columns".format(date=spec.date_column)
        if spec.geo.is_panel and spec.geo.geo_column:
            feature_cols = "[{date!r}, {geo!r}] + channel_columns".format(
                date=spec.date_column, geo=spec.geo.geo_column
            )
        if controls:
            feature_cols += " + control_columns"
        lines += [
            f"X = df[{feature_cols}]",
            f"y = df[{spec.target_column!r}].rename({spec.target_column!r})  # name must match target_column",
            "",
        ]

        if priors and priors.get("model_config"):
            lines += [
                "# Priors from /mmm-recommend-priors. Regenerate rather than hand-editing:",
                "# the audit report explains why each one is what it is.",
                "import json",
                "from agent_mmm.backends.pymc_backend import build_model_config_priors",
                "",
                'with open("mmm-workspace/priors/model_config.json") as f:',
                "    model_config = build_model_config_priors(json.load(f)[\"model_config\"])",
                "",
            ]

        ctor = [
            "mmm = MMM(",
            f"    date_column={spec.date_column!r},",
            "    channel_columns=channel_columns,",
            f"    target_column={spec.target_column!r},",
            f"    adstock={adstock_cls}(l_max={l_max}),",
            f"    saturation={saturation_cls}(),",
        ]
        if controls:
            ctor.append("    control_columns=control_columns,")
        if spec.seasonality.yearly_fourier_modes:
            ctor.append(f"    yearly_seasonality={spec.seasonality.yearly_fourier_modes},")
        if spec.geo.is_panel and spec.geo.geo_column:
            ctor.append(f"    dims=({spec.geo.geo_column!r},),  # tuple, not a bare string")
        if spec.trend.kind == TrendKind.time_varying_intercept:
            ctor.append("    time_varying_intercept=True,")
        if spec.architecture.time_varying_media:
            ctor.append("    time_varying_media=True,")
        if spec.architecture.link == LinkFunction.log:
            ctor.append("    link='log',  # multiplicative model; components combine on the log scale")
        ctor.append(f"    adstock_first={spec.architecture.adstock_first},")
        if priors and priors.get("model_config"):
            ctor.append("    model_config=model_config,")
        ctor.append(")")
        lines += ctor + [""]

        lines += [
            "# 1. Prior predictive check BEFORE fitting.",
            "mmm.build_model(X, y)",
            "prior = mmm.sample_prior_predictive(X=X, y=y, samples=500)",
            "",
        ]

        if spec.experiments:
            lines += [
                "# 2. Calibrate with experiments. This is what separates an MMM you can act",
                "#    on from a curve fit: it constrains the saturation curve at a point that",
                "#    was actually measured.",
                "df_lift_test = pd.DataFrame(",
                "    [",
            ]
            for e in spec.experiments:
                if e.lift_absolute is None or e.lift_se is None or e.spend_during_test is None:
                    continue
                lines.append(
                    "        {"
                    f"'channel': {e.channel!r}, 'x': {e.baseline_spend or 0.0}, "
                    f"'delta_x': {e.spend_during_test}, 'delta_y': {e.lift_absolute}, "
                    f"'sigma': {e.lift_se}"
                    "},"
                )
            lines += [
                "    ]",
                ")",
                "mmm.add_lift_test_measurements(df_lift_test)",
                "",
            ]

        s = spec.sampler
        lines += [
            f"# {'3' if spec.experiments else '2'}. Fit.",
            "mmm.fit(",
            "    X, y,",
            f"    draws={s.draws}, tune={s.tune}, chains={s.chains},",
            f"    target_accept={s.target_accept}, random_seed={s.random_seed},",
            ")",
            "",
            "# Posterior predictive is returned NORMALISED; rescale before any metric.",
            "pp = mmm.sample_posterior_predictive(X, extend_idata=True)",
            "target_scale = float(mmm.get_scales_as_xarray()['target_scale'])",
            f"y_hat = pp[{spec.target_column!r}] * target_scale",
            "",
            "# Attribution. Counterfactual incrementality is link-aware and carryover-aware;",
            "# dividing raw contributions by spend is not.",
            "roas = mmm.incrementality.contribution_over_spend(frequency='quarterly')",
            "contributions = mmm.summary.contributions()",
            "",
            "mmm.save('mmm-workspace/model.nc')  # needs h5netcdf or netCDF4 installed",
        ]
        return "\n".join(lines) + "\n"
