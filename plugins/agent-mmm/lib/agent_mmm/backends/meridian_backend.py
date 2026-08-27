"""Google Meridian backend (code generation).

Verified against google-meridian 1.8.0.

Meridian differs from pymc-marketing in ways that change how a spec must be
expressed, not just how it is written:

* **Geo-hierarchical by default.** Meridian is built for geo panels with
  population weighting. A national model is a special case (one geo).
* **ROI-parameterised media priors.** ``media_prior_type="roi"`` puts the prior
  directly on the quantity stakeholders argue about, so calibrating to an
  experiment means setting ``roi_m``, not reverse-engineering a coefficient.
* **First-class reach and frequency.** RF channels have their own tensors and
  an optimal-frequency analysis; no other open framework does this.
* **Explicit non-media treatments and organic media.** Price, promotions and
  organic activity have their own slots, so they cannot accidentally acquire a
  ROAS.
* **Knot-based time effects** instead of Fourier seasonality.
"""
from __future__ import annotations

from typing import Any

from agent_mmm.backends.base import Capability, TranslationResult
from agent_mmm.spec import ChannelRole, MMMSpec, TargetUnitKind, TrendKind
from agent_mmm.utils.moment_match import lognormal_from_range

_ADSTOCK_SUPPORTED = {"geometric", "binomial"}


class MeridianBackend:
    name = "meridian"
    capability = Capability.codegen

    def translate(self, spec: MMMSpec, priors: dict | None = None, **_: Any) -> TranslationResult:
        unsupported: list[str] = []
        warnings: list[str] = []

        media = spec.channels_with_role(ChannelRole.paid_media)
        rf = spec.channels_with_role(ChannelRole.paid_reach_frequency)
        organic = spec.channels_with_role(ChannelRole.organic_media)
        non_media = spec.channels_with_role(ChannelRole.non_media_treatment)

        if not spec.geo.is_panel:
            warnings.append(
                "Spec is national. Meridian runs national models fine, but its hierarchical "
                "geo pooling — the main reason to choose it — is inactive. If geo data exists, "
                "use it: a geo panel multiplies the effective sample size and is what makes "
                "geo-experiment calibration coherent."
            )
        elif not spec.geo.population_column:
            warnings.append(
                "Geo panel without a population column. Meridian requires population for geo "
                "models; without it you cannot use per-capita control scaling and the "
                "baseline_geo choice becomes arbitrary."
            )

        structure_priors = (priors or {}).get("structure", {})
        requested_adstocks = {
            col: s.get("adstock", "geometric") for col, s in structure_priors.items()
        }
        exotic = sorted({a for a in requested_adstocks.values() if a not in _ADSTOCK_SUPPORTED})
        if exotic:
            unsupported.append(
                f"Meridian's adstock_decay_spec supports only 'geometric' and 'binomial'; "
                f"requested {exotic} will be mapped to 'binomial' (which has a delayed peak) "
                "for delayed/Weibull requests. Expect slightly different carryover shapes than "
                "the pymc-marketing run."
            )
        if spec.architecture.link.value == "log":
            unsupported.append(
                "Meridian has no log-link/multiplicative option. Its media terms are additive "
                "on the (population-scaled) KPI. Keep the multiplicative variant in "
                "pymc-marketing if that structure matters."
            )
        if spec.seasonality.yearly_fourier_modes and spec.trend.kind != TrendKind.knots:
            warnings.append(
                "Fourier seasonality has no Meridian equivalent. Time effects are handled by "
                "knots instead — set trend.kind='knots' and choose n_knots. Fewer knots means "
                "more of the seasonal variation is available for media to explain, which is "
                "exactly the trade-off to think about deliberately."
            )

        for ch in media + rf:
            if ch.exposure_column is None and ch.spend_column == ch.column:
                warnings.append(
                    f"{ch.column}: modelled on spend. Meridian prefers an exposure metric "
                    "(impressions/GRPs) as `media` with `media_spend` alongside, so cost "
                    "inflation is not mistaken for delivery."
                )

        code = self._render_code(spec, media, rf, organic, non_media, priors)
        return TranslationResult(
            framework=self.name,
            capability=self.capability,
            code=code,
            data_contract=self.data_contract(spec),
            structure={
                "media_channels": [c.column for c in media],
                "rf_channels": [c.column for c in rf],
                "organic_media": [c.column for c in organic],
                "non_media_treatments": [c.column for c in non_media],
                "controls": [c.column for c in spec.controls],
                "media_prior_type": "roi",
                "knots": spec.trend.n_knots,
                "is_geo": spec.geo.is_panel,
            },
            unsupported=unsupported,
            warnings=warnings,
        )

    @staticmethod
    def data_contract(spec: MMMSpec) -> dict[str, Any]:
        return {
            "shape": "long, one row per (geo, time). A national model uses a single constant geo.",
            "required_columns": {
                "time": spec.date_column,
                "geo": spec.geo.geo_column or "geo (constant for national)",
                "kpi": spec.target_column,
                "population": spec.geo.population_column or "population (required for geo models)",
                "revenue_per_kpi": spec.target_unit.revenue_per_unit_column
                or ("not needed — kpi is revenue" if spec.target_unit.kind == TargetUnitKind.monetary
                    else "recommended for non-revenue KPIs so ROI is in money"),
                "media": [c.model_input_column for c in spec.channels_with_role(ChannelRole.paid_media)],
                "media_spend": [
                    c.spend_column for c in spec.channels_with_role(ChannelRole.paid_media) if c.spend_column
                ],
                "reach": [c.reach_column for c in spec.channels_with_role(ChannelRole.paid_reach_frequency)],
                "frequency": [c.frequency_column for c in spec.channels_with_role(ChannelRole.paid_reach_frequency)],
                "controls": [c.column for c in spec.controls],
                "non_media_treatments": [
                    c.column for c in spec.channels_with_role(ChannelRole.non_media_treatment)
                ],
            },
            "rectangular": True,
            "notes": [
                "kpi_type is 'revenue' or 'non_revenue'. With 'non_revenue', supply "
                "revenue_per_kpi to get ROI in currency instead of KPI units.",
                "media and media_spend may have more time periods than the KPI "
                "(n_media_times >= n_times) so pre-window media still drives adstock.",
                "Every (geo, time) combination must be present — Meridian requires a "
                "rectangular panel and will not impute gaps.",
            ],
        }

    def _render_code(self, spec, media, rf, organic, non_media, priors) -> str:
        roi_priors = (priors or {}).get("roi_priors", {})
        n_knots = spec.trend.n_knots

        def _q(items):
            return "[" + ", ".join(repr(i) for i in items if i) + "]"

        lines = [
            '"""Generated by agent-mmm for google-meridian >= 1.8."""',
            "import numpy as np",
            "import pandas as pd",
            "import tensorflow_probability as tfp",
            "from meridian.data import load",
            "from meridian.model import model, prior_distribution, spec as model_spec",
            "from meridian.analysis import analyzer, optimizer, summarizer",
            "",
            "tfd = tfp.distributions",
            "",
            f"df = pd.read_csv({spec.data_path!r})",
            "",
            "# ---------------------------------------------------------------- #",
            "# 1. Declare what each column IS. Meridian keeps paid media, organic",
            "#    media, non-media treatments and controls in separate slots, which",
            "#    is why a price column here can never accidentally get a ROAS.",
            "# ---------------------------------------------------------------- #",
            "coord_to_columns = load.CoordToColumns(",
            f"    time={spec.date_column!r},",
            f"    geo={(spec.geo.geo_column or 'geo')!r},",
            f"    kpi={spec.target_column!r},",
            f"    population={(spec.geo.population_column or 'population')!r},",
        ]
        if spec.target_unit.revenue_per_unit_column:
            lines.append(f"    revenue_per_kpi={spec.target_unit.revenue_per_unit_column!r},")
        if media:
            lines.append(f"    media={_q([c.model_input_column for c in media])},")
            lines.append(f"    media_spend={_q([c.spend_column for c in media])},")
        if rf:
            lines.append(f"    reach={_q([c.reach_column for c in rf])},")
            lines.append(f"    frequency={_q([c.frequency_column for c in rf])},")
            lines.append(f"    rf_spend={_q([c.spend_column for c in rf])},")
        if organic:
            lines.append(f"    organic_media={_q([c.model_input_column for c in organic])},")
        if non_media:
            lines.append(f"    non_media_treatments={_q([c.column for c in non_media])},")
        if spec.controls:
            lines.append(f"    controls={_q([c.column for c in spec.controls])},")
        lines += [")", ""]

        kpi_type = "revenue" if spec.target_unit.kind == TargetUnitKind.monetary else "non_revenue"
        lines += [
            "loader = load.DataFrameDataLoader(",
            "    df=df,",
            f"    kpi_type={kpi_type!r},",
            "    coord_to_columns=coord_to_columns,",
        ]
        if media:
            mapping = {c.model_input_column: (c.label or c.column) for c in media}
            spend_mapping = {c.spend_column: (c.label or c.column) for c in media if c.spend_column}
            lines.append(f"    media_to_channel={mapping!r},")
            lines.append(f"    media_spend_to_channel={spend_mapping!r},")
        lines += [")", "data = loader.load()", ""]

        lines += [
            "# ---------------------------------------------------------------- #",
            "# 2. Priors. Meridian puts the prior on ROI itself, which means the",
            "#    number in this file is the number a marketer can argue with.",
            "#    Channel order MUST match data.media_channel.",
            "# ---------------------------------------------------------------- #",
        ]
        if roi_priors:
            names, mus, sigmas = [], [], []
            for ch in media:
                col = ch.model_input_column
                entry = roi_priors.get(col)
                if entry is None:
                    mu, sigma = lognormal_from_range(0.3, 6.0, 0.90)
                else:
                    mu, sigma = entry["mu"], entry["sigma"]
                names.append(ch.label or ch.column)
                mus.append(round(float(mu), 4))
                sigmas.append(round(float(sigma), 4))
            lines += [
                f"# channel order: {names}",
                f"roi_mu = np.array({mus})",
                f"roi_sigma = np.array({sigmas})",
                "prior = prior_distribution.PriorDistribution(",
                "    roi_m=tfd.LogNormal(roi_mu, roi_sigma, name='roi_m'),",
                ")",
            ]
        else:
            lines += [
                "# Meridian's default roi_m is LogNormal(0.2, 0.9): a median ROI of ~1.2 with",
                "# a 90% range of roughly 0.3-5.3. Replace it with channel-specific beliefs.",
                "prior = prior_distribution.PriorDistribution()",
            ]
        lines.append("")

        adstock_map = {}
        for col, s in ((priors or {}).get("structure") or {}).items():
            ch = spec.channel_by_name(col)
            if ch is None:
                continue
            requested = s.get("adstock", "geometric")
            adstock_map[ch.label or ch.column] = (
                "geometric" if requested in _ADSTOCK_SUPPORTED else "binomial"
            )

        max_lag = max(
            (s.get("l_max", 8) for s in ((priors or {}).get("structure") or {}).values()),
            default=8,
        )
        lines += [
            "spec_kwargs = dict(",
            "    prior=prior,",
            "    media_prior_type='roi',",
            f"    max_lag={max_lag},",
            "    hill_before_adstock=False,",
        ]
        if adstock_map:
            lines.append(f"    adstock_decay_spec={adstock_map!r},")
        if n_knots:
            lines.append(
                f"    knots={n_knots},  # fewer knots => less flexible baseline => more variance left for media"
            )
        else:
            lines.append(
                "    # knots=None means one knot per period for geo models: a maximally flexible"
            )
            lines.append(
                "    # baseline that can absorb media's own signal. Set an explicit number, or"
            )
            lines.append("    # enable_aks=True to let Meridian choose.")
        if spec.geo.baseline_geo:
            lines.append(f"    baseline_geo={spec.geo.baseline_geo!r},")
        if non_media:
            lines.append("    non_media_treatments_prior_type='contribution',")
        controls_scaled = [c.column for c in spec.controls if c.scale_by_population]
        if controls_scaled:
            lines.append(
                f"    # per-capita controls: {controls_scaled}"
            )
            lines.append(
                "    control_population_scaling_id=np.isin("
                f"data.control_variable, {controls_scaled!r}),"
            )
        if spec.validation.holdout_start:
            lines.append(
                "    # holdout_id masks KPI (not media) for out-of-sample evaluation"
            )
        lines += [")", "", "model_spec_obj = model_spec.ModelSpec(**spec_kwargs)", ""]

        lines += [
            "# ---------------------------------------------------------------- #",
            "# 3. Fit. Meridian runs its own EDA guardrail before sampling and will",
            "#    refuse to fit data with critical issues — read the findings rather",
            "#    than working around them.",
            "# ---------------------------------------------------------------- #",
            "mmm = model.Meridian(input_data=data, model_spec=model_spec_obj)",
            "mmm.sample_prior(500)",
            "mmm.sample_posterior(",
            f"    n_chains={max(spec.sampler.chains, 4)}, n_adapt=500, n_burnin=500, "
            f"n_keep={spec.sampler.draws}, seed={spec.sampler.random_seed},",
            ")",
            "health = mmm.review()  # post-fit model health checks",
            "",
            "# ---------------------------------------------------------------- #",
            "# 4. Analyse.",
            "# ---------------------------------------------------------------- #",
            "an = analyzer.Analyzer(mmm)",
            "an.rhat_summary()",
            "an.predictive_accuracy()",
            "summary = an.summary_metrics()          # ROI, mROI, CPIK, contribution",
            "baseline = an.baseline_summary_metrics()",
            "",
            "# A baseline that goes negative means the model is borrowing from the future",
            "# to pay media. Check it before believing any ROI in this run.",
            "neg = an.negative_baseline_probability()",
            "print('P(baseline < 0):', neg)",
            "",
            "an.response_curves()",
            "an.adstock_decay()",
            "an.hill_curves()",
        ]
        if rf:
            lines += [
                "",
                "# Reach/frequency channels get an optimal-frequency analysis: how many",
                "# times to hit a person, not just how much to spend.",
                "an.optimal_freq()",
            ]
        lines += [
            "",
            "opt = optimizer.BudgetOptimizer(mmm)",
            "results = opt.optimize(fixed_budget=True)",
            "results.output_optimization_summary('optimization.html', '.')",
            "",
            "summarizer.Summarizer(mmm).output_model_results_summary('summary.html', '.')",
            "model.save_mmm(mmm, 'mmm-workspace/meridian_model.pkl')",
        ]
        return "\n".join(lines) + "\n"
