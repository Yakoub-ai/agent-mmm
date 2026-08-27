"""Meta Robyn backend (code generation, R).

Verified against Robyn 3.12.1 (CRAN, May 2026).

Robyn is frequentist, not Bayesian, and that single difference propagates
everywhere:

* **No priors.** Beliefs enter as hyperparameter *bounds* and as sign
  constraints (``paid_media_signs``), not as distributions. A spec's priors
  become bounds; the rest of the belief is simply lost, which is why the
  translation reports it rather than pretending otherwise.
* **No credible intervals.** Robyn returns a Pareto front of many models, not a
  posterior. "Uncertainty" is the spread across the Pareto set plus bootstrapped
  intervals on the decomposition — a different object from a posterior, and it
  should not be reported as one.
* **Regularisation replaces shrinkage.** Ridge with an evolutionary search over
  hyperparameters (Nevergrad) does the job priors do elsewhere.
* **Calibration is a first-class objective.** ``calibration_input`` adds a third
  objective (MAPE against lift tests) to NRMSE and DECOMP.RSSD.
"""
from __future__ import annotations

from typing import Any

from agent_mmm.backends.base import Capability, TranslationResult
from agent_mmm.spec import ChannelRole, DataGranularity, ExpectedSign, MMMSpec, TargetUnitKind
from agent_mmm.utils.channel_classifier import halflife_to_alpha

_ADSTOCK_MAP = {
    "geometric": "geometric",
    "delayed": "weibull_pdf",
    "weibull_cdf": "weibull_cdf",
    "weibull_pdf": "weibull_pdf",
    "binomial": "weibull_pdf",
}


def _r_vec(items, quote: bool = True) -> str:
    if not items:
        return "NULL"
    body = ", ".join(f'"{i}"' if quote else str(i) for i in items)
    return f"c({body})"


class RobynBackend:
    name = "robyn"
    capability = Capability.codegen

    def translate(self, spec: MMMSpec, priors: dict | None = None, **_: Any) -> TranslationResult:
        unsupported: list[str] = []
        warnings: list[str] = []

        if spec.geo.is_panel:
            unsupported.append(
                "Robyn has no hierarchical geo model. Either aggregate to national, or fit "
                "one model per geo and lose all cross-geo pooling. If geo structure matters, "
                "Meridian or a pymc-marketing panel model is the right tool."
            )
        if spec.architecture.link.value == "log":
            unsupported.append(
                "Robyn fits a ridge regression on the response scale; there is no log-link "
                "option. Log-transform the target yourself if you need multiplicative structure, "
                "and remember that decomposition then no longer sums to the target."
            )
        if spec.experiments:
            warnings.append(
                "Experiments map to `calibration_input`, which adds MAPE-to-lift as a third "
                "optimisation objective rather than constraining the likelihood. The effect is "
                "softer than a Bayesian lift-test constraint: Robyn prefers models that match "
                "your test, it does not require them to."
            )
        else:
            warnings.append(
                "No calibration_input. Robyn's Pareto front will contain many models that fit "
                "equally well and disagree about channel effects; without lift tests there is "
                "no principled way to choose among them."
            )
        if spec.granularity == DataGranularity.monthly:
            unsupported.append(
                "Robyn expects daily or weekly data; monthly series are too short for its "
                "evolutionary search to be meaningful."
            )

        priors_present = bool((priors or {}).get("per_channel_audit"))
        if priors_present:
            warnings.append(
                "Bayesian priors were translated into hyperparameter bounds. Bounds carry "
                "the centre of your belief but none of its shape or strength — a tight prior "
                "and a weak one with the same range become the same object here."
            )

        code = self._render_code(spec, priors)
        return TranslationResult(
            framework=self.name,
            capability=self.capability,
            code=code,
            data_contract=self.data_contract(spec),
            structure={
                "adstock": self._pick_adstock(priors),
                "paid_media_spends": [
                    c.spend_column for c in spec.channels_with_role(ChannelRole.paid_media) if c.spend_column
                ],
                "organic_vars": [
                    c.column for c in spec.channels_with_role(ChannelRole.organic_media)
                ],
                "context_vars": spec.control_columns(),
            },
            unsupported=unsupported,
            warnings=warnings,
        )

    @staticmethod
    def _pick_adstock(priors: dict | None) -> str:
        """Robyn uses ONE adstock for the whole model."""
        structure = (priors or {}).get("structure") or {}
        kinds = [_ADSTOCK_MAP.get(s.get("adstock", "geometric"), "geometric") for s in structure.values()]
        if not kinds:
            return "geometric"
        return max(set(kinds), key=kinds.count)

    @staticmethod
    def data_contract(spec: MMMSpec) -> dict[str, Any]:
        paid = spec.channels_with_role(ChannelRole.paid_media)
        return {
            "shape": "wide, one row per date, national only",
            "required_columns": {
                "date_var": spec.date_column,
                "dep_var": spec.target_column,
                "paid_media_spends": [c.spend_column for c in paid if c.spend_column],
                "paid_media_vars": [c.model_input_column for c in paid],
                "organic_vars": [c.column for c in spec.channels_with_role(ChannelRole.organic_media)],
                "context_vars": spec.control_columns(),
            },
            "rectangular": False,
            "notes": [
                "paid_media_spends and paid_media_vars must be the same length and aligned; "
                "vars may be exposure metrics while spends are always cost.",
                "dep_var_type is 'revenue' or 'conversion'.",
                "Robyn adds trend/season/holiday via Prophet — supply prophet_country and "
                "prophet_vars rather than hand-building calendar dummies.",
                "Needs >= 2 years of data (Robyn warns below 104 weeks) for the "
                "Prophet decomposition to be stable.",
            ],
        }

    def _render_code(self, spec: MMMSpec, priors: dict | None) -> str:
        paid = spec.channels_with_role(ChannelRole.paid_media)
        organic = spec.channels_with_role(ChannelRole.organic_media)
        controls = spec.control_columns()
        adstock = self._pick_adstock(priors)
        audit = {a["column"]: a for a in (priors or {}).get("per_channel_audit", [])}

        spends = [c.spend_column for c in paid if c.spend_column]
        media_vars = [c.model_input_column for c in paid]
        signs = [
            "positive" if c.expected_sign == ExpectedSign.positive else "default"
            for c in paid
        ]
        context_signs = [
            {"positive": "positive", "negative": "negative", "unconstrained": "default"}[
                c.expected_sign.value
            ]
            for c in spec.controls
        ] + ["default"] * len(spec.channels_with_role(ChannelRole.non_media_treatment))

        dep_var_type = "revenue" if spec.target_unit.kind == TargetUnitKind.monetary else "conversion"

        lines = [
            "# Generated by agent-mmm for Robyn >= 3.12 (R).",
            "library(Robyn)",
            "library(dplyr)",
            "",
            f'dt_input <- read.csv("{spec.data_path}")',
            f'dt_input${spec.date_column} <- as.Date(dt_input${spec.date_column})',
            "",
            "# ------------------------------------------------------------------ #",
            "# 1. Inputs. Robyn separates paid media (spend + optional exposure),",
            "#    organic variables (no spend, so no ROAS), and context variables",
            "#    (confounders and business levers).",
            "# ------------------------------------------------------------------ #",
            "InputCollect <- robyn_inputs(",
            "  dt_input = dt_input,",
            f'  dep_var = "{spec.target_column}",',
            f'  dep_var_type = "{dep_var_type}",',
            f'  date_var = "{spec.date_column}",',
            f"  paid_media_spends = {_r_vec(spends)},",
            f"  paid_media_vars = {_r_vec(media_vars)},",
            f"  paid_media_signs = {_r_vec(signs)},",
            f"  organic_vars = {_r_vec([c.column for c in organic])},",
            f"  context_vars = {_r_vec(controls)},",
            f"  context_signs = {_r_vec(context_signs)},",
        ]
        if spec.seasonality.holiday_country:
            lines += [
                f'  prophet_country = "{spec.seasonality.holiday_country}",',
                '  prophet_vars = c("trend", "season", "holiday"),',
                '  prophet_signs = c("default", "default", "default"),',
            ]
        else:
            lines += [
                '  prophet_vars = c("trend", "season"),  # add "holiday" once prophet_country is set',
            ]
        lines += [
            f'  adstock = "{adstock}",',
            ")",
            "",
            "# ------------------------------------------------------------------ #",
            "# 2. Hyperparameter bounds. Robyn searches inside these with Nevergrad;",
            "#    they are the only place prior knowledge can enter the model.",
            "#      thetas — geometric decay per period (only for adstock='geometric')",
            "#      shapes/scales — Weibull adstock (only for weibull_*)",
            "#      alphas — Hill exponent: >1 gives an S-curve, <1 concave from zero",
            "#      gammas — Hill inflection as a fraction of the channel's range",
            "# ------------------------------------------------------------------ #",
            "hyperparameters <- list(",
        ]
        for ch in paid + organic:
            col = ch.model_input_column
            name = ch.spend_column if ch.role == ChannelRole.paid_media and ch.spend_column else col
            a = audit.get(col, {})
            alpha_mu = a.get("alpha_mu")
            if alpha_mu is None:
                alpha_mu = halflife_to_alpha(ch.halflife_periods or 1.5)
            sd = a.get("alpha_sigma", 0.15)
            lo = max(0.0, round(alpha_mu - 2 * sd, 3))
            hi = min(0.95, round(alpha_mu + 2 * sd, 3))
            if hi <= lo:
                lo, hi = 0.0, 0.5
            if adstock == "geometric":
                lines.append(f"  {name}_thetas = c({lo}, {hi}),")
            else:
                lines.append(f"  {name}_shapes = c(0.5, 3),")
                lines.append(f"  {name}_scales = c(0.01, 0.2),")
            lines.append(f"  {name}_alphas = c(0.5, 3),")
            lines.append(f"  {name}_gammas = c(0.3, 1),")
        lines += [
            "  train_size = c(0.5, 0.8)  # time-series validation split",
            ")",
            "InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)",
            "",
        ]

        if spec.experiments:
            lines += [
                "# ------------------------------------------------------------------ #",
                "# 3. Calibration. Each row is a measured lift; Robyn scores candidate",
                "#    models on how closely they reproduce it, as a third objective",
                "#    alongside NRMSE and DECOMP.RSSD.",
                "# ------------------------------------------------------------------ #",
                "calibration_input <- data.frame(",
            ]
            chans, starts, ends, lifts, spends_c, confs = [], [], [], [], [], []
            for e in spec.experiments:
                if e.lift_absolute is None:
                    continue
                ch = spec.channel_by_name(e.channel)
                chans.append((ch.spend_column if ch and ch.spend_column else e.channel))
                starts.append(str(e.start_date) if e.start_date else "")
                ends.append(str(e.end_date) if e.end_date else "")
                lifts.append(e.lift_absolute)
                spends_c.append(e.spend_during_test or 0)
                confs.append(e.confidence or 0.9)
            lines += [
                f"  channel = {_r_vec(chans)},",
                f"  liftStartDate = as.Date({_r_vec(starts)}),",
                f"  liftEndDate = as.Date({_r_vec(ends)}),",
                f"  liftAbs = {_r_vec(lifts, quote=False)},",
                f"  spend = {_r_vec(spends_c, quote=False)},",
                f"  confidence = {_r_vec(confs, quote=False)},",
                f'  metric = "{spec.target_column}",',
                f'  calibration_scope = "immediate"',
                ")",
                "InputCollect <- robyn_inputs(InputCollect = InputCollect, "
                "calibration_input = calibration_input)",
                "",
            ]

        step = "4" if spec.experiments else "3"
        lines += [
            "# ------------------------------------------------------------------ #",
            f"# {step}. Run. Robyn returns a Pareto front, not one model. Choosing among",
            "#    them is a modelling decision: DECOMP.RSSD penalises allocations that",
            "#    stray far from historical spend shares, which is a prior in disguise.",
            "# ------------------------------------------------------------------ #",
            "OutputModels <- robyn_run(",
            "  InputCollect = InputCollect,",
            "  iterations = 2000,",
            "  trials = 5,",
            "  ts_validation = TRUE,   # holds out the tail; leave TRUE",
            "  add_penalty_factor = FALSE,",
            f"  seed = {spec.sampler.random_seed or 123},",
            ")",
            "",
            "OutputCollect <- robyn_outputs(",
            "  InputCollect, OutputModels,",
            "  pareto_fronts = 'auto',",
            "  csv_out = 'pareto',",
            "  clusters = TRUE,        # groups similar solutions so you compare families, not noise",
            "  export = TRUE,",
            "  plot_folder = './mmm-workspace/robyn/',",
            ")",
            "",
            "# Pick a model, then check convergence and one-pagers before believing it.",
            "select_model <- OutputCollect$allSolutions[1]",
            "",
            "AllocatorCollect <- robyn_allocator(",
            "  InputCollect = InputCollect,",
            "  OutputCollect = OutputCollect,",
            "  select_model = select_model,",
            "  scenario = 'max_response',",
            "  channel_constr_low = 0.7,",
            "  channel_constr_up = 1.5,",
            "  export = TRUE,",
            ")",
            "",
            "robyn_write(InputCollect, OutputCollect, select_model, "
            "dir = './mmm-workspace/robyn/')",
        ]
        return "\n".join(lines) + "\n"
