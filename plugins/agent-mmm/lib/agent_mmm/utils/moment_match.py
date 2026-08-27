"""Moment matching: turn beliefs a human can state into distribution parameters.

Nobody has an intuition about ``Beta(2.1, 6.3)``. People do have intuitions about
"about 60% of the effect carries into next week, give or take 15 points" and
"ROAS is probably between 1 and 6". These helpers convert the second kind of
statement into the first, so priors stay auditable.
"""
from __future__ import annotations

import math


def beta_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """Convert (mu, sigma) -> (alpha, beta) for a Beta distribution.

    The concentration ``C = mu(1-mu)/sigma^2 - 1`` is only valid when the
    requested sigma is achievable for that mean; a Beta with mean ``mu`` cannot
    have a standard deviation above ``sqrt(mu(1-mu))``. Rather than fail, we
    floor the concentration at 0.5, which yields the widest sensible prior.
    """
    if not (0 < mu < 1):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    C = max(mu * (1 - mu) / sigma**2 - 1, 0.5)
    return mu * C, (1 - mu) * C


def gamma_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """Convert (mu, sigma) -> (alpha, beta) for a Gamma distribution (rate form)."""
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    alpha = (mu / sigma) ** 2
    beta = mu / sigma**2
    return alpha, beta


def lognormal_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    """LogNormal (mu, sigma) on the log scale from a mean and sd on the natural scale."""
    if mean <= 0:
        raise ValueError(f"mean must be > 0, got {mean}")
    if std <= 0:
        raise ValueError(f"std must be > 0, got {std}")
    var_ratio = (std / mean) ** 2
    mu = math.log(mean) - 0.5 * math.log(var_ratio + 1)
    sigma = math.sqrt(math.log(var_ratio + 1))
    return mu, sigma


def lognormal_from_range(low: float, high: float, mass: float = 0.90) -> tuple[float, float]:
    """LogNormal (mu, sigma) placing ``mass`` of the probability inside [low, high].

    This is the preferred way to state an ROI prior: "I believe this channel's
    ROAS is between 0.5 and 6 with 90% confidence" is a claim a marketer can
    challenge, and it is exactly what frameworks with ROI-parameterised media
    effects (Meridian) want.

    Mirrors Meridian's ``lognormal_dist_from_range``.
    """
    if not (0 < low < high):
        raise ValueError(f"require 0 < low < high, got low={low}, high={high}")
    if not (0 < mass < 1):
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    # Standard-normal quantile via the inverse error function.
    z = math.sqrt(2) * _erfinv(mass)
    sigma = math.log(high / low) / (2 * z)
    mu = math.log(high) - z * sigma
    return mu, sigma


def lognormal_quantile(mu: float, sigma: float, q: float) -> float:
    """Quantile of a LogNormal(mu, sigma) — handy for reporting prior ranges back."""
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1), got {q}")
    z = math.sqrt(2) * _erfinv(2 * q - 1)
    return math.exp(mu + sigma * z)


def _erfinv(x: float) -> float:
    """Inverse error function (Giles' rational approximation, ~1e-7 accurate).

    Implemented here so the prior helpers stay dependency-free; scipy is used
    elsewhere in the package but priors are computed in contexts (report
    rendering, code generation) where importing scipy is unnecessary weight.
    """
    if not (-1 < x < 1):
        raise ValueError(f"erfinv domain is (-1, 1), got {x}")
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (
            3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
            0.00021858087, -0.00125372503, -0.00417768164,
            0.246640727, 1.50140941,
        ):
            p = p * w + c
    else:
        w = math.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (
            0.000100950558, 0.00134934322, -0.00367342844,
            0.00573950773, -0.0076224613, 0.00943887047,
            1.00167406, 2.83297682,
        ):
            p = p * w + c
    return p * x
