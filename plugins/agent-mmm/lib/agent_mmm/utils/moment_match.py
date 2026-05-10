"""Moment-matching utilities: convert (mu, sigma) to Beta/Gamma distribution parameters.

Formulas from mmm-model-building/SKILL.md lines 58-69.
"""
from __future__ import annotations


def beta_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """Convert (mu, sigma) → (alpha, beta) for a Beta distribution."""
    if not (0 < mu < 1):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    C = max(mu * (1 - mu) / sigma**2 - 1, 0.5)
    alpha = mu * C
    beta = (1 - mu) * C
    return alpha, beta


def gamma_moment_match(mu: float, sigma: float) -> tuple[float, float]:
    """Convert (mu, sigma) → (alpha, beta) for a Gamma distribution."""
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    alpha = (mu / sigma) ** 2
    beta = mu / sigma**2
    return alpha, beta
