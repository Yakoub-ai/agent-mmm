"""Framework backends for agent-mmm."""
from __future__ import annotations

from agent_mmm.backends.base import Capability, MMMBackend, TranslationResult
from agent_mmm.spec import Framework

__all__ = ["Capability", "MMMBackend", "TranslationResult", "get_backend", "available_backends"]


def get_backend(framework: Framework | str) -> MMMBackend:
    """Return the backend for a framework."""
    key = framework.value if isinstance(framework, Framework) else str(framework)
    if key in ("pymc-marketing", "pymc_marketing", "pymc"):
        from agent_mmm.backends.pymc_backend import PyMCMarketingBackend

        return PyMCMarketingBackend()
    if key == "meridian":
        from agent_mmm.backends.meridian_backend import MeridianBackend

        return MeridianBackend()
    if key == "robyn":
        from agent_mmm.backends.robyn_backend import RobynBackend

        return RobynBackend()
    raise ValueError(
        f"Unknown framework '{key}'. Supported: pymc-marketing, meridian, robyn."
    )


def available_backends() -> dict[str, str]:
    """Framework name -> capability level."""
    return {
        "pymc-marketing": Capability.executable.value,
        "meridian": Capability.codegen.value,
        "robyn": Capability.codegen.value,
    }
