"""Backend protocol: one spec, many frameworks.

A backend translates an :class:`~agent_mmm.spec.MMMSpec` into a concrete model.
Two capability levels exist and the difference is honest rather than cosmetic:

``executable``
    The backend can build and fit the model in this Python process.
    pymc-marketing is executable.

``codegen``
    The backend emits runnable, framework-idiomatic code plus the data contract
    it requires, but cannot run it here (Meridian needs TensorFlow, Robyn needs
    R). Generated code is reviewed and run by the user in the right environment.

Every backend must implement ``translate()``, which returns a
:class:`TranslationResult`. That result is the single artefact downstream tools
(reports, agents, slash commands) consume, so a codegen backend and an
executable one are interchangeable for planning purposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from agent_mmm.spec import MMMSpec


class Capability(str, Enum):
    executable = "executable"
    codegen = "codegen"


@dataclass
class TranslationResult:
    """What a backend produces from a spec."""

    framework: str
    capability: Capability
    code: str
    """Runnable code for the target framework."""

    data_contract: dict[str, Any] = field(default_factory=dict)
    """Columns, shapes, and dtypes the framework requires of the dataset."""

    structure: dict[str, Any] = field(default_factory=dict)
    """Resolved model structure — per-channel transforms, dims, link, etc."""

    unsupported: list[str] = field(default_factory=list)
    """Spec features this framework cannot express. Never silently dropped:
    every entry here must be surfaced to the user, because a spec feature that
    vanishes without comment is how a model ends up not meaning what its author
    thinks it means."""

    warnings: list[str] = field(default_factory=list)
    model: Any = None
    """The built model object, for executable backends only."""


@runtime_checkable
class MMMBackend(Protocol):
    name: str
    capability: Capability

    def translate(self, spec: MMMSpec, **kwargs: Any) -> TranslationResult:
        """Compile a spec into framework-specific code (and a model, if executable)."""
        ...
