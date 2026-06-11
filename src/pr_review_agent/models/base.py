"""The ModelClient protocol every backend implements.

Deliberately minimal: one synchronous text-in/text-out call. No temperature
knob — current Anthropic models (Opus 4.7+) reject sampling parameters
outright, and determinism for repeated runs comes from the cache layer, not
from sampling settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


class ModelError(Exception):
    """A model backend request failed."""


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """One conversation turn."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A completed model call with API-reported token usage."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


class ModelClient(Protocol):
    """A pluggable model backend.

    ``purpose`` is a telemetry label ("triage", "review", "judge") recorded
    by the caching layer; backends ignore it.
    """

    @property
    def model(self) -> str: ...

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse: ...
