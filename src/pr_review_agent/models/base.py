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


class DailyQuotaError(ModelError):
    """A per-*day* quota was exhausted.

    Distinct from transient per-minute throttling: the window resets in hours,
    not seconds, so retrying within a run is pure waste. The eval loop catches
    this to stop cleanly and resume later — the model-call cache replays the
    work already done at $0.
    """


class ContextLimitError(ModelError):
    """The estimated prompt would exceed the model's context window.

    Raised *before* the API call so an oversized PR fails loud (with the token
    count) rather than being silently truncated by the backend. Carries the
    numbers so the caller, which knows the PR number, can log and categorize.
    """

    def __init__(
        self, *, estimated_prompt_tokens: int, max_tokens: int, limit: int, model: str
    ) -> None:
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.max_tokens = max_tokens
        self.limit = limit
        self.model_name = model
        super().__init__(
            f"estimated prompt {estimated_prompt_tokens} + max_tokens {max_tokens} "
            f"= {estimated_prompt_tokens + max_tokens} tokens exceeds {model} "
            f"context limit {limit}"
        )


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
