"""Gemini backend (google-genai SDK) — the default for the agent loop.

Rate-limit aware: the free tier is a few requests per minute, and the agent
fires triage+review calls in a tight loop, so 429 RESOURCE_EXHAUSTED is
expected. Transient throttling is retried with the server's ``retryDelay``
when present (else exponential backoff), each wait capped. A per-*day* quota
hit, by contrast, fails fast as ``DailyQuotaError`` — retrying a daily cap
just burns the budget. ``sleep`` is injectable so backoff is unit-tested
without real waiting (same approach as the GitHub client). Backoff and quota
classification are shared with the Cerebras backend via ``models.retry``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Never

from google import genai
from google.genai import types as genai_types

from pr_review_agent.models.base import DailyQuotaError, ModelError, ModelMessage, ModelResponse
from pr_review_agent.models.retry import (
    TransientError,
    call_with_backoff,
    is_daily_quota,
    is_retryable_code,
    is_retryable_text,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """ModelClient backed by the Gemini API, with rate-limit backoff."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        client: genai.Client | None = None,
        max_retries: int = 5,
        max_sleep_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._max_sleep_seconds = max_sleep_seconds
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        contents = [
            genai_types.Content(
                role="user" if message.role == "user" else "model",
                parts=[genai_types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.0,
            # Gemini 2.5 models think by default, and thinking tokens are drawn
            # from max_output_tokens — a small structured budget gets consumed
            # by reasoning, truncating the JSON. These calls want deterministic
            # JSON, not reasoning, so disable thinking.
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )

        def _attempt() -> ModelResponse:
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except Exception as exc:  # the SDK raises a wide family of errors
                _raise_classified(exc)
            usage = response.usage_metadata
            return ModelResponse(
                text=response.text or "",
                model=self._model,
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            )

        return call_with_backoff(
            _attempt,
            label="Gemini",
            purpose=purpose,
            max_retries=self._max_retries,
            max_sleep_seconds=self._max_sleep_seconds,
            sleep=self._sleep,
            logger=logger,
        )


def _raise_classified(exc: Exception) -> Never:
    """Translate a google-genai error into a retry/daily/fatal signal."""
    text = str(exc)
    if is_daily_quota(text):
        raise DailyQuotaError(
            "Gemini daily quota exhausted; resume later (cached calls replay free): "
            f"{text[:200]}"
        ) from exc
    if is_retryable_code(getattr(exc, "code", None)) or is_retryable_text(text):
        raise TransientError(text, retry_after=parse_retry_after(text)) from exc
    raise ModelError(f"Gemini request failed: {exc}") from exc
