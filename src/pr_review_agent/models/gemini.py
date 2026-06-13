"""Gemini backend (google-genai SDK) — the default for the agent loop.

Rate-limit aware: the free tier is a few requests per minute, and the agent
fires triage+review calls in a tight loop, so 429 RESOURCE_EXHAUSTED is
expected. Retries honor the server's ``retryDelay`` when present, fall back
to exponential backoff otherwise, and cap each wait. ``sleep`` is injectable
so the backoff is unit-tested without real waiting (same approach as the
GitHub client).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence

from google import genai
from google.genai import types as genai_types

from pr_review_agent.models.base import ModelError, ModelMessage, ModelResponse

logger = logging.getLogger(__name__)

# Matches the RetryInfo hint Gemini returns, e.g. "'retryDelay': '34s'".
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s")
# Retryable when not exposed as a numeric .code attribute.
_RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "429")
_RETRYABLE_CODES = frozenset({429, 500, 503})


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
            # Gemini 2.5 models think by default, and thinking tokens are
            # drawn from max_output_tokens — a small structured budget gets
            # consumed by reasoning, truncating the JSON. These calls want
            # deterministic JSON, not reasoning, so disable thinking.
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except Exception as exc:  # the SDK raises a wide family of errors
                if attempt < self._max_retries and _is_retryable(exc):
                    delay = self._retry_delay(exc, attempt)
                    logger.warning(
                        "Gemini %s call rate-limited/unavailable; retrying in %.1fs "
                        "(attempt %d/%d): %s",
                        purpose or "model",
                        delay,
                        attempt + 1,
                        self._max_retries,
                        _short(exc),
                    )
                    self._sleep(delay)
                    continue
                raise ModelError(f"Gemini request failed: {exc}") from exc
            usage = response.usage_metadata
            return ModelResponse(
                text=response.text or "",
                model=self._model,
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            )
        raise ModelError("Gemini request failed after exhausting retries")  # pragma: no cover

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        match = _RETRY_DELAY_RE.search(str(exc))
        if match is not None:
            delay = float(match.group(1)) + 1.0  # small cushion past the window reset
        else:
            delay = min(2.0**attempt, self._max_sleep_seconds)
        return min(max(delay, 1.0), self._max_sleep_seconds)


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_CODES:
        return True
    text = str(exc)
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _short(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:160]
