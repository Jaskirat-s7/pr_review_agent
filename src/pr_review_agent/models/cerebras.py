"""Cerebras backend — OpenAI-compatible inference over plain httpx.

Talks to ``https://api.cerebras.ai/v1`` the same way :class:`OllamaClient`
talks to a local server: an injectable ``transport`` keeps the suite offline.
Three Cerebras-specific concerns shape this client:

- **Dynamic model selection.** Cerebras rotates and deprecates models, so a
  single hardcoded id rots. At startup we query ``GET /models`` and pick the
  first available id from a configured preference list, failing loud if none
  match.
- **Context-limit guard.** The free tier caps context at 8,192 tokens. Before
  each call we sum the estimated prompt tokens (plus the reserved output
  budget); if it would exceed the limit we raise :class:`ContextLimitError`
  rather than let the API truncate silently. The caller (which knows the PR
  number) logs and categorizes it.
- **Quota classification.** Backoff and per-day-vs-per-minute detection are
  shared with the Gemini backend via ``models.retry``, so a daily-cap hit
  fails fast and resumable instead of burning retries.

Cost: the free tier is $0. Pricing in config carries Cerebras's *list* price
so the cost-per-PR column stays API-list-equivalent (the cost-at-scale story);
the report prose notes actual marginal spend was $0 — same split as the Gemini
free-tier and Claude Code subscription runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from pr_review_agent.estimate import estimate_tokens
from pr_review_agent.models.base import (
    ContextLimitError,
    DailyQuotaError,
    ModelError,
    ModelMessage,
    ModelResponse,
)
from pr_review_agent.models.retry import (
    TransientError,
    call_with_backoff,
    is_daily_quota,
    parse_retry_after,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
# Ordered by preference; the first id the API actually offers wins.
DEFAULT_MODEL_PREFERENCES = ("qwen-3-32b", "llama-4-scout")
# Free-tier context window (prompt + completion), in tokens.
DEFAULT_CONTEXT_LIMIT = 8192


class CerebrasClient:
    """ModelClient backed by Cerebras's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model_preferences: Sequence[str] = DEFAULT_MODEL_PREFERENCES,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        model: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 5,
        max_sleep_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._preferences = tuple(model_preferences)
        self._context_limit = context_limit
        self._max_retries = max_retries
        self._max_sleep_seconds = max_sleep_seconds
        self._sleep = sleep
        # Resolve eagerly at startup so an empty/incompatible model list fails
        # before any review work begins. ``model`` overrides discovery (tests).
        self._model = model or self._select_model()

    @property
    def model(self) -> str:
        return self._model

    @property
    def context_limit(self) -> int:
        return self._context_limit

    def close(self) -> None:
        self._client.close()

    def _select_model(self) -> str:
        try:
            response = self._client.get("/models")
        except httpx.HTTPError as exc:
            raise ModelError(f"Cerebras model discovery failed: {exc}") from exc
        if response.status_code >= 400:
            raise ModelError(
                f"Cerebras model discovery failed: HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        data = response.json()
        entries = data.get("data", []) if isinstance(data, dict) else []
        offered: list[str] = [
            entry["id"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
        for preference in self._preferences:
            if preference in offered:
                logger.info("Cerebras model selected: %s", preference)
                return preference
        raise ModelError(
            f"none of the preferred Cerebras models {list(self._preferences)} are available; "
            f"models offered: {sorted(offered)}. Update [models].cerebras_models in config.toml."
        )

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        est_prompt = estimate_tokens(system + "".join(m.content for m in messages))
        # Reserve room for the completion: the cap covers prompt + output.
        if est_prompt + max_tokens > self._context_limit:
            raise ContextLimitError(
                estimated_prompt_tokens=est_prompt,
                max_tokens=max_tokens,
                limit=self._context_limit,
                model=self._model,
            )
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
        }

        def _attempt() -> ModelResponse:
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise TransientError(f"connection error: {exc}") from exc
            if response.status_code == 429 or response.status_code >= 500:
                body = response.text
                if is_daily_quota(body):
                    raise DailyQuotaError(
                        "Cerebras daily quota exhausted; resume later (cached calls "
                        f"replay free): {body[:200]}"
                    )
                retry_after = _retry_after_header(response) or parse_retry_after(body)
                raise TransientError(
                    f"HTTP {response.status_code}: {body[:200]}", retry_after=retry_after
                )
            if response.status_code >= 400:
                raise ModelError(
                    f"Cerebras request failed with HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return self._parse(response)

        return call_with_backoff(
            _attempt,
            label="Cerebras",
            purpose=purpose,
            max_retries=self._max_retries,
            max_sleep_seconds=self._max_sleep_seconds,
            sleep=self._sleep,
            logger=logger,
        )

    def _parse(self, response: httpx.Response) -> ModelResponse:
        data = response.json()
        if not isinstance(data, dict):
            raise ModelError("Cerebras returned a non-object JSON response")
        choices = data.get("choices")
        text = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text = str(message.get("content") or "")
        raw_usage = data.get("usage")
        usage: dict[str, object] = raw_usage if isinstance(raw_usage, dict) else {}
        return ModelResponse(
            text=text,
            model=self._model,
            input_tokens=_int(usage, "prompt_tokens"),
            output_tokens=_int(usage, "completion_tokens"),
        )


def _retry_after_header(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(usage: dict[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
