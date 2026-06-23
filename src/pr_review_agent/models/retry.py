"""Shared rate-limit backoff and quota classification for model backends.

Two failure modes are distinguished, because they want opposite responses:

- **Transient throttling** — per-minute rate limits, 5xx, "overloaded". The
  window clears in seconds, so retry: honor the server's retry hint when
  present, else exponential backoff, each wait capped. Backends signal this by
  raising :class:`TransientError`.
- **Daily-quota exhaustion** — a per-*day* cap. The window resets in hours, so
  retrying is pure waste (this is exactly what burned five retries per call on
  the Gemini free tier). Backends raise :class:`~pr_review_agent.models.base.DailyQuotaError`
  and the loop re-raises immediately, no sleeping.

:func:`is_daily_quota` recognizes both providers' daily signals: the Gemini
free tier reports it as RESOURCE_EXHAUSTED with a quotaId containing
"PerDay"/"FreeTier"; an OpenAI-compatible backend (Cerebras) reports a 429
whose body names a per-day request/token limit.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TypeVar

from pr_review_agent.models.base import ModelError

T = TypeVar("T")

# Server retry hints, in seconds: Gemini's "retryDelay": "34s", an OpenAI-style
# "retry_after": 12, or a Retry-After header surfaced into the message.
_RETRY_DELAY_RE = re.compile(
    r"retry[-_ ]?(?:delay|after)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE
)
# Markers that mean a *daily* cap specifically — fail fast, do not retry.
# Matched after normalizing "_"/"-" to spaces, so "PerDay", "per-day",
# "tokens_per_day", and "requests per day" all hit "per day"/"perday".
_DAILY_MARKERS = ("per day", "perday", "daily", "day limit")
# Markers/codes that mean a *transient* condition — retry with backoff.
_RETRYABLE_MARKERS = (
    "resource_exhausted",
    "unavailable",
    "overloaded",
    "too_many_requests",
    "rate limit",
    "ratelimit",
    "429",
)
_RETRYABLE_CODES = frozenset({429, 500, 503})


class TransientError(Exception):
    """A retryable throttle/availability error, with an optional server hint."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def is_daily_quota(text: str) -> bool:
    """True if an error string names a per-day quota (vs per-minute throttling)."""
    low = text.lower().replace("_", " ").replace("-", " ")
    return any(marker in low for marker in _DAILY_MARKERS)


def is_retryable_code(code: object) -> bool:
    return isinstance(code, int) and code in _RETRYABLE_CODES


def is_retryable_text(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _RETRYABLE_MARKERS)


def parse_retry_after(text: str) -> float | None:
    match = _RETRY_DELAY_RE.search(text)
    return float(match.group(1)) if match is not None else None


def backoff_delay(retry_after: float | None, attempt: int, max_sleep: float) -> float:
    """Server hint + 1s cushion when present, else exponential; clamped to a cap."""
    delay = retry_after + 1.0 if retry_after is not None else 2.0**attempt
    return min(max(delay, 1.0), max_sleep)


def call_with_backoff(
    fn: Callable[[], T],
    *,
    label: str,
    purpose: str,
    max_retries: int,
    max_sleep_seconds: float,
    sleep: Callable[[float], None],
    logger: logging.Logger,
) -> T:
    """Run ``fn`` with shared retry semantics.

    ``fn`` must translate backend errors into :class:`TransientError` (retry),
    :class:`~pr_review_agent.models.base.DailyQuotaError` (fail fast), or
    :class:`~pr_review_agent.models.base.ModelError` (non-retryable). Anything
    else propagates unchanged.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except TransientError as exc:
            if attempt >= max_retries:
                raise ModelError(
                    f"{label} request failed after {max_retries} retries: {exc}"
                ) from exc
            delay = backoff_delay(exc.retry_after, attempt, max_sleep_seconds)
            logger.warning(
                "%s %s call rate-limited/unavailable; retrying in %.1fs (attempt %d/%d): %s",
                label,
                purpose or "model",
                delay,
                attempt + 1,
                max_retries,
                str(exc).replace("\n", " ")[:160],
            )
            sleep(delay)
    raise ModelError(f"{label} request failed after exhausting retries")  # pragma: no cover
