"""Tests for the shared backoff + quota-classification helpers."""

from __future__ import annotations

import logging

import pytest

from pr_review_agent.models.base import DailyQuotaError, ModelError
from pr_review_agent.models.retry import (
    TransientError,
    backoff_delay,
    call_with_backoff,
    is_daily_quota,
    is_retryable_code,
    is_retryable_text,
    parse_retry_after,
)

logger = logging.getLogger("test")


@pytest.mark.parametrize(
    "text",
    [
        "quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        "you exceeded your requests per day quota",
        "Daily token limit reached",
        "tokens_per_day exceeded",
    ],
)
def test_is_daily_quota_true(text: str) -> None:
    assert is_daily_quota(text)


@pytest.mark.parametrize(
    "text",
    [
        "429 RESOURCE_EXHAUSTED (per minute)",
        "rate limit: requests per minute",
        "503 UNAVAILABLE the model is overloaded",
    ],
)
def test_is_daily_quota_false(text: str) -> None:
    assert not is_daily_quota(text)


def test_retryable_detection() -> None:
    assert is_retryable_code(429) and is_retryable_code(503)
    assert not is_retryable_code(400)
    assert is_retryable_text("RESOURCE_EXHAUSTED") and is_retryable_text("overloaded")
    assert not is_retryable_text("invalid argument")


def test_parse_retry_after_formats() -> None:
    assert parse_retry_after("'retryDelay': '34s'") == 34.0
    assert parse_retry_after("retry_after: 12") == 12.0
    assert parse_retry_after("Retry-After = 5s") == 5.0
    assert parse_retry_after("no hint here") is None


def test_backoff_delay_hint_and_cushion_and_cap() -> None:
    assert backoff_delay(34.0, 0, 60.0) == 35.0  # hint + 1s cushion
    assert backoff_delay(300.0, 0, 60.0) == 60.0  # clamped to cap
    assert backoff_delay(None, 0, 60.0) == 1.0  # 2**0 floored to 1.0
    assert backoff_delay(None, 3, 60.0) == 8.0  # 2**3


def test_call_with_backoff_daily_quota_propagates_immediately() -> None:
    sleeps: list[float] = []

    def fn() -> str:
        raise DailyQuotaError("daily cap")

    with pytest.raises(DailyQuotaError):
        call_with_backoff(
            fn,
            label="X",
            purpose="t",
            max_retries=5,
            max_sleep_seconds=60.0,
            sleep=sleeps.append,
            logger=logger,
        )
    assert sleeps == []  # never retried


def test_call_with_backoff_exhausts_then_model_error() -> None:
    sleeps: list[float] = []

    def fn() -> str:
        raise TransientError("throttled", retry_after=None)

    with pytest.raises(ModelError, match="failed after 2 retries"):
        call_with_backoff(
            fn,
            label="X",
            purpose="t",
            max_retries=2,
            max_sleep_seconds=60.0,
            sleep=sleeps.append,
            logger=logger,
        )
    assert sleeps == [1.0, 2.0]  # exponential, two retries
