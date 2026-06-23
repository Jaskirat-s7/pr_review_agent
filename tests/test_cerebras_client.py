"""Tests for the Cerebras backend (no live API calls)."""

from __future__ import annotations

import json

import httpx
import pytest

from pr_review_agent.models.base import (
    ContextLimitError,
    DailyQuotaError,
    ModelError,
    ModelMessage,
)
from pr_review_agent.models.cerebras import CerebrasClient

MESSAGES = [ModelMessage("user", "hi")]


def _models_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


def _chat_response(content: str, prompt_tokens: int, completion_tokens: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def test_selects_first_available_preference() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        # llama offered, qwen not: preference order should pick llama-4-scout.
        return _models_response(["llama-4-scout", "other-model"])

    client = CerebrasClient(
        "csk-test",
        model_preferences=("qwen-3-32b", "llama-4-scout"),
        transport=httpx.MockTransport(handler),
    )
    assert client.model == "llama-4-scout"
    client.close()


def test_no_preferred_model_fails_loud() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _models_response(["some-new-model"])

    with pytest.raises(ModelError, match="none of the preferred Cerebras models"):
        CerebrasClient(
            "csk-test",
            model_preferences=("qwen-3-32b", "llama-4-scout"),
            transport=httpx.MockTransport(handler),
        )


def test_parses_openai_envelope_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response(["qwen-3-32b"])
        captured["payload"] = json.loads(request.content)
        return _chat_response("looks good", prompt_tokens=42, completion_tokens=8)

    client = CerebrasClient("csk-test", transport=httpx.MockTransport(handler))
    response = client.complete("sys", MESSAGES, max_tokens=256)
    assert response.text == "looks good"
    # usage.prompt_tokens -> input_tokens, usage.completion_tokens -> output_tokens
    assert (response.input_tokens, response.output_tokens) == (42, 8)
    assert response.model == "qwen-3-32b"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen-3-32b"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["max_tokens"] == 256
    client.close()


def test_context_guard_raises_before_call_with_token_count() -> None:
    sent = {"chat": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response(["qwen-3-32b"])
        sent["chat"] = True  # must never be reached
        return _chat_response("x", 1, 1)

    client = CerebrasClient(
        "csk-test", context_limit=100, transport=httpx.MockTransport(handler)
    )
    big = [ModelMessage("user", "x" * 1000)]
    with pytest.raises(ContextLimitError) as exc_info:
        client.complete("sys", big, max_tokens=64)
    err = exc_info.value
    assert err.limit == 100
    assert err.estimated_prompt_tokens >= 250  # ~1000 chars / 4
    assert "exceeds" in str(err)
    assert sent["chat"] is False  # guarded before any HTTP call
    client.close()


def test_daily_quota_fails_fast_without_retry() -> None:
    sleeps: list[float] = []
    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response(["qwen-3-32b"])
        calls["chat"] += 1
        return httpx.Response(
            429, json={"error": {"message": "you exceeded your requests per day quota"}}
        )

    client = CerebrasClient(
        "csk-test", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    with pytest.raises(DailyQuotaError, match="daily quota"):
        client.complete("sys", MESSAGES)
    assert calls["chat"] == 1  # no retries burned
    assert sleeps == []
    client.close()


def test_transient_429_retries_then_succeeds() -> None:
    sleeps: list[float] = []
    calls = {"chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response(["qwen-3-32b"])
        calls["chat"] += 1
        if calls["chat"] == 1:
            # Per-minute throttle (no daily marker), with a retry hint.
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                json={"error": {"message": "rate limit: too many requests per minute"}},
            )
        return _chat_response("recovered", 5, 1)

    client = CerebrasClient(
        "csk-test", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    response = client.complete("sys", MESSAGES)
    assert response.text == "recovered"
    assert calls["chat"] == 2
    assert sleeps == [3.0]  # retry-after 2s + 1s cushion
    client.close()


def test_non_retryable_4xx_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return _models_response(["qwen-3-32b"])
        return httpx.Response(400, text="bad request: malformed messages")

    sleeps: list[float] = []
    client = CerebrasClient(
        "csk-test", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    with pytest.raises(ModelError, match="HTTP 400"):
        client.complete("sys", MESSAGES)
    assert sleeps == []  # 4xx (non-429) is not retried
    client.close()
