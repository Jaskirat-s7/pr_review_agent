"""Tests for the three ModelClient backends (no live API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import pytest

from pr_review_agent.models.anthropic_client import AnthropicClient
from pr_review_agent.models.base import DailyQuotaError, ModelError, ModelMessage
from pr_review_agent.models.gemini import GeminiClient
from pr_review_agent.models.ollama import OllamaClient

MESSAGES = [ModelMessage("user", "hi")]


def test_ollama_sends_system_first_and_parses_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello!"},
                "prompt_eval_count": 12,
                "eval_count": 7,
            },
        )

    client = OllamaClient(model="test-model", transport=httpx.MockTransport(handler))
    response = client.complete("sys", MESSAGES)
    assert response.text == "hello!"
    assert (response.input_tokens, response.output_tokens) == (12, 7)
    assert response.model == "test-model"
    client.close()


def test_ollama_http_error_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = OllamaClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="HTTP 500"):
        client.complete("sys", MESSAGES)
    client.close()


def test_ollama_connection_error_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = OllamaClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="refused"):
        client.complete("sys", MESSAGES)
    client.close()


def test_anthropic_parses_text_blocks_and_usage() -> None:
    captured: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text="judged."),
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=9),
        )

    stub = SimpleNamespace(messages=SimpleNamespace(create=create))
    client = AnthropicClient("key", "claude-opus-4-8", client=cast("anthropic.Anthropic", stub))
    response = client.complete("sys", MESSAGES, max_tokens=512)
    assert response.text == "judged."
    assert (response.input_tokens, response.output_tokens) == (50, 9)
    assert captured["model"] == "claude-opus-4-8"
    assert captured["system"] == "sys"
    assert captured["max_tokens"] == 512
    assert "temperature" not in captured  # removed on Opus 4.7+; sending it would 400


def test_anthropic_error_wrapped_as_model_error() -> None:
    def create(**kwargs: Any) -> Any:
        raise anthropic.AnthropicError("rate limited")

    stub = SimpleNamespace(messages=SimpleNamespace(create=create))
    client = AnthropicClient("key", client=cast("anthropic.Anthropic", stub))
    with pytest.raises(ModelError, match="rate limited"):
        client.complete("sys", MESSAGES)


def test_gemini_parses_text_and_usage() -> None:
    captured: dict[str, Any] = {}

    def generate_content(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            text="flash says hi",
            usage_metadata=SimpleNamespace(prompt_token_count=30, candidates_token_count=4),
        )

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client = GeminiClient("key", "gemini-2.5-flash", client=cast("Any", stub))
    response = client.complete("sys", MESSAGES)
    assert response.text == "flash says hi"
    assert (response.input_tokens, response.output_tokens) == (30, 4)
    assert captured["model"] == "gemini-2.5-flash"


def test_gemini_error_wrapped_as_model_error() -> None:
    def generate_content(**kwargs: Any) -> Any:
        raise RuntimeError("bad request: invalid argument")

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient("key", client=cast("Any", stub), sleep=sleeps.append)
    with pytest.raises(ModelError, match="invalid argument"):
        client.complete("sys", MESSAGES)
    assert sleeps == []  # non-retryable: no backoff


class _RateLimitError(Exception):
    """Stands in for a google-genai 429; carries a numeric .code."""

    def __init__(self, message: str, code: int = 429) -> None:
        super().__init__(message)
        self.code = code


def _ok() -> Any:
    return SimpleNamespace(
        text="ok", usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=2)
    )


def test_gemini_retries_429_and_honors_retry_delay() -> None:
    calls = 0

    def generate_content(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _RateLimitError("429 RESOURCE_EXHAUSTED ... 'retryDelay': '34s' ...", code=429)
        return _ok()

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient("key", client=cast("Any", stub), sleep=sleeps.append)
    response = client.complete("sys", MESSAGES)
    assert response.text == "ok"
    assert calls == 2
    assert sleeps == [35.0]  # retryDelay 34s + 1s cushion


def test_gemini_retry_delay_capped_at_max_sleep() -> None:
    def generate_content(**kwargs: Any) -> Any:
        raise _RateLimitError("RESOURCE_EXHAUSTED 'retryDelay': '300s'", code=429)

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient(
        "key",
        client=cast("Any", stub),
        max_retries=1,
        max_sleep_seconds=60.0,
        sleep=sleeps.append,
    )
    with pytest.raises(ModelError, match="Gemini request failed"):
        client.complete("sys", MESSAGES)
    assert sleeps == [60.0]  # 300s clamped to the cap


def test_gemini_exponential_backoff_without_retry_hint() -> None:
    calls = 0

    def generate_content(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise _RateLimitError("503 UNAVAILABLE: the model is overloaded", code=503)
        return _ok()

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient("key", client=cast("Any", stub), sleep=sleeps.append)
    response = client.complete("sys", MESSAGES)
    assert response.text == "ok"
    assert sleeps == [1.0, 2.0]  # 2**0 floored to 1.0, then 2**1


def test_gemini_daily_quota_fails_fast_without_retry() -> None:
    calls = 0

    def generate_content(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        # The free-tier daily cap: quotaId names a per-day quota.
        raise _RateLimitError(
            "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            code=429,
        )

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient("key", client=cast("Any", stub), sleep=sleeps.append)
    with pytest.raises(DailyQuotaError, match="daily quota"):
        client.complete("sys", MESSAGES)
    assert calls == 1  # no retries burned on a daily cap
    assert sleeps == []


def test_gemini_retries_exhausted_raises() -> None:
    def generate_content(**kwargs: Any) -> Any:
        raise _RateLimitError("429 RESOURCE_EXHAUSTED", code=429)

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps: list[float] = []
    client = GeminiClient("key", client=cast("Any", stub), max_retries=2, sleep=sleeps.append)
    with pytest.raises(ModelError, match="Gemini request failed"):
        client.complete("sys", MESSAGES)
    assert len(sleeps) == 2  # two retries, then give up
