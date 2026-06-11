"""Tests for the three ModelClient backends (no live API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import pytest

from pr_review_agent.models.anthropic_client import AnthropicClient
from pr_review_agent.models.base import ModelError, ModelMessage
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
        raise RuntimeError("quota exceeded")

    stub = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client = GeminiClient("key", client=cast("Any", stub))
    with pytest.raises(ModelError, match="quota exceeded"):
        client.complete("sys", MESSAGES)
