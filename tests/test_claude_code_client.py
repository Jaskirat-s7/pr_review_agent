"""Tests for the Claude Code (`claude -p`) backend, with an injected runner."""

from __future__ import annotations

import json

import pytest

from pr_review_agent.models.base import ModelError, ModelMessage
from pr_review_agent.models.claude_code import ClaudeCodeClient

MESSAGES = [ModelMessage("user", "judge this")]


def test_builds_command_and_parses_usage() -> None:
    captured: dict[str, object] = {}

    def runner(args: list[str], prompt: str, timeout: float) -> str:
        captured["args"] = args
        captured["prompt"] = prompt
        return json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": '{"verdict": "match"}',
                "usage": {"input_tokens": 120, "output_tokens": 8},
                "total_cost_usd": 0.0,
            }
        )

    client = ClaudeCodeClient(claude_model="opus", runner=runner)
    response = client.complete("be strict", MESSAGES, purpose="judge")
    assert response.text == '{"verdict": "match"}'
    assert response.model == "claude-code"
    assert (response.input_tokens, response.output_tokens) == (120, 8)

    args = captured["args"]
    assert isinstance(args, list)
    assert args[:5] == ["claude", "-p", "--output-format", "json", "--max-turns"]
    assert "--append-system-prompt" in args
    assert args[args.index("--append-system-prompt") + 1] == "be strict"
    assert args[args.index("--model") + 1] == "opus"
    assert captured["prompt"] == "judge this"  # user turn goes via stdin


def test_no_model_flag_when_unset() -> None:
    def runner(args: list[str], prompt: str, timeout: float) -> str:
        assert "--model" not in args
        return json.dumps({"is_error": False, "result": "ok", "usage": {}})

    client = ClaudeCodeClient(runner=runner)
    response = client.complete("sys", MESSAGES)
    assert response.text == "ok"
    assert response.input_tokens == 0  # missing usage tolerated


def test_missing_executable_raises_model_error() -> None:
    def runner(args: list[str], prompt: str, timeout: float) -> str:
        raise FileNotFoundError("claude")

    client = ClaudeCodeClient(runner=runner)
    with pytest.raises(ModelError, match="not found on PATH"):
        client.complete("sys", MESSAGES)


def test_cli_error_envelope_raises() -> None:
    def runner(args: list[str], prompt: str, timeout: float) -> str:
        return json.dumps({"is_error": True, "result": "usage limit reached"})

    client = ClaudeCodeClient(runner=runner)
    with pytest.raises(ModelError, match="usage limit reached"):
        client.complete("sys", MESSAGES)


def test_non_json_output_raises() -> None:
    def runner(args: list[str], prompt: str, timeout: float) -> str:
        return "not json at all"

    client = ClaudeCodeClient(runner=runner)
    with pytest.raises(ModelError, match="non-JSON"):
        client.complete("sys", MESSAGES)
