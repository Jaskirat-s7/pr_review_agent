"""Claude Code backend: shell out to the `claude` CLI under a Pro/Max plan.

Used for the eval judge so judging draws on a Claude Code subscription
instead of metered API credits. Usage is logged into the same SQLite ledger
as every other backend, but cost is recorded as $0 — the ledger's purpose is
*API* spend, and these calls are covered by the subscription, not billed per
token. The model label is the fixed string "claude-code" (mapped to $0
pricing) so the underlying plan model can change without reshaping the
ledger.

Calls are single-shot (`--max-turns 1`): the system prompt goes via
`--append-system-prompt`, the user turns via stdin, and `--output-format
json` gives a parseable envelope with token usage.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence

from pr_review_agent.jsonutil import parse_model_json
from pr_review_agent.models.base import ModelError, ModelMessage, ModelResponse

CLAUDE_CODE_MODEL = "claude-code"

# (args, stdin_prompt, timeout) -> stdout
Runner = Callable[[list[str], str, float], str]


class ClaudeCodeClient:
    """ModelClient backed by the local `claude` CLI in print mode."""

    def __init__(
        self,
        *,
        claude_model: str = "",
        executable: str = "claude",
        timeout: float = 300.0,
        runner: Runner | None = None,
    ) -> None:
        self._claude_model = claude_model
        self._executable = executable
        self._timeout = timeout
        self._runner = runner or _run_subprocess

    @property
    def model(self) -> str:
        return CLAUDE_CODE_MODEL

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        prompt = "\n\n".join(m.content for m in messages)
        args = [self._executable, "-p", "--output-format", "json", "--max-turns", "1"]
        if system:
            args += ["--append-system-prompt", system]
        if self._claude_model:
            args += ["--model", self._claude_model]
        try:
            stdout = self._runner(args, prompt, self._timeout)
        except FileNotFoundError as exc:
            raise ModelError(f"claude executable {self._executable!r} not found on PATH") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ModelError(f"claude -p failed: {exc}") from exc

        data = parse_model_json(stdout)
        if not isinstance(data, dict):
            raise ModelError(f"claude -p returned non-JSON output: {stdout[:200]!r}")
        if data.get("is_error"):
            raise ModelError(f"claude -p reported an error: {str(data.get('result'))[:200]}")
        result = data.get("result")
        if not isinstance(result, str):
            raise ModelError("claude -p response is missing a string 'result'")
        raw_usage = data.get("usage")
        usage: dict[str, object] = raw_usage if isinstance(raw_usage, dict) else {}
        return ModelResponse(
            text=result,
            model=CLAUDE_CODE_MODEL,
            input_tokens=_int(usage, "input_tokens"),
            output_tokens=_int(usage, "output_tokens"),
        )


def _run_subprocess(args: list[str], prompt: str, timeout: float) -> str:
    result = subprocess.run(
        args, input=prompt, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ModelError(f"claude -p exited {result.returncode}: {detail[:200]}")
    return result.stdout


def _int(usage: dict[str, object], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
