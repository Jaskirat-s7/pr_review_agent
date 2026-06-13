"""Tests for the eval run module (agent over cases; stubbed model + workspace)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from pr_review_agent.config import ContextConfig, ReviewConfig
from pr_review_agent.evals.run import (
    WorkspaceFactory,
    build_run_result,
    review_case,
    to_run_comments,
)
from pr_review_agent.evals.schema import EvalCase
from pr_review_agent.models.base import ModelMessage, ModelResponse
from pr_review_agent.workspace import WorkspaceError

APP_SOURCE = 'from helpers import greet\n\n\ndef run() -> None:\n    greet("hi")\n'
HELPERS_SOURCE = 'def greet(name: str) -> str:\n    return f"hi {name}"\n'


def _diff() -> str:
    lines = APP_SOURCE.splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    return (
        "diff --git a/app.py b/app.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/app.py\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}\n"
    )


def _case() -> EvalCase:
    return EvalCase(
        repo="octo/widgets",
        number=7,
        title="Use greet",
        review_sha="a" * 40,
        reconstructed=True,
        diff=_diff(),
        human_comments=(),
    )


class _ScriptedModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return "gemini-2.5-flash"

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        self.calls.append(purpose)
        if purpose == "triage":
            text = json.dumps({"worth_reviewing": True, "category": "bug"})
        else:
            text = json.dumps(
                [{"line": 5, "severity": "major", "confidence": 0.9, "comment": "discarded result"}]
            )
        return ModelResponse(text=text, model="gemini-2.5-flash", input_tokens=10, output_tokens=4)


def _workspace_with(head: Path) -> WorkspaceFactory:
    @contextmanager
    def factory(clone_url: str, sha: str, *, token: str | None = None) -> Iterator[Path]:
        yield head

    return factory


@contextmanager
def _failing_workspace(clone_url: str, sha: str, *, token: str | None = None) -> Iterator[Path]:
    # `if clone_url` keeps the yield reachable for the type checker while the
    # raise still fires at runtime (clone_url is always a non-empty string).
    if clone_url:
        raise WorkspaceError("commit gone")
    yield Path()


def test_review_case_uses_context_when_checkout_succeeds(tmp_path: Path) -> None:
    head = tmp_path / "head"
    head.mkdir()
    (head / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (head / "helpers.py").write_text(HELPERS_SOURCE, encoding="utf-8")

    model = _ScriptedModel()
    result = review_case(
        _case(),
        model,
        review_config=ReviewConfig(),
        context_config=ContextConfig(),
        clone_url="https://github.com/octo/widgets.git",
        token=None,
        workspace_factory=_workspace_with(head),
    )
    (comment,) = result.comments
    assert comment.line == 5
    assert comment.has_context  # greet() was resolved from the checked-out repo
    assert "triage" in model.calls and "review" in model.calls


def test_review_case_falls_back_without_context_on_checkout_failure() -> None:
    model = _ScriptedModel()
    result = review_case(
        _case(),
        model,
        review_config=ReviewConfig(no_context_confidence_threshold=0.8),
        context_config=ContextConfig(),
        clone_url="https://github.com/octo/widgets.git",
        token=None,
        workspace_factory=_failing_workspace,
    )
    # 0.9 >= 0.8 no-context threshold → still kept, tagged has_context=False.
    (comment,) = result.comments
    assert not comment.has_context


def test_build_run_result_maps_fields_and_failures() -> None:
    head_model = _ScriptedModel()
    case = _case()
    result = review_case(
        case,
        head_model,
        review_config=ReviewConfig(),
        context_config=ContextConfig(),
        clone_url="x",
        token=None,
        workspace_factory=_failing_workspace,
    )
    run_result = build_run_result(
        case, result, backend="gemini", model="gemini-2.5-flash", cost_usd=0.0012
    )
    assert run_result.repo == "octo/widgets"
    assert run_result.number == 7
    assert run_result.backend == "gemini"
    assert run_result.cost_usd == 0.0012
    assert run_result.failures == 0
    assert to_run_comments(result) == run_result.comments
