"""Tests for `pra review` end-to-end with stubbed GitHub, workspace, and model."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from pr_review_agent import cli
from pr_review_agent.github.models import PullRequest, Review, ReviewCommentDraft
from pr_review_agent.models.base import ModelMessage, ModelResponse

runner = CliRunner()

HEAD_SHA = "b" * 40

APP_SOURCE = 'from helpers import greet\n\n\ndef run() -> None:\n    greet("hi")\n'
HELPERS_SOURCE = 'def greet(name: str) -> str:\n    return f"hi {name}"\n'

_PR = PullRequest(
    number=7,
    title="Use greet helper",
    body="",
    state="open",
    author="octocat",
    base_ref="main",
    base_sha="a" * 40,
    head_ref="feature/greet",
    head_sha=HEAD_SHA,
    merged=False,
    draft=False,
    url="https://github.com/octo/widgets/pull/7",
    changed_files=1,
    additions=5,
    deletions=0,
)


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


class _StubGitHub:
    """Records created reviews at class level so re-invocations see them."""

    created: ClassVar[list[dict[str, object]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubGitHub:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get_pr(self, repo: str, number: int) -> PullRequest:
        return _PR

    def get_pr_diff(self, repo: str, number: int) -> str:
        return _diff()

    def get_authenticated_user(self) -> str:
        return "pra-bot"

    def list_reviews(self, repo: str, number: int) -> list[Review]:
        return [
            Review(
                review_id=i + 1,
                author="pra-bot",
                body=str(entry["body"]),
                state="COMMENTED",
                commit_id=HEAD_SHA,
            )
            for i, entry in enumerate(type(self).created)
        ]

    def create_review(
        self,
        repo: str,
        number: int,
        *,
        commit_id: str,
        body: str,
        comments: Sequence[ReviewCommentDraft],
    ) -> int:
        type(self).created.append(
            {"commit_id": commit_id, "body": body, "comments": list(comments)}
        )
        return len(type(self).created)


class _FakeModel:
    """Flags the single hunk, then emits one valid review comment."""

    @property
    def model(self) -> str:
        return "fake"

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        if purpose == "triage":
            text = json.dumps({"worth_reviewing": True, "category": "bug"})
        else:
            text = json.dumps(
                [
                    {
                        "line": 5,
                        "severity": "major",
                        "confidence": 0.9,
                        "comment": "greet() result is discarded.",
                    }
                ]
            )
        return ModelResponse(text=text, model="fake", input_tokens=10, output_tokens=5)


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stub GitHub, workspace, and model factory; run in a temp cwd."""
    head = tmp_path / "head"
    head.mkdir()
    (head / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (head / "helpers.py").write_text(HELPERS_SOURCE, encoding="utf-8")

    @contextmanager
    def fake_workspace(
        clone_url: str, pr_number: int, expected_head_sha: str, *, token: str | None = None
    ) -> Iterator[Path]:
        yield head

    _StubGitHub.created = []
    monkeypatch.setattr(cli, "GitHubClient", _StubGitHub)
    monkeypatch.setattr(cli, "pr_head_workspace", fake_workspace)
    monkeypatch.setattr(cli, "build_model_client", lambda backend, config: _FakeModel())
    monkeypatch.chdir(tmp_path)  # pra.sqlite3 lands here
    return tmp_path


def test_dry_run_is_default_and_posts_nothing(wired: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = runner.invoke(cli.app, ["review", "octo/widgets", "7"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "dry run — nothing posted" in result.output
    assert "greet() result is discarded." in result.output
    assert "app.py:5" in result.output
    assert _StubGitHub.created == []
    assert (wired / "pra.sqlite3").is_file()  # calls were cached and costed


def test_post_creates_single_review_with_marker(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    result = runner.invoke(
        cli.app, ["review", "octo/widgets", "7", "--post"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert len(_StubGitHub.created) == 1
    posted = _StubGitHub.created[0]
    assert f"<!-- pr-review-agent:octo/widgets#7@{HEAD_SHA} -->" in str(posted["body"])
    assert posted["commit_id"] == HEAD_SHA
    drafts = posted["comments"]
    assert isinstance(drafts, list)
    assert drafts[0].path == "app.py"
    assert drafts[0].line == 5


def test_rerun_on_same_head_is_a_noop(wired: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    first = runner.invoke(cli.app, ["review", "octo/widgets", "7", "--post"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(cli.app, ["review", "octo/widgets", "7", "--post"])
    assert second.exit_code == 0, second.output
    assert "already posted" in second.output
    assert len(_StubGitHub.created) == 1  # no double-post


def test_post_without_token_fails(wired: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = runner.invoke(cli.app, ["review", "octo/widgets", "7", "--post"])
    assert result.exit_code == 1
    assert _StubGitHub.created == []
