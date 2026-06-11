"""Tests for `pra context` (GitHub client and workspace checkout stubbed)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pr_review_agent import cli
from pr_review_agent.github.models import PullRequest

runner = CliRunner()

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
    head_sha="b" * 40,
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


class _StubClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get_pr(self, repo: str, number: int) -> PullRequest:
        return _PR

    def get_pr_diff(self, repo: str, number: int) -> str:
        return _diff()


def test_context_prints_retrieved_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = tmp_path / "head"
    head.mkdir()
    (head / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (head / "helpers.py").write_text(HELPERS_SOURCE, encoding="utf-8")

    recorded: dict[str, object] = {}

    @contextmanager
    def fake_workspace(
        clone_url: str, pr_number: int, expected_head_sha: str, *, token: str | None = None
    ) -> Iterator[Path]:
        recorded["clone_url"] = clone_url
        recorded["pr_number"] = pr_number
        recorded["sha"] = expected_head_sha
        yield head

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "GitHubClient", _StubClient)
    monkeypatch.setattr(cli, "pr_head_workspace", fake_workspace)
    result = runner.invoke(cli.app, ["context", "octo/widgets", "7"])
    assert result.exit_code == 0, result.output
    assert "greet" in result.output
    assert "helpers.py" in result.output
    assert recorded["clone_url"] == "https://github.com/octo/widgets.git"
    assert recorded["pr_number"] == 7
    assert recorded["sha"] == "b" * 40
