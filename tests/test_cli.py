"""Tests for the `pra` CLI (GitHub client stubbed; no live calls)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from conftest import load_fixture
from pr_review_agent import cli
from pr_review_agent.github.client import GitHubAPIError
from pr_review_agent.github.models import PullRequest

runner = CliRunner()

_PR = PullRequest(
    number=123,
    title="Add retry logic to fetcher",
    body="Fixes #100",
    state="open",
    author="octocat",
    base_ref="main",
    base_sha="aaa111aaa111",
    head_ref="feature/retry",
    head_sha="bbb222bbb222",
    merged=False,
    draft=False,
    url="https://github.com/octo/widgets/pull/123",
    changed_files=2,
    additions=10,
    deletions=3,
)


class _StubClient:
    """Stands in for GitHubClient; returns canned data."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get_pr(self, repo: str, number: int) -> PullRequest:
        return _PR

    def get_pr_diff(self, repo: str, number: int) -> str:
        return load_fixture("diffs", "multi.diff")


class _FailingClient(_StubClient):
    def get_pr(self, repo: str, number: int) -> PullRequest:
        raise GitHubAPIError("Not Found", status_code=404, url="https://api.github.com/x")


def test_fetch_prints_diff_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "GitHubClient", _StubClient)
    result = runner.invoke(cli.app, ["fetch", "octo/widgets", "123"])
    assert result.exit_code == 0, result.output
    assert "PR #123" in result.output
    assert "src/app.py" in result.output
    assert "docs/notes.md" in result.output
    assert "old_module.py" in result.output
    assert "5 file(s) changed" in result.output


def test_fetch_reports_api_errors_and_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "GitHubClient", _FailingClient)
    result = runner.invoke(cli.app, ["fetch", "octo/widgets", "999"])
    assert result.exit_code == 1


def test_fetch_rejects_non_positive_pr_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "GitHubClient", _StubClient)
    result = runner.invoke(cli.app, ["fetch", "octo/widgets", "0"])
    assert result.exit_code != 0
