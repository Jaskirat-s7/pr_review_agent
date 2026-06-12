"""Tests for `pra eval` CLI wiring (build-dataset and report)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pr_review_agent import cli
from pr_review_agent.evals.schema import (
    CASES_FILE,
    CaseJudgment,
    EvalCase,
    HumanComment,
    HumanJudgment,
    RunComment,
    RunResult,
    judgments_path,
    load_cases,
    runs_path,
    write_jsonl,
)
from pr_review_agent.github.models import PullRequest, ReviewComment

runner = CliRunner()


class _StubGitHub:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubGitHub:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def list_pull_requests(self, repo: str, *, state: str = "closed") -> Iterator[PullRequest]:
        yield PullRequest(
            number=7,
            title="Add retry",
            body="",
            state="closed",
            author="octocat",
            base_ref="main",
            base_sha="b" * 40,
            head_ref="f",
            head_sha="h" * 40,
            merged=True,
            draft=False,
            url="",
            changed_files=1,
            additions=1,
            deletions=0,
            merged_at="2026-02-10T00:00:00Z",
            updated_at="2026-02-11T00:00:00Z",
        )

    def list_review_comments(self, repo: str, number: int) -> list[ReviewComment]:
        def comment(body: str, created: str) -> ReviewComment:
            return ReviewComment(
                comment_id=1,
                path="app.py",
                body=body,
                author="alice",
                line=10,
                original_line=10,
                start_line=None,
                side="RIGHT",
                commit_id="h" * 40,
                created_at=created,
                in_reply_to_id=None,
                original_commit_id="o" * 40,
            )

        return [
            comment("This retry loop has no backoff at all.", "2026-02-01T00:00:00Z"),
            comment("Exception context is swallowed here.", "2026-02-02T00:00:00Z"),
        ]

    def get_pr_diff(self, repo: str, number: int) -> str:
        return "diff --git a/final.py b/final.py\n"

    def compare_diff(self, repo: str, base: str, head: str) -> str:
        return "diff --git a/prerev.py b/prerev.py\n"


def test_build_dataset_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "GitHubClient", _StubGitHub)
    out = tmp_path / "dataset"
    result = runner.invoke(
        cli.app,
        ["eval", "build-dataset", "octo/widgets", "--since", "2026-01-01", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "Selected 1 case(s)" in result.output
    (case,) = load_cases(out / CASES_FILE)
    assert case.reconstructed
    assert case.review_sha == "o" * 40


def test_report_cli_writes_markdown(tmp_path: Path) -> None:
    case = EvalCase(
        repo="octo/widgets",
        number=7,
        title="t",
        review_sha="s",
        reconstructed=True,
        diff="d",
        human_comments=(HumanComment("alice", "app.py", 10, "issue", "t"),),
    )
    run = RunResult(
        repo="octo/widgets",
        number=7,
        review_sha="s",
        backend="gemini",
        model="m",
        comments=(RunComment("app.py", 10, "major", 0.9, "found", "bug", True),),
        cost_usd=0.001,
        failures=0,
    )
    judgment = CaseJudgment(
        repo="octo/widgets",
        number=7,
        backend="gemini",
        human_judgments=(HumanJudgment(0, "match", 0, "same"),),
        extra_judgments=(),
    )
    write_jsonl(tmp_path / CASES_FILE, [case])
    write_jsonl(runs_path(tmp_path, "gemini"), [run])
    write_jsonl(judgments_path(tmp_path, "gemini"), [judgment])

    result = runner.invoke(cli.app, ["eval", "report", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "| recall vs human comments | 1.00 |" in result.output
    assert (tmp_path / "report.md").is_file()
