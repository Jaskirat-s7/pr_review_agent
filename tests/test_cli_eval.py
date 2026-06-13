"""Tests for `pra eval` CLI wiring (build-dataset and report)."""

from __future__ import annotations

import json
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


def test_run_cli_writes_run_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Sequence
    from contextlib import contextmanager

    from pr_review_agent import cli
    from pr_review_agent.evals.schema import EvalCase, load_runs, runs_path
    from pr_review_agent.models.base import ModelMessage, ModelResponse

    app_source = 'from helpers import greet\n\n\ndef run() -> None:\n    greet("hi")\n'
    lines = app_source.splitlines()
    diff = (
        "diff --git a/app.py b/app.py\nnew file mode 100644\n--- /dev/null\n+++ b/app.py\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join(f"+{x}" for x in lines) + "\n"
    )
    case = EvalCase(
        repo="octo/widgets",
        number=7,
        title="t",
        review_sha="a" * 40,
        reconstructed=True,
        diff=diff,
        human_comments=(),
    )
    write_jsonl(tmp_path / CASES_FILE, [case])

    head = tmp_path / "head"
    head.mkdir()
    (head / "app.py").write_text(app_source, encoding="utf-8")
    (head / "helpers.py").write_text("def greet(n):\n    return n\n", encoding="utf-8")

    class _Model:
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
            text = (
                json.dumps({"worth_reviewing": True, "category": "bug"})
                if purpose == "triage"
                else json.dumps(
                    [{"line": 5, "severity": "major", "confidence": 0.9, "comment": "discarded"}]
                )
            )
            return ModelResponse(
                text=text, model="gemini-2.5-flash", input_tokens=10, output_tokens=4
            )

    @contextmanager
    def fake_ws(clone_url: str, sha: str, *, token: str | None = None):  # type: ignore[no-untyped-def]
        yield head

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "build_model_client", lambda backend, config: _Model())
    monkeypatch.setattr("pr_review_agent.evals.run.commit_workspace", fake_ws)
    monkeypatch.chdir(tmp_path)  # pra.sqlite3 lands here

    result = runner.invoke(
        cli.app, ["eval", "run", str(tmp_path), "--backend", "gemini"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    runs = load_runs(runs_path(tmp_path, "gemini"))
    assert len(runs) == 1
    (run,) = runs
    assert run.backend == "gemini"
    assert run.comments[0].line == 5
    assert run.comments[0].has_context


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
