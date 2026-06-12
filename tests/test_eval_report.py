"""Tests for the markdown eval report."""

from __future__ import annotations

from pathlib import Path

from pr_review_agent.evals.report import generate_report
from pr_review_agent.evals.schema import (
    CASES_FILE,
    CaseJudgment,
    EvalCase,
    ExtraJudgment,
    HumanComment,
    HumanJudgment,
    RunComment,
    RunResult,
    judgments_path,
    runs_path,
    write_jsonl,
)


def _case(number: int, *, reconstructed: bool) -> EvalCase:
    return EvalCase(
        repo="octo/widgets",
        number=number,
        title=f"PR {number}",
        review_sha=f"sha{number}",
        reconstructed=reconstructed,
        diff="diff --git a/x b/x\n",
        human_comments=(
            HumanComment("alice", "app.py", 10, "issue one", "t1"),
            HumanComment("bob", "app.py", 20, "issue two", "t2"),
        ),
    )


def _run(number: int, backend: str, *, cost: float) -> RunResult:
    return RunResult(
        repo="octo/widgets",
        number=number,
        review_sha=f"sha{number}",
        backend=backend,
        model=f"{backend}-model",
        comments=(
            RunComment("app.py", 10, "major", 0.9, "found one", "bug", True),
            RunComment("app.py", 30, "minor", 0.7, "extra thing", "bug", False),
        ),
        cost_usd=cost,
        failures=0,
    )


def _judgment(number: int, backend: str, *, first_verdict: str) -> CaseJudgment:
    return CaseJudgment(
        repo="octo/widgets",
        number=number,
        backend=backend,
        human_judgments=(
            HumanJudgment(0, first_verdict, 0 if first_verdict != "miss" else None, "r"),
            HumanJudgment(1, "miss", None, "r"),
        ),
        extra_judgments=(ExtraJudgment(1, "false-positive", "noise"),),
    )


def _build_dataset_dir(tmp_path: Path) -> Path:
    cases = [_case(1, reconstructed=True), _case(2, reconstructed=False)]
    write_jsonl(tmp_path / CASES_FILE, cases)
    for backend, verdicts in (("gemini", ("match", "miss")), ("anthropic", ("match", "match"))):
        write_jsonl(
            runs_path(tmp_path, backend),
            [_run(1, backend, cost=0.002), _run(2, backend, cost=0.004)],
        )
        write_jsonl(
            judgments_path(tmp_path, backend),
            [
                _judgment(1, backend, first_verdict=verdicts[0]),
                _judgment(2, backend, first_verdict=verdicts[1]),
            ],
        )
    return tmp_path


def test_headline_metrics_and_ceiling_column(tmp_path: Path) -> None:
    report = generate_report(_build_dataset_dir(tmp_path))
    assert "anthropic (ceiling)" in report
    # Columns sort alphabetically: anthropic first, then gemini.
    # anthropic: match=2, miss=2 of 4 human comments → recall 0.50
    # gemini: match=1, miss=3 → recall 0.25
    assert "| recall vs human comments | 0.50 | 0.25 |" in report
    assert "| false positives per PR | 1.00 | 1.00 |" in report
    assert "| cost per PR (USD) | $0.0030 | $0.0030 |" in report
    assert "match + 0.5·partial" in report  # scoring legend is stated


def test_contamination_split_present(tmp_path: Path) -> None:
    report = generate_report(_build_dataset_dir(tmp_path))
    assert "## Recall by dataset contamination" in report
    # anthropic matched on both cases (0.50 each); gemini only on the
    # reconstructed case — its fallback recall collapses to 0.00.
    assert "| reconstructed (clean) | 0.50 | 0.50 |" in report
    assert "| final-diff fallback | 0.50 | 0.00 |" in report


def test_context_split_uses_agent_comment_flags(tmp_path: Path) -> None:
    report = generate_report(_build_dataset_dir(tmp_path))
    assert "## Agent-comment outcomes by retrieved context" in report
    # matched agent comment (index 0) has context; the false positive (index 1) doesn't
    assert "| matched a human comment | 1 | 0 |" in report
    assert "| false positive | 0 | 2 |" in report


def test_report_without_judgments_says_so(tmp_path: Path) -> None:
    write_jsonl(tmp_path / CASES_FILE, [_case(1, reconstructed=True)])
    report = generate_report(tmp_path)
    assert "No judgments found" in report
