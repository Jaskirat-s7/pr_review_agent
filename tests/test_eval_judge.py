"""Tests for the LLM-as-judge and the manual-validation CSV sample."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from pr_review_agent.evals.judge import EvalJudge, export_sample
from pr_review_agent.evals.schema import (
    CaseJudgment,
    EvalCase,
    ExtraJudgment,
    HumanComment,
    HumanJudgment,
    RunComment,
    RunResult,
)
from pr_review_agent.models.base import ModelMessage, ModelResponse


class ScriptedJudge:
    """Returns queued responses; records the prompts it saw."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "scripted-judge"

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        assert purpose == "judge"
        self.prompts.append(messages[0].content)
        return ModelResponse(
            text=self._responses.pop(0), model="scripted-judge", input_tokens=5, output_tokens=5
        )


def _case(n_human: int = 2) -> EvalCase:
    return EvalCase(
        repo="octo/widgets",
        number=7,
        title="Add retry",
        review_sha="abc",
        reconstructed=True,
        diff="diff --git a/app.py b/app.py\n",
        human_comments=tuple(
            HumanComment("alice", "app.py", 10 + i, f"human issue {i}", f"2026-02-0{i + 1}")
            for i in range(n_human)
        ),
    )


def _run(n_agent: int = 2) -> RunResult:
    return RunResult(
        repo="octo/widgets",
        number=7,
        review_sha="abc",
        backend="gemini",
        model="gemini-2.5-flash",
        comments=tuple(
            RunComment(
                file_path="app.py",
                line=10 + i,
                severity="major",
                confidence=0.9,
                body=f"agent finding {i}",
                category="bug",
                has_context=i == 0,
            )
            for i in range(n_agent)
        ),
        cost_usd=0.001,
        failures=0,
    )


def test_match_miss_and_extra_classification() -> None:
    judge_client = ScriptedJudge(
        [
            json.dumps({"verdict": "match", "matched_agent_index": 0, "reason": "same issue"}),
            json.dumps({"verdict": "miss", "matched_agent_index": None, "reason": "absent"}),
            json.dumps({"verdict": "plausible-extra", "reason": "valid catch"}),
        ]
    )
    (judgment,) = EvalJudge(judge_client).judge_all([_case()], [_run()])
    assert judgment.human_judgments[0].verdict == "match"
    assert judgment.human_judgments[0].matched_agent_index == 0
    assert judgment.human_judgments[1].verdict == "miss"
    # only the unmatched agent comment (index 1) is classified as extra
    (extra,) = judgment.extra_judgments
    assert extra.agent_index == 1
    assert extra.verdict == "plausible-extra"
    assert len(judge_client.prompts) == 3


def test_partial_match_excludes_agent_comment_from_extras() -> None:
    judge_client = ScriptedJudge(
        [
            json.dumps({"verdict": "partial", "matched_agent_index": 1, "reason": "vague"}),
            json.dumps({"verdict": "miss", "matched_agent_index": None, "reason": ""}),
            json.dumps({"verdict": "false-positive", "reason": "confused"}),
        ]
    )
    (judgment,) = EvalJudge(judge_client).judge_all([_case()], [_run()])
    (extra,) = judgment.extra_judgments
    assert extra.agent_index == 0  # index 1 was partially matched
    assert extra.verdict == "false-positive"


def test_no_agent_comments_means_miss_without_model_calls() -> None:
    judge_client = ScriptedJudge([])  # any call would pop from an empty list
    (judgment,) = EvalJudge(judge_client).judge_all([_case()], [_run(n_agent=0)])
    assert [j.verdict for j in judgment.human_judgments] == ["miss", "miss"]
    assert all(j.reason == "agent produced no comments" for j in judgment.human_judgments)
    assert judgment.extra_judgments == ()
    assert judge_client.prompts == []


def test_unparseable_judge_output_is_error_not_miss() -> None:
    judge_client = ScriptedJudge(
        [
            "I think it matches, sort of?",  # not JSON
            json.dumps({"verdict": "match", "matched_agent_index": 99, "reason": "bad index"}),
            json.dumps({"verdict": "plausible-extra", "reason": ""}),
            json.dumps({"verdict": "plausible-extra", "reason": ""}),
        ]
    )
    (judgment,) = EvalJudge(judge_client).judge_all([_case()], [_run()])
    assert judgment.human_judgments[0].verdict == "error"
    # out-of-range matched index is discarded but the verdict survives
    assert judgment.human_judgments[1].verdict == "match"
    assert judgment.human_judgments[1].matched_agent_index is None


def test_case_without_run_result_is_skipped() -> None:
    judge_client = ScriptedJudge([])
    judgments = EvalJudge(judge_client).judge_all([_case()], [])
    assert judgments == []


def test_export_sample_is_seeded_and_sized(tmp_path: Path) -> None:
    judgments = [
        CaseJudgment(
            repo="octo/widgets",
            number=7,
            backend="gemini",
            human_judgments=tuple(HumanJudgment(i, "miss", None, f"reason {i}") for i in range(8)),
            extra_judgments=(
                ExtraJudgment(0, "false-positive", "noise"),
                ExtraJudgment(1, "plausible-extra", "fine"),
            ),
        )
    ]
    case = _case(n_human=8)
    run = _run()
    out = tmp_path / "sample.csv"
    written = export_sample(judgments, [case], [run], out, fraction=0.2, seed=42)
    assert written == 2  # 10 rows * 0.2
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:5] == ["kind", "repo", "number", "index", "verdict"]
    assert len(rows) == 3  # header + 2 sampled

    # deterministic for the same seed
    out2 = tmp_path / "sample2.csv"
    export_sample(judgments, [case], [run], out2, fraction=0.2, seed=42)
    assert (
        out.read_text(encoding="utf-8")[out.read_text(encoding="utf-8").find("\n") :]
        == (out2.read_text(encoding="utf-8")[out2.read_text(encoding="utf-8").find("\n") :])
    )
