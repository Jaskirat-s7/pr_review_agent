"""LLM-as-judge: grade agent comments against the human review of record.

For each human comment: did any agent comment identify the same issue
(match / partial / miss)? Each agent comment matched by no human comment is
classified plausible-extra or false-positive. Unparseable judge output is
recorded as verdict "error" — never silently coerced — so the report can
exclude it instead of scoring it as a miss.
"""

from __future__ import annotations

import csv
import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from pr_review_agent.evals.schema import (
    CaseJudgment,
    EvalCase,
    ExtraJudgment,
    HumanComment,
    HumanJudgment,
    RunComment,
    RunResult,
)
from pr_review_agent.jsonutil import parse_model_json
from pr_review_agent.models.base import ModelClient, ModelError, ModelMessage
from pr_review_agent.prompts import load_prompt

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_JUDGE_MAX_TOKENS = 256
_MATCH_VERDICTS = frozenset({"match", "partial", "miss"})
_EXTRA_VERDICTS = frozenset({"plausible-extra", "false-positive"})


class EvalJudge:
    """Judges one backend's run results against the dataset."""

    def __init__(
        self,
        client: ModelClient,
        *,
        delay_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._delay_seconds = delay_seconds
        self._sleep = sleep
        self._system = load_prompt("judge_system").template
        self._match_user = load_prompt("judge_match_user")
        self._extra_user = load_prompt("judge_extra_user")

    def judge_all(self, cases: Sequence[EvalCase], runs: Sequence[RunResult]) -> list[CaseJudgment]:
        runs_by_key = {(r.repo, r.number): r for r in runs}
        judgments: list[CaseJudgment] = []
        for case in cases:
            run = runs_by_key.get((case.repo, case.number))
            if run is None:
                logger.warning("no run result for %s#%d; skipping", case.repo, case.number)
                continue
            # Space out cases so a 50-PR batch doesn't burst against the
            # Claude Code 5-hour usage window. A run is resumable: judgments
            # are written per backend, so a partial batch can be re-run.
            if judgments and self._delay_seconds > 0:
                self._sleep(self._delay_seconds)
            judgments.append(self.judge_case(case, run))
        return judgments

    def judge_case(self, case: EvalCase, run: RunResult) -> CaseJudgment:
        matched: set[int] = set()
        human_judgments: list[HumanJudgment] = []
        for index, human in enumerate(case.human_comments):
            if not run.comments:
                human_judgments.append(
                    HumanJudgment(index, "miss", None, "agent produced no comments")
                )
                continue
            human_judgments.append(self._judge_match(case, run, index, human))
        for judgment in human_judgments:
            if (
                judgment.verdict in ("match", "partial")
                and judgment.matched_agent_index is not None
            ):
                matched.add(judgment.matched_agent_index)

        extra_judgments = [
            self._judge_extra(case, agent_index, comment)
            for agent_index, comment in enumerate(run.comments)
            if agent_index not in matched
        ]
        return CaseJudgment(
            repo=case.repo,
            number=case.number,
            backend=run.backend,
            human_judgments=tuple(human_judgments),
            extra_judgments=tuple(extra_judgments),
        )

    def _judge_match(
        self, case: EvalCase, run: RunResult, index: int, human: HumanComment
    ) -> HumanJudgment:
        user = self._match_user.substitute(
            repo=case.repo,
            number=str(case.number),
            title=case.title,
            path=human.path,
            line="?" if human.line is None else str(human.line),
            body=human.body,
            agent_comments=_render_agent_comments(run.comments),
        )
        payload = self._ask(user, f"{case.repo}#{case.number} human[{index}]")
        if not isinstance(payload, dict) or payload.get("verdict") not in _MATCH_VERDICTS:
            return HumanJudgment(index, "error", None, "unparseable judge output")
        verdict = str(payload["verdict"])
        raw_index = payload.get("matched_agent_index")
        matched_index = raw_index if isinstance(raw_index, int) else None
        if matched_index is not None and not 0 <= matched_index < len(run.comments):
            matched_index = None
        if verdict == "miss":
            matched_index = None
        return HumanJudgment(index, verdict, matched_index, str(payload.get("reason", "")))

    def _judge_extra(self, case: EvalCase, agent_index: int, comment: RunComment) -> ExtraJudgment:
        user = self._extra_user.substitute(
            repo=case.repo,
            number=str(case.number),
            title=case.title,
            human_comments=_render_human_comments(case.human_comments),
            file_path=comment.file_path,
            line=str(comment.line),
            body=comment.body,
        )
        payload = self._ask(user, f"{case.repo}#{case.number} agent[{agent_index}]")
        if not isinstance(payload, dict) or payload.get("verdict") not in _EXTRA_VERDICTS:
            return ExtraJudgment(agent_index, "error", "unparseable judge output")
        return ExtraJudgment(agent_index, str(payload["verdict"]), str(payload.get("reason", "")))

    def _ask(self, user: str, label: str) -> object:
        try:
            response = self._client.complete(
                self._system,
                [ModelMessage("user", user)],
                max_tokens=_JUDGE_MAX_TOKENS,
                purpose="judge",
            )
        except ModelError as exc:
            logger.warning("judge call failed for %s: %s", label, exc)
            return None
        return parse_model_json(response.text)


def export_sample(
    judgments: Sequence[CaseJudgment],
    cases: Sequence[EvalCase],
    runs: Sequence[RunResult],
    path: Path,
    *,
    fraction: float = 0.2,
    seed: int = 42,
) -> int:
    """Export a seeded random sample of judgments to CSV for manual checks."""
    cases_by_key = {(c.repo, c.number): c for c in cases}
    runs_by_key = {(r.repo, r.number): r for r in runs}
    rows: list[list[str]] = []
    for judgment in judgments:
        key = (judgment.repo, judgment.number)
        case = cases_by_key.get(key)
        run = runs_by_key.get(key)
        for hj in judgment.human_judgments:
            human = _safe(case.human_comments, hj.human_index) if case else None
            agent = (
                _safe(run.comments, hj.matched_agent_index)
                if run and hj.matched_agent_index is not None
                else None
            )
            rows.append(
                [
                    "human",
                    judgment.repo,
                    str(judgment.number),
                    str(hj.human_index),
                    hj.verdict,
                    hj.reason,
                    _excerpt(human.body) if human else "",
                    _excerpt(agent.body) if agent else "",
                ]
            )
        for ej in judgment.extra_judgments:
            agent = _safe(run.comments, ej.agent_index) if run else None
            rows.append(
                [
                    "extra",
                    judgment.repo,
                    str(judgment.number),
                    str(ej.agent_index),
                    ej.verdict,
                    ej.reason,
                    "",
                    _excerpt(agent.body) if agent else "",
                ]
            )
    if not rows:
        return 0
    sample_size = max(1, round(len(rows) * fraction))
    sampled = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "kind",
                "repo",
                "number",
                "index",
                "verdict",
                "reason",
                "human_excerpt",
                "agent_excerpt",
            ]
        )
        writer.writerows(sampled)
    return len(sampled)


def _render_agent_comments(comments: Sequence[RunComment]) -> str:
    return "\n".join(
        f"[{i}] {c.file_path}:{c.line} (severity {c.severity}, "
        f"confidence {c.confidence:.2f}): {c.body}"
        for i, c in enumerate(comments)
    )


def _render_human_comments(comments: Sequence[HumanComment]) -> str:
    if not comments:
        return "(none)"
    return "\n".join(f"- [{c.path}:{'?' if c.line is None else c.line}] {c.body}" for c in comments)


def _excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _safe(items: Sequence[_T], index: int) -> _T | None:
    return items[index] if 0 <= index < len(items) else None
