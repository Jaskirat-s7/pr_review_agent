"""JSONL schemas shared across the eval pipeline.

A dataset directory looks like:

    dataset/
      cases.jsonl                 # EvalCase per line (build-dataset)
      runs/<backend>.jsonl        # RunResult per line (eval run)
      judgments/<backend>.jsonl   # CaseJudgment per line (eval judge)
      judgments/<backend>_sample.csv  # 20% manual-validation sample
      report.md                   # eval report
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CASES_FILE = "cases.jsonl"


class EvalDataError(Exception):
    """Raised when an eval JSONL file is missing or malformed."""


def runs_path(dataset_dir: Path, backend: str) -> Path:
    return dataset_dir / "runs" / f"{backend}.jsonl"


def judgments_path(dataset_dir: Path, backend: str) -> Path:
    return dataset_dir / "judgments" / f"{backend}.jsonl"


def sample_path(dataset_dir: Path, backend: str) -> Path:
    return dataset_dir / "judgments" / f"{backend}_sample.csv"


@dataclass(frozen=True, slots=True)
class HumanComment:
    """One substantive human review comment (the reference standard)."""

    author: str
    path: str
    line: int | None
    body: str
    created_at: str


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One historical PR: the diff to review plus the human comments."""

    repo: str
    number: int
    title: str
    review_sha: str  # the commit the agent should review
    reconstructed: bool  # True = faithful pre-review state (decision #1)
    diff: str
    human_comments: tuple[HumanComment, ...]


@dataclass(frozen=True, slots=True)
class RunComment:
    """An agent comment as stored in run results."""

    file_path: str
    line: int
    severity: str
    confidence: float
    body: str
    category: str
    has_context: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    """The agent's output for one case."""

    repo: str
    number: int
    review_sha: str
    backend: str
    model: str
    comments: tuple[RunComment, ...]
    cost_usd: float
    failures: int  # triage+review failures; nonzero flags corrupted recall


@dataclass(frozen=True, slots=True)
class HumanJudgment:
    """Did any agent comment identify this human comment's issue?"""

    human_index: int
    verdict: str  # match | partial | miss | error
    matched_agent_index: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class ExtraJudgment:
    """Classification of an agent comment with no human counterpart."""

    agent_index: int
    verdict: str  # plausible-extra | false-positive | error
    reason: str


@dataclass(frozen=True, slots=True)
class CaseJudgment:
    repo: str
    number: int
    backend: str
    human_judgments: tuple[HumanJudgment, ...]
    extra_judgments: tuple[ExtraJudgment, ...]


def write_jsonl(path: Path, items: Iterable[object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")  # type: ignore[call-overload]
            count += 1
    return count


def load_cases(path: Path) -> list[EvalCase]:
    return [
        EvalCase(
            repo=_req_str(obj, "repo", path),
            number=_req_int(obj, "number", path),
            title=str(obj.get("title", "")),
            review_sha=_req_str(obj, "review_sha", path),
            reconstructed=bool(obj.get("reconstructed", False)),
            diff=_req_str(obj, "diff", path),
            human_comments=tuple(
                HumanComment(
                    author=str(c.get("author", "")),
                    path=str(c.get("path", "")),
                    line=c.get("line") if isinstance(c.get("line"), int) else None,
                    body=str(c.get("body", "")),
                    created_at=str(c.get("created_at", "")),
                )
                for c in obj.get("human_comments", [])
                if isinstance(c, dict)
            ),
        )
        for obj in _read_jsonl(path)
    ]


def load_runs(path: Path) -> list[RunResult]:
    return [
        RunResult(
            repo=_req_str(obj, "repo", path),
            number=_req_int(obj, "number", path),
            review_sha=str(obj.get("review_sha", "")),
            backend=str(obj.get("backend", "")),
            model=str(obj.get("model", "")),
            comments=tuple(
                RunComment(
                    file_path=str(c.get("file_path", "")),
                    line=int(c.get("line", 0)),
                    severity=str(c.get("severity", "")),
                    confidence=float(c.get("confidence", 0.0)),
                    body=str(c.get("body", "")),
                    category=str(c.get("category", "")),
                    has_context=bool(c.get("has_context", False)),
                )
                for c in obj.get("comments", [])
                if isinstance(c, dict)
            ),
            cost_usd=float(obj.get("cost_usd", 0.0)),
            failures=int(obj.get("failures", 0)),
        )
        for obj in _read_jsonl(path)
    ]


def load_judgments(path: Path) -> list[CaseJudgment]:
    return [
        CaseJudgment(
            repo=_req_str(obj, "repo", path),
            number=_req_int(obj, "number", path),
            backend=str(obj.get("backend", "")),
            human_judgments=tuple(
                HumanJudgment(
                    human_index=int(j.get("human_index", -1)),
                    verdict=str(j.get("verdict", "error")),
                    matched_agent_index=(
                        j.get("matched_agent_index")
                        if isinstance(j.get("matched_agent_index"), int)
                        else None
                    ),
                    reason=str(j.get("reason", "")),
                )
                for j in obj.get("human_judgments", [])
                if isinstance(j, dict)
            ),
            extra_judgments=tuple(
                ExtraJudgment(
                    agent_index=int(j.get("agent_index", -1)),
                    verdict=str(j.get("verdict", "error")),
                    reason=str(j.get("reason", "")),
                )
                for j in obj.get("extra_judgments", [])
                if isinstance(j, dict)
            ),
        )
        for obj in _read_jsonl(path)
    ]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise EvalDataError(f"missing eval file: {path}")
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalDataError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvalDataError(f"{path}:{line_no}: expected a JSON object")
            yield obj


def _req_str(obj: dict[str, Any], key: str, path: Path) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise EvalDataError(f"{path}: missing or non-string {key!r}")
    return value


def _req_int(obj: dict[str, Any], key: str, path: Path) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvalDataError(f"{path}: missing or non-integer {key!r}")
    return value
