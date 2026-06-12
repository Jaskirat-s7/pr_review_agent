"""Build an eval dataset from merged PRs with substantive human reviews.

Pre-review reconstruction (approved decision #1): the agent should review
the code the *first human reviewer* saw, not the merged result — otherwise
the diff already incorporates fixes those very comments prompted, and
recall is contaminated. The earliest substantive review comment's
``original_commit_id`` identifies that head; the diff is reconstructed via
the compare API. When that fails (commit garbage-collected after a force
push), the case falls back to the final PR diff with
``reconstructed: false`` so the report can exclude or disclose it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pr_review_agent.evals.schema import CASES_FILE, EvalCase, HumanComment, write_jsonl
from pr_review_agent.github.client import GitHubError
from pr_review_agent.github.models import PullRequest, ReviewComment

logger = logging.getLogger(__name__)

_MIN_BODY_CHARS = 10


class DatasetSource(Protocol):
    """The slice of GitHubClient the builder needs (stubable in tests)."""

    def list_pull_requests(self, repo: str, *, state: str = "closed") -> Iterator[PullRequest]: ...

    def list_review_comments(self, repo: str, number: int) -> list[ReviewComment]: ...

    def get_pr_diff(self, repo: str, number: int) -> str: ...

    def compare_diff(self, repo: str, base: str, head: str) -> str: ...


@dataclass(frozen=True, slots=True)
class BuildStats:
    scanned: int
    selected: int
    reconstructed: int
    fallback_final_diff: int
    skipped_few_comments: int
    skipped_unmerged: int


def build_dataset(
    source: DatasetSource,
    repo: str,
    *,
    since: str,
    min_comments: int,
    out_dir: Path,
    max_cases: int | None = None,
) -> BuildStats:
    """Scan recently updated PRs and write ``cases.jsonl`` under ``out_dir``.

    ``since`` is an ISO date (YYYY-MM-DD); ISO-8601 timestamps compare
    correctly as strings against it.
    """
    cases: list[EvalCase] = []
    scanned = reconstructed = fallback = few_comments = unmerged = 0

    for pr in source.list_pull_requests(repo, state="closed"):
        if pr.updated_at and pr.updated_at < since:
            break  # sorted by updated desc: everything after this is older
        scanned += 1
        if not pr.merged_at or pr.merged_at < since:
            unmerged += 1
            continue
        substantive = [c for c in source.list_review_comments(repo, pr.number) if _substantive(c)]
        if len(substantive) < min_comments:
            few_comments += 1
            continue

        case = _build_case(source, repo, pr, substantive)
        cases.append(case)
        if case.reconstructed:
            reconstructed += 1
        else:
            fallback += 1
        if max_cases is not None and len(cases) >= max_cases:
            break

    write_jsonl(out_dir / CASES_FILE, cases)
    return BuildStats(
        scanned=scanned,
        selected=len(cases),
        reconstructed=reconstructed,
        fallback_final_diff=fallback,
        skipped_few_comments=few_comments,
        skipped_unmerged=unmerged,
    )


def _build_case(
    source: DatasetSource,
    repo: str,
    pr: PullRequest,
    substantive: list[ReviewComment],
) -> EvalCase:
    earliest = min(substantive, key=lambda c: c.created_at)
    pre_review_sha = earliest.original_commit_id or earliest.commit_id
    diff: str | None = None
    review_sha = pre_review_sha
    is_reconstructed = False
    if pre_review_sha and pr.base_sha:
        try:
            diff = source.compare_diff(repo, pr.base_sha, pre_review_sha)
            is_reconstructed = True
        except GitHubError as exc:
            logger.warning(
                "%s#%d: pre-review state %s unavailable (%s); falling back to final diff",
                repo,
                pr.number,
                pre_review_sha[:12],
                exc,
            )
    if diff is None:
        diff = source.get_pr_diff(repo, pr.number)
        review_sha = pr.head_sha
    return EvalCase(
        repo=repo,
        number=pr.number,
        title=pr.title,
        review_sha=review_sha,
        reconstructed=is_reconstructed,
        diff=diff,
        human_comments=tuple(
            HumanComment(
                author=c.author,
                path=c.path,
                line=c.line if c.line is not None else c.original_line,
                body=c.body,
                created_at=c.created_at,
            )
            for c in sorted(substantive, key=lambda c: c.created_at)
        ),
    )


def _substantive(comment: ReviewComment) -> bool:
    """Top-level, human-authored, with enough body to encode an issue."""
    if comment.in_reply_to_id is not None:
        return False  # replies are discussion, not independent findings
    if comment.author.endswith("[bot]") or not comment.author:
        return False
    return len(comment.body.strip()) >= _MIN_BODY_CHARS
