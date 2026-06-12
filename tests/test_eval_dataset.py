"""Tests for the eval dataset builder (stubbed GitHub source, no network)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pr_review_agent.evals.dataset import build_dataset
from pr_review_agent.evals.schema import CASES_FILE, load_cases
from pr_review_agent.github.client import GitHubAPIError
from pr_review_agent.github.models import PullRequest, ReviewComment

SINCE = "2026-01-01"


def _pr(number: int, *, merged_at: str | None, updated_at: str) -> PullRequest:
    return PullRequest(
        number=number,
        title=f"PR {number}",
        body="",
        state="closed",
        author="octocat",
        base_ref="main",
        base_sha="base" + "0" * 36,
        head_ref=f"feature/{number}",
        head_sha=f"head{number}" + "0" * 34,
        merged=merged_at is not None,
        draft=False,
        url="",
        changed_files=1,
        additions=1,
        deletions=0,
        merged_at=merged_at,
        updated_at=updated_at,
    )


def _comment(
    body: str,
    *,
    author: str = "alice",
    created_at: str = "2026-02-01T00:00:00Z",
    in_reply_to: int | None = None,
    original_commit_id: str = "prerev" + "0" * 34,
) -> ReviewComment:
    return ReviewComment(
        comment_id=hash((body, author, created_at)) % 10_000,
        path="app.py",
        body=body,
        author=author,
        line=10,
        original_line=10,
        start_line=None,
        side="RIGHT",
        commit_id="head" + "0" * 36,
        created_at=created_at,
        in_reply_to_id=in_reply_to,
        original_commit_id=original_commit_id,
    )


class StubSource:
    """Implements the DatasetSource protocol with canned data."""

    def __init__(
        self,
        prs: list[PullRequest],
        comments: dict[int, list[ReviewComment]],
        *,
        compare_fails: bool = False,
    ) -> None:
        self._prs = prs
        self._comments = comments
        self._compare_fails = compare_fails
        self.comment_requests: list[int] = []

    def list_pull_requests(self, repo: str, *, state: str = "closed") -> Iterator[PullRequest]:
        yield from self._prs

    def list_review_comments(self, repo: str, number: int) -> list[ReviewComment]:
        self.comment_requests.append(number)
        return self._comments.get(number, [])

    def get_pr_diff(self, repo: str, number: int) -> str:
        return f"diff --git a/final{number}.py b/final{number}.py\n"

    def compare_diff(self, repo: str, base: str, head: str) -> str:
        if self._compare_fails:
            raise GitHubAPIError("gone", status_code=404, url="x")
        return f"diff --git a/prerev.py b/prerev.py (head {head[:6]})\n"


def substantive_pair(original_sha: str = "prerev" + "0" * 34) -> list[ReviewComment]:
    return [
        _comment(
            "This retry loop never backs off between attempts.",
            created_at="2026-02-01T00:00:00Z",
            original_commit_id=original_sha,
        ),
        _comment(
            "The error branch swallows the original exception context.",
            created_at="2026-02-02T00:00:00Z",
            original_commit_id="later" + "0" * 35,
        ),
    ]


def test_selects_reconstructs_and_writes_cases(tmp_path: Path) -> None:
    noise = [
        _comment("Automated lint summary for this PR.", author="linty[bot]"),
        _comment("Agreed, will fix.", in_reply_to=1),
        _comment("ok"),  # too short to encode an issue
    ]
    source = StubSource(
        prs=[_pr(1, merged_at="2026-02-10T00:00:00Z", updated_at="2026-02-11T00:00:00Z")],
        comments={1: substantive_pair() + noise},
    )
    stats = build_dataset(source, "octo/widgets", since=SINCE, min_comments=2, out_dir=tmp_path)
    assert stats.selected == 1
    assert stats.reconstructed == 1
    (case,) = load_cases(tmp_path / CASES_FILE)
    assert case.repo == "octo/widgets"
    assert case.reconstructed
    # the earliest substantive comment's original commit wins
    assert case.review_sha == "prerev" + "0" * 34
    assert case.diff.startswith("diff --git a/prerev.py")
    assert len(case.human_comments) == 2  # bot, reply, and trivial filtered out
    assert case.human_comments[0].body.startswith("This retry loop")


def test_compare_failure_falls_back_to_final_diff(tmp_path: Path) -> None:
    source = StubSource(
        prs=[_pr(2, merged_at="2026-02-10T00:00:00Z", updated_at="2026-02-11T00:00:00Z")],
        comments={2: substantive_pair()},
        compare_fails=True,
    )
    stats = build_dataset(source, "octo/widgets", since=SINCE, min_comments=2, out_dir=tmp_path)
    assert stats.fallback_final_diff == 1
    (case,) = load_cases(tmp_path / CASES_FILE)
    assert not case.reconstructed
    assert case.review_sha == "head2" + "0" * 34  # falls back to the merged head
    assert case.diff.startswith("diff --git a/final2.py")


def test_filters_and_early_stop(tmp_path: Path) -> None:
    prs = [
        _pr(1, merged_at="2026-02-10T00:00:00Z", updated_at="2026-02-11T00:00:00Z"),
        _pr(3, merged_at="2026-02-09T00:00:00Z", updated_at="2026-02-10T00:00:00Z"),
        _pr(4, merged_at=None, updated_at="2026-02-09T00:00:00Z"),
        _pr(5, merged_at="2025-06-01T00:00:00Z", updated_at="2025-12-01T00:00:00Z"),
        _pr(6, merged_at="2025-05-01T00:00:00Z", updated_at="2025-11-01T00:00:00Z"),
    ]
    source = StubSource(
        prs=prs,
        comments={1: substantive_pair(), 3: substantive_pair()[:1]},
    )
    stats = build_dataset(source, "octo/widgets", since=SINCE, min_comments=2, out_dir=tmp_path)
    assert stats.selected == 1
    assert stats.skipped_few_comments == 1  # PR 3
    assert stats.skipped_unmerged == 1  # PR 4
    # PR 5 is older than --since: iteration stops, PR 6 is never even scanned
    assert stats.scanned == 3
    assert source.comment_requests == [1, 3]


def test_max_cases_stops_early(tmp_path: Path) -> None:
    prs = [
        _pr(1, merged_at="2026-02-10T00:00:00Z", updated_at="2026-02-11T00:00:00Z"),
        _pr(2, merged_at="2026-02-09T00:00:00Z", updated_at="2026-02-10T00:00:00Z"),
    ]
    source = StubSource(prs=prs, comments={1: substantive_pair(), 2: substantive_pair()})
    stats = build_dataset(
        source, "octo/widgets", since=SINCE, min_comments=2, out_dir=tmp_path, max_cases=1
    )
    assert stats.selected == 1
    assert source.comment_requests == [1]
