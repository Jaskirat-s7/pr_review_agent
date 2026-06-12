"""Typed views over GitHub REST API payloads.

Parsing is lenient about optional fields (GitHub nulls many of them) but
strict about identity fields like PR number and comment id.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PullRequest:
    """Metadata for a pull request."""

    number: int
    title: str
    body: str
    state: str
    author: str
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    merged: bool
    draft: bool
    url: str
    changed_files: int
    additions: int
    deletions: int
    merged_at: str | None = None
    updated_at: str = ""

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> PullRequest:
        number = data.get("number")
        if not isinstance(number, int):
            raise ValueError("pull request payload is missing 'number'")
        base = _table(data, "base")
        head = _table(data, "head")
        return cls(
            number=number,
            title=_str(data, "title"),
            body=_str(data, "body"),
            state=_str(data, "state"),
            author=_str(_table(data, "user"), "login"),
            base_ref=_str(base, "ref"),
            base_sha=_str(base, "sha"),
            head_ref=_str(head, "ref"),
            head_sha=_str(head, "sha"),
            merged=_bool(data, "merged"),
            draft=_bool(data, "draft"),
            url=_str(data, "html_url"),
            changed_files=_int(data, "changed_files"),
            additions=_int(data, "additions"),
            deletions=_int(data, "deletions"),
            merged_at=_opt_str(data, "merged_at"),
            updated_at=_str(data, "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class PRFile:
    """A changed file in a pull request, including its patch hunks."""

    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None
    previous_filename: str | None

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> PRFile:
        filename = data.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("PR file payload is missing 'filename'")
        return cls(
            filename=filename,
            status=_str(data, "status"),
            additions=_int(data, "additions"),
            deletions=_int(data, "deletions"),
            changes=_int(data, "changes"),
            patch=_opt_str(data, "patch"),
            previous_filename=_opt_str(data, "previous_filename"),
        )


@dataclass(frozen=True, slots=True)
class ReviewComment:
    """An inline review comment on a pull request."""

    comment_id: int
    path: str
    body: str
    author: str
    line: int | None
    original_line: int | None
    start_line: int | None
    side: str | None
    commit_id: str
    created_at: str
    in_reply_to_id: int | None
    original_commit_id: str = ""  # the head the comment was originally made on

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> ReviewComment:
        comment_id = data.get("id")
        if not isinstance(comment_id, int):
            raise ValueError("review comment payload is missing 'id'")
        return cls(
            comment_id=comment_id,
            path=_str(data, "path"),
            body=_str(data, "body"),
            author=_str(_table(data, "user"), "login"),
            line=_opt_int(data, "line"),
            original_line=_opt_int(data, "original_line"),
            start_line=_opt_int(data, "start_line"),
            side=_opt_str(data, "side"),
            commit_id=_str(data, "commit_id"),
            created_at=_str(data, "created_at"),
            in_reply_to_id=_opt_int(data, "in_reply_to_id"),
            original_commit_id=_str(data, "original_commit_id"),
        )


@dataclass(frozen=True, slots=True)
class Review:
    """A submitted PR review (top-level, may carry inline comments)."""

    review_id: int
    author: str
    body: str
    state: str
    commit_id: str

    @classmethod
    def from_api(cls, data: Mapping[str, Any]) -> Review:
        review_id = data.get("id")
        if not isinstance(review_id, int):
            raise ValueError("review payload is missing 'id'")
        return cls(
            review_id=review_id,
            author=_str(_table(data, "user"), "login"),
            body=_str(data, "body"),
            state=_str(data, "state"),
            commit_id=_str(data, "commit_id"),
        )


@dataclass(frozen=True, slots=True)
class ReviewCommentDraft:
    """An inline comment to be posted (new-file side)."""

    path: str
    line: int
    body: str


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _opt_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _opt_int(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    return value if isinstance(value, bool) else False
