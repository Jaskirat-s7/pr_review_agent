"""Structured representation of a unified diff with exact line numbers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FileStatus(StrEnum):
    """How a file changed in the diff."""

    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"
    RENAMED = "renamed"
    COPIED = "copied"


class LineKind(StrEnum):
    """The role of a single diff line."""

    CONTEXT = "context"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class DiffLine:
    """One line of a hunk, with its line number on each side.

    ``old_lineno`` is ``None`` for added lines; ``new_lineno`` is ``None``
    for removed lines. ``content`` excludes the leading ``+``/``-``/space.
    """

    kind: LineKind
    content: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass(frozen=True, slots=True)
class Hunk:
    """A contiguous block of changes, as declared by an ``@@`` header."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[DiffLine, ...]


@dataclass(frozen=True, slots=True)
class FileDiff:
    """All changes to a single file.

    ``old_path`` is ``None`` for added files; ``new_path`` is ``None`` for
    deleted files. Binary files and pure renames have no hunks.
    """

    old_path: str | None
    new_path: str | None
    status: FileStatus
    is_binary: bool
    hunks: tuple[Hunk, ...]

    @property
    def path(self) -> str:
        """The current path of the file (old path for deletions)."""
        path = self.new_path if self.new_path is not None else self.old_path
        if path is None:  # pragma: no cover - parser never constructs this
            raise ValueError("file diff has neither old nor new path")
        return path

    @property
    def additions(self) -> int:
        return sum(1 for h in self.hunks for line in h.lines if line.kind is LineKind.ADDED)

    @property
    def deletions(self) -> int:
        return sum(1 for h in self.hunks for line in h.lines if line.kind is LineKind.REMOVED)
