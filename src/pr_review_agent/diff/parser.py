"""Exact unified-diff parser.

Review comments need correct line anchors, so this parser is strict: hunk
bodies are consumed by the exact line counts declared in the ``@@`` header,
and any unexpected content raises :class:`DiffParseError` instead of being
skipped.
"""

from __future__ import annotations

import re

from pr_review_agent.diff.models import DiffLine, FileDiff, FileStatus, Hunk, LineKind

_GIT_HEADER_PREFIX = "diff --git "
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?$")
_GIT_HEADER_PATHS_RE = re.compile(r"^a/(.*) b/(.*)$")
_GIT_HEADER_QUOTED_RE = re.compile(r'^("(?:[^"\\]|\\.)*") (.+)$')

# Extended header lines that carry no information we keep.
_IGNORED_HEADER_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "mode ",
)


class DiffParseError(Exception):
    """Raised when a diff cannot be parsed exactly."""


def parse_diff(text: str) -> list[FileDiff]:
    """Parse a git-style unified diff into a list of :class:`FileDiff`."""
    lines = text.splitlines()
    files: list[FileDiff] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(_GIT_HEADER_PREFIX):
            file_diff, i = _parse_file(lines, i)
            files.append(file_diff)
        elif not line.strip():
            i += 1
        else:
            raise DiffParseError(f"unexpected content outside any file section: {line!r}")
    return files


def _parse_file(lines: list[str], i: int) -> tuple[FileDiff, int]:
    git_old, git_new = _paths_from_git_header(lines[i])
    i += 1
    status = FileStatus.MODIFIED
    is_binary = False
    rename_old: str | None = None
    rename_new: str | None = None
    marker_old: str | None = None
    marker_new: str | None = None
    saw_old_marker = False
    saw_new_marker = False
    hunks: list[Hunk] = []

    while i < len(lines):
        line = lines[i]
        if line.startswith(_GIT_HEADER_PREFIX):
            break
        if line.startswith("@@"):
            if is_binary:
                raise DiffParseError(f"hunk found in binary file section: {line!r}")
            hunk, i = _parse_hunk(lines, i)
            hunks.append(hunk)
            continue
        if line.startswith("new file mode "):
            status = FileStatus.ADDED
        elif line.startswith("deleted file mode "):
            status = FileStatus.DELETED
        elif line.startswith("rename from "):
            status = FileStatus.RENAMED
            rename_old = line[len("rename from ") :]
        elif line.startswith("rename to "):
            rename_new = line[len("rename to ") :]
        elif line.startswith("copy from "):
            status = FileStatus.COPIED
            rename_old = line[len("copy from ") :]
        elif line.startswith("copy to "):
            rename_new = line[len("copy to ") :]
        elif line.startswith("--- "):
            marker_old = _marker_path(line[4:], "a/")
            saw_old_marker = True
        elif line.startswith("+++ "):
            marker_new = _marker_path(line[4:], "b/")
            saw_new_marker = True
        elif line.startswith("Binary files ") and line.endswith(" differ"):
            is_binary = True
        elif line == "GIT binary patch":
            is_binary = True
            i += 1
            while i < len(lines) and not lines[i].startswith(_GIT_HEADER_PREFIX):
                i += 1
            continue
        elif line.startswith(_IGNORED_HEADER_PREFIXES) or not line.strip():
            pass
        else:
            raise DiffParseError(f"unexpected line in file header: {line!r}")
        i += 1

    # `--- /dev/null` / `+++ /dev/null` imply added/deleted even when the
    # explicit mode lines are absent.
    if saw_old_marker and marker_old is None and status is FileStatus.MODIFIED:
        status = FileStatus.ADDED
    if saw_new_marker and marker_new is None and status is FileStatus.MODIFIED:
        status = FileStatus.DELETED

    # Path priority: ---/+++ markers > rename/copy lines > diff --git header
    # (the last is ambiguous for paths containing spaces).
    old_path = marker_old if saw_old_marker else (rename_old or git_old)
    new_path = marker_new if saw_new_marker else (rename_new or git_new)
    if status is FileStatus.ADDED:
        old_path = None
    elif status is FileStatus.DELETED:
        new_path = None
    if old_path is None and new_path is None:
        raise DiffParseError("could not determine file paths for a diff section")

    file_diff = FileDiff(
        old_path=old_path,
        new_path=new_path,
        status=status,
        is_binary=is_binary,
        hunks=tuple(hunks),
    )
    return file_diff, i


def _parse_hunk(lines: list[str], i: int) -> tuple[Hunk, int]:
    match = _HUNK_HEADER_RE.match(lines[i])
    if match is None:
        raise DiffParseError(f"malformed hunk header: {lines[i]!r}")
    old_start = int(match[1])
    old_count = int(match[2]) if match[2] is not None else 1
    new_start = int(match[3])
    new_count = int(match[4]) if match[4] is not None else 1
    section = match[5] or ""
    i += 1

    body: list[DiffLine] = []
    old_lineno = old_start
    new_lineno = new_start
    old_left = old_count
    new_left = new_count
    while old_left > 0 or new_left > 0:
        if i >= len(lines):
            raise DiffParseError(
                f"diff truncated inside hunk @@ -{old_start},{old_count} "
                f"+{new_start},{new_count} @@"
            )
        line = lines[i]
        if line.startswith("\\"):
            # "\ No newline at end of file": consumes no line numbers.
            i += 1
            continue
        tag = line[:1] or " "  # a trimmed empty line is an empty context line
        content = line[1:]
        if tag == " ":
            if old_left <= 0 or new_left <= 0:
                raise DiffParseError(f"hunk has more lines than its header declares: {line!r}")
            body.append(DiffLine(LineKind.CONTEXT, content, old_lineno, new_lineno))
            old_lineno += 1
            new_lineno += 1
            old_left -= 1
            new_left -= 1
        elif tag == "-":
            if old_left <= 0:
                raise DiffParseError(f"hunk has more '-' lines than its header declares: {line!r}")
            body.append(DiffLine(LineKind.REMOVED, content, old_lineno, None))
            old_lineno += 1
            old_left -= 1
        elif tag == "+":
            if new_left <= 0:
                raise DiffParseError(f"hunk has more '+' lines than its header declares: {line!r}")
            body.append(DiffLine(LineKind.ADDED, content, None, new_lineno))
            new_lineno += 1
            new_left -= 1
        else:
            raise DiffParseError(f"unexpected line inside hunk: {line!r}")
        i += 1

    # A no-newline marker may also follow the final hunk line.
    if i < len(lines) and lines[i].startswith("\\"):
        i += 1

    hunk = Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        section=section,
        lines=tuple(body),
    )
    return hunk, i


def _paths_from_git_header(line: str) -> tuple[str | None, str | None]:
    """Best-effort path extraction from a ``diff --git a/x b/x`` line.

    Ambiguous for unquoted paths containing spaces; callers prefer the
    ``---``/``+++`` markers and rename/copy lines when present.
    """
    rest = line[len(_GIT_HEADER_PREFIX) :]
    if rest.startswith('"'):
        match = _GIT_HEADER_QUOTED_RE.match(rest)
        if match is None:
            return None, None
        old = _strip_prefix(_unquote(match[1]), "a/")
        new_raw = match[2]
        new = _strip_prefix(_unquote(new_raw) if new_raw.startswith('"') else new_raw, "b/")
        return old, new
    match = _GIT_HEADER_PATHS_RE.match(rest)
    if match is None:
        return None, None
    return match[1], match[2]


def _marker_path(raw: str, prefix: str) -> str | None:
    """Parse the path from a ``--- ``/``+++ `` line; ``None`` for /dev/null."""
    value = _unquote(raw) if raw.startswith('"') else raw.split("\t", 1)[0]
    if value == "/dev/null":
        return None
    return _strip_prefix(value, prefix)


def _strip_prefix(path: str, prefix: str) -> str:
    return path[len(prefix) :] if path.startswith(prefix) else path


def _unquote(quoted: str) -> str:
    """Decode a git C-style quoted path (octal escapes encode UTF-8 bytes)."""
    inner = quoted[1:-1] if quoted.startswith('"') and quoted.endswith('"') else quoted
    decoded = inner.encode("utf-8").decode("unicode_escape")
    return decoded.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
