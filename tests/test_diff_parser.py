"""Tests for the unified diff parser, including exact line-number mapping."""

from __future__ import annotations

import pytest

from conftest import load_fixture
from pr_review_agent.diff.models import DiffLine, FileStatus, LineKind
from pr_review_agent.diff.parser import DiffParseError, parse_diff


def test_empty_diff_parses_to_no_files() -> None:
    assert parse_diff("") == []
    assert parse_diff("\n\n") == []


def test_modified_file_paths_and_status() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "modify.diff"))
    assert file_diff.old_path == "src/app.py"
    assert file_diff.new_path == "src/app.py"
    assert file_diff.path == "src/app.py"
    assert file_diff.status is FileStatus.MODIFIED
    assert not file_diff.is_binary
    assert len(file_diff.hunks) == 2
    assert file_diff.additions == 3
    assert file_diff.deletions == 2


def test_modified_file_first_hunk_line_numbers_are_exact() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "modify.diff"))
    hunk = file_diff.hunks[0]
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 5, 1, 6)
    assert hunk.lines == (
        DiffLine(LineKind.CONTEXT, "import os", 1, 1),
        DiffLine(LineKind.REMOVED, "import sys", 2, None),
        DiffLine(LineKind.ADDED, "import sys", None, 2),
        DiffLine(LineKind.ADDED, "import json", None, 3),
        DiffLine(LineKind.CONTEXT, "", 3, 4),
        DiffLine(LineKind.CONTEXT, "def main() -> None:", 4, 5),
        DiffLine(LineKind.CONTEXT, "    pass", 5, 6),
    )


def test_modified_file_second_hunk_carries_section_and_offsets() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "modify.diff"))
    hunk = file_diff.hunks[1]
    assert hunk.section == "def helper():"
    assert hunk.lines == (
        DiffLine(LineKind.CONTEXT, "    a = 1", 20, 21),
        DiffLine(LineKind.REMOVED, "    b = 2", 21, None),
        DiffLine(LineKind.ADDED, "    b = 3", None, 22),
        DiffLine(LineKind.CONTEXT, "    return a", 22, 23),
    )


def test_new_file() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "new_file.diff"))
    assert file_diff.status is FileStatus.ADDED
    assert file_diff.old_path is None
    assert file_diff.new_path == "docs/notes.md"
    (hunk,) = file_diff.hunks
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (0, 0, 1, 2)
    assert hunk.lines == (
        DiffLine(LineKind.ADDED, "hello", None, 1),
        DiffLine(LineKind.ADDED, "world", None, 2),
    )


def test_deleted_file() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "deleted_file.diff"))
    assert file_diff.status is FileStatus.DELETED
    assert file_diff.old_path == "old_module.py"
    assert file_diff.new_path is None
    assert file_diff.path == "old_module.py"
    (hunk,) = file_diff.hunks
    assert hunk.lines == (
        DiffLine(LineKind.REMOVED, "x = 1", 1, None),
        DiffLine(LineKind.REMOVED, "y = 2", 2, None),
    )


def test_pure_rename_has_no_hunks() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "rename_pure.diff"))
    assert file_diff.status is FileStatus.RENAMED
    assert file_diff.old_path == "pkg/a.py"
    assert file_diff.new_path == "pkg/b.py"
    assert file_diff.hunks == ()
    assert not file_diff.is_binary


def test_rename_with_edits() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "rename_edit.diff"))
    assert file_diff.status is FileStatus.RENAMED
    assert file_diff.old_path == "utils/helpers.py"
    assert file_diff.new_path == "utils/util_helpers.py"
    (hunk,) = file_diff.hunks
    assert hunk.lines == (
        DiffLine(LineKind.CONTEXT, "def add(a, b):", 7, 7),
        DiffLine(LineKind.REMOVED, "    return a + b", 8, None),
        DiffLine(LineKind.ADDED, "    return a + b  # noqa", None, 8),
    )


def test_binary_file() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "binary.diff"))
    assert file_diff.is_binary
    assert file_diff.status is FileStatus.ADDED
    assert file_diff.path == "assets/logo.png"
    assert file_diff.hunks == ()
    assert file_diff.additions == 0
    assert file_diff.deletions == 0


def test_no_newline_markers_do_not_consume_line_numbers() -> None:
    (file_diff,) = parse_diff(load_fixture("diffs", "no_newline.diff"))
    (hunk,) = file_diff.hunks
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (1, 1, 1, 1)
    assert hunk.lines == (
        DiffLine(LineKind.REMOVED, "1.0.0", 1, None),
        DiffLine(LineKind.ADDED, "1.0.1", None, 1),
    )


def test_multi_file_diff_preserves_order_and_statuses() -> None:
    files = parse_diff(load_fixture("diffs", "multi.diff"))
    assert [(f.path, f.status) for f in files] == [
        ("src/app.py", FileStatus.MODIFIED),
        ("docs/notes.md", FileStatus.ADDED),
        ("old_module.py", FileStatus.DELETED),
        ("pkg/b.py", FileStatus.RENAMED),
        ("assets/logo.png", FileStatus.ADDED),
    ]
    assert files[4].is_binary
    assert sum(len(f.hunks) for f in files) == 4


def test_quoted_unicode_paths_are_decoded() -> None:
    diff = (
        'diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
        "index 1111111..2222222 100644\n"
        '--- "a/caf\\303\\251.py"\n'
        '+++ "b/caf\\303\\251.py"\n'
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    (file_diff,) = parse_diff(diff)
    assert file_diff.path == "café.py"
    assert file_diff.old_path == "café.py"


def test_hunk_count_shorthand_means_one() -> None:
    diff = (
        "diff --git a/f.txt b/f.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -3 +3 @@\n"
        "-old\n"
        "+new\n"
    )
    (file_diff,) = parse_diff(diff)
    (hunk,) = file_diff.hunks
    assert (hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) == (3, 1, 3, 1)
    assert hunk.lines[0].old_lineno == 3
    assert hunk.lines[1].new_lineno == 3


def test_mode_change_only_diff_has_no_hunks() -> None:
    diff = "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
    (file_diff,) = parse_diff(diff)
    assert file_diff.status is FileStatus.MODIFIED
    assert file_diff.path == "run.sh"
    assert file_diff.hunks == ()


def test_truncated_hunk_raises() -> None:
    diff = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n only one line\n"
    with pytest.raises(DiffParseError, match="truncated"):
        parse_diff(diff)


def test_excess_hunk_lines_raise() -> None:
    diff = (
        "diff --git a/f.txt b/f.txt\n"
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1,1 +1,1 @@\n"
        " context\n"
        "+surplus\n"
        "+surplus2\n"
    )
    with pytest.raises(DiffParseError):
        parse_diff(diff)


def test_garbage_outside_file_sections_raises() -> None:
    with pytest.raises(DiffParseError, match="outside any file section"):
        parse_diff("this is not a diff\n")


def test_unexpected_header_line_raises() -> None:
    diff = (
        "diff --git a/f.txt b/f.txt\n"
        "??? bogus extended header\n"
        "--- a/f.txt\n"
        "+++ b/f.txt\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    with pytest.raises(DiffParseError, match="unexpected line in file header"):
        parse_diff(diff)
