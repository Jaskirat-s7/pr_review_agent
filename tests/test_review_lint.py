"""Tests for the ruff/mypy lint-dedup runner (runs the real tools)."""

from __future__ import annotations

from pathlib import Path

from pr_review_agent.review.lint import RuffMypyRunner

BAD_SOURCE = """\
import os


def f(x: int) -> str:
    return x
"""


def test_finds_ruff_and_mypy_locations(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(BAD_SOURCE, encoding="utf-8")
    findings = RuffMypyRunner().findings(tmp_path, ["bad.py"])
    assert ("bad.py", 1) in findings  # unused import (F401, found by the linter)
    assert ("bad.py", 5) in findings  # returning int from -> str (found by the type checker)


def test_non_python_and_missing_files_yield_nothing(tmp_path: Path) -> None:
    runner = RuffMypyRunner()
    assert runner.findings(tmp_path, ["README.md"]) == set()
    assert runner.findings(tmp_path, ["ghost.py"]) == set()
