"""Tests for the Python-file walk."""

from __future__ import annotations

from pathlib import Path

from pr_review_agent.rag.walk import walk_python_files


def test_walk_selects_py_and_skips_noise(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "notes.md").write_text("", encoding="utf-8")
    (tmp_path / "top.py").write_text("", encoding="utf-8")
    for noise in (".git", ".venv", "__pycache__", "build"):
        (tmp_path / noise).mkdir()
        (tmp_path / noise / "junk.py").write_text("", encoding="utf-8")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_python_files(tmp_path)}
    assert found == {"pkg/a.py", "top.py"}


def test_walk_is_sorted(tmp_path: Path) -> None:
    for name in ("c.py", "a.py", "b.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    names = [p.name for p in walk_python_files(tmp_path)]
    assert names == ["a.py", "b.py", "c.py"]
