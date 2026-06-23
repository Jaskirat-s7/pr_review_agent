"""Finding the Python files to index under a checked-out repo."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# Directories never worth indexing: VCS metadata, virtualenvs, caches, build
# outputs. Matched by exact name at any depth.
_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}
)


def walk_python_files(root: Path) -> Iterator[Path]:
    """Yield ``*.py`` files under ``root``, skipping VCS/venv/cache/build dirs.

    Hidden directories (dot-prefixed) are skipped too; hidden files are not.
    Paths are yielded in sorted order for a deterministic index.
    """
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            yield from walk_python_files(entry)
        elif entry.suffix == ".py":
            yield entry
