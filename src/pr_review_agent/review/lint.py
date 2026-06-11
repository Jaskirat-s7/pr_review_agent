"""Linter findings used to drop agent comments a linter would already make.

Best-effort by design: a missing ruff/mypy executable degrades to "no
findings" with a warning rather than failing the review run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


def _find_tool(name: str) -> str | None:
    """Locate a tool on PATH or next to the running interpreter (venv bin)."""
    found = shutil.which(name)
    if found is not None:
        return found
    sibling = Path(sys.executable).parent / name
    return str(sibling) if sibling.is_file() else None


class LintRunner(Protocol):
    """Yields (file_path, line) pairs a linter flags in the given files."""

    def findings(self, repo_root: Path, files: Sequence[str]) -> set[tuple[str, int]]: ...


class RuffMypyRunner:
    """Runs ruff and mypy on the PR head workspace and collects locations."""

    def findings(self, repo_root: Path, files: Sequence[str]) -> set[tuple[str, int]]:
        python_files = [f for f in files if f.endswith(".py") and (repo_root / f).is_file()]
        if not python_files:
            return set()
        return self._ruff(repo_root, python_files) | self._mypy(repo_root, python_files)

    def _ruff(self, repo_root: Path, files: list[str]) -> set[tuple[str, int]]:
        executable = _find_tool("ruff")
        if executable is None:
            logger.warning("ruff not found on PATH; skipping lint dedup for ruff")
            return set()
        result = subprocess.run(
            [executable, "check", "--output-format", "json", "--exit-zero", *files],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("ruff failed (%d): %s", result.returncode, result.stderr[:200])
            return set()
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            logger.warning("could not parse ruff JSON output")
            return set()
        found: set[tuple[str, int]] = set()
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                location = item.get("location")
                if isinstance(filename, str) and isinstance(location, dict):
                    row = location.get("row")
                    if isinstance(row, int):
                        found.add((_relative(filename, repo_root), row))
        return found

    def _mypy(self, repo_root: Path, files: list[str]) -> set[tuple[str, int]]:
        executable = _find_tool("mypy")
        if executable is None:
            logger.warning("mypy not found on PATH; skipping lint dedup for mypy")
            return set()
        result = subprocess.run(
            [
                executable,
                "--ignore-missing-imports",
                "--follow-imports=silent",
                "--no-error-summary",
                *files,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in (0, 1):  # 2+ means mypy itself crashed
            logger.warning("mypy failed (%d): %s", result.returncode, result.stderr[:200])
            return set()
        found: set[tuple[str, int]] = set()
        for line in result.stdout.splitlines():
            # Format: path:line: error: message  [code]
            parts = line.split(":", 2)
            if len(parts) >= 3 and parts[1].strip().isdigit() and "error" in parts[2]:
                found.add((_relative(parts[0].strip(), repo_root), int(parts[1])))
        return found


def _relative(filename: str, repo_root: Path) -> str:
    path = Path(filename)
    if path.is_absolute():
        with contextlib.suppress(ValueError):
            path = path.relative_to(repo_root.resolve())
    return path.as_posix()
