"""Tests for PR-head checkout, using a local git repo as origin (no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pr_review_agent.workspace import WorkspaceError, pr_head_workspace


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def origin(tmp_path: Path) -> tuple[str, str]:
    """A local 'origin' repo exposing its HEAD as refs/pull/7/head."""
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(["init", "--quiet", "-b", "main"], repo)
    (repo / "hello.py").write_text("GREETING = 'hi'\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "--quiet", "-m", "init"], repo)
    head_sha = _git(["rev-parse", "HEAD"], repo).strip()
    _git(["update-ref", "refs/pull/7/head", head_sha], repo)
    return repo.as_uri(), head_sha


def test_checkout_pr_head_and_cleanup(origin: tuple[str, str]) -> None:
    url, head_sha = origin
    with pr_head_workspace(url, 7, head_sha) as workdir:
        kept = workdir
        assert (workdir / "hello.py").read_text(encoding="utf-8") == "GREETING = 'hi'\n"
        checked_out = _git(["rev-parse", "HEAD"], workdir).strip()
        assert checked_out == head_sha
    assert not kept.exists()  # cleaned up on exit


def test_cleanup_happens_on_error_inside_block(origin: tuple[str, str]) -> None:
    url, head_sha = origin
    with pytest.raises(RuntimeError, match="boom"), pr_head_workspace(url, 7, head_sha) as workdir:
        kept = workdir
        raise RuntimeError("boom")
    assert not kept.exists()


def test_moved_head_sha_raises(origin: tuple[str, str]) -> None:
    url, _ = origin
    stale_sha = "0" * 40
    with pytest.raises(WorkspaceError, match="PR head moved"), pr_head_workspace(url, 7, stale_sha):
        pass  # pragma: no cover


def test_missing_pull_ref_raises(origin: tuple[str, str]) -> None:
    url, head_sha = origin
    with pytest.raises(WorkspaceError, match="failed"), pr_head_workspace(url, 99, head_sha):
        pass  # pragma: no cover
