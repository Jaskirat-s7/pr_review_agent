"""Temporary checkout of a pull request's head commit.

Uses GitHub's ``refs/pull/<n>/head`` ref with a depth-1 fetch, so the
workspace holds exactly the tree the diff was computed against. The fetched
commit is verified against the head SHA from the PR metadata; if the PR moved
between the API call and the fetch, we fail loudly rather than reviewing one
tree with anchors from another.

Auth never touches argv or on-disk config: the token is passed to git as an
``Authorization`` header through ``GIT_CONFIG_*`` environment variables.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceError(Exception):
    """Raised when the PR head cannot be checked out."""


@contextmanager
def pr_head_workspace(
    clone_url: str,
    pr_number: int,
    expected_head_sha: str,
    *,
    token: str | None = None,
) -> Iterator[Path]:
    """Yield a temp directory containing the PR head checkout; clean up after."""
    if shutil.which("git") is None:
        raise WorkspaceError("git executable not found on PATH")
    workdir = Path(tempfile.mkdtemp(prefix="pra-workspace-"))
    try:
        _checkout_pr_head(workdir, clone_url, pr_number, expected_head_sha, token)
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@contextmanager
def commit_workspace(
    clone_url: str,
    sha: str,
    *,
    token: str | None = None,
) -> Iterator[Path]:
    """Yield a temp checkout of an arbitrary commit SHA; clean up after.

    Used by the eval harness to review a PR's pre-review state (a commit that
    is not necessarily the current head). Raises :class:`WorkspaceError` if
    the commit cannot be fetched — the caller degrades to reviewing without
    repository context rather than failing the whole run.
    """
    if shutil.which("git") is None:
        raise WorkspaceError("git executable not found on PATH")
    workdir = Path(tempfile.mkdtemp(prefix="pra-workspace-"))
    try:
        env = _git_env(token)
        git = ["git", "-C", str(workdir)]
        _run([*git, "init", "--quiet"], env)
        _run([*git, "fetch", "--quiet", "--depth", "1", clone_url, sha], env)
        fetched = _run([*git, "rev-parse", "FETCH_HEAD"], env).strip()
        if fetched != sha:
            raise WorkspaceError(f"fetched {fetched} but requested {sha}")
        _run([*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"], env)
        logger.info("checked out commit %s", sha[:12])
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _checkout_pr_head(
    workdir: Path,
    clone_url: str,
    pr_number: int,
    expected_head_sha: str,
    token: str | None,
) -> None:
    env = _git_env(token)
    git = ["git", "-C", str(workdir)]
    _run([*git, "init", "--quiet"], env)
    _run(
        [*git, "fetch", "--quiet", "--depth", "1", clone_url, f"refs/pull/{pr_number}/head"],
        env,
    )
    fetched = _run([*git, "rev-parse", "FETCH_HEAD"], env).strip()
    if fetched != expected_head_sha:
        raise WorkspaceError(
            f"PR head moved: expected {expected_head_sha}, fetched {fetched}; "
            "re-fetch the PR metadata and retry"
        )
    _run([*git, "checkout", "--quiet", "--detach", "FETCH_HEAD"], env)
    logger.info("checked out PR #%d head %s", pr_number, fetched[:12])


def _git_env(token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {basic}"
    return env


def _run(args: list[str], env: dict[str, str]) -> str:
    # args never contain the token, so they are safe to echo in errors.
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise WorkspaceError(f"'{' '.join(args)}' failed: {detail}")
    return result.stdout
