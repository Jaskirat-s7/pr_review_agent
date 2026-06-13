"""Run the agent over eval cases and collect its comments.

Each case carries the diff to review (already reconstructed at the pre-review
commit). Context retrieval needs the repository tree at that commit, so we
check out ``review_sha`` into a temp workspace. If that checkout fails (the
commit was garbage-collected, or git/network is unavailable), the case is
still reviewed — just without repository context — rather than failing the
whole batch; those comments are tagged has_context=False and the report's
context split reflects it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from pr_review_agent.config import ContextConfig, ReviewConfig
from pr_review_agent.context.models import PRContext
from pr_review_agent.context.retriever import ContextRetriever
from pr_review_agent.diff.parser import parse_diff
from pr_review_agent.evals.schema import EvalCase, RunComment, RunResult
from pr_review_agent.models.base import ModelClient
from pr_review_agent.review.engine import ReviewEngine
from pr_review_agent.review.lint import RuffMypyRunner
from pr_review_agent.review.models import ReviewResult
from pr_review_agent.workspace import WorkspaceError, commit_workspace

logger = logging.getLogger(__name__)

# (clone_url, sha, *, token) -> context manager yielding a checkout dir
WorkspaceFactory = Callable[..., AbstractContextManager[Path]]

_EMPTY_CONTEXT = PRContext(files=(), total_tokens=0, dropped_symbols=0)


def review_case(
    case: EvalCase,
    client: ModelClient,
    *,
    review_config: ReviewConfig,
    context_config: ContextConfig,
    clone_url: str,
    token: str | None,
    workspace_factory: WorkspaceFactory | None = None,
) -> ReviewResult:
    """Run the two-tier engine over one case, with repo context when available."""
    # Resolved at call time (not as a default) so the module global stays
    # monkeypatchable in tests.
    factory = workspace_factory or commit_workspace
    files = parse_diff(case.diff)
    try:
        with factory(clone_url, case.review_sha, token=token) as workdir:
            pr_context = ContextRetriever(
                workdir, token_budget=context_config.token_budget
            ).retrieve(files)
            engine = ReviewEngine(client, config=review_config, lint_runner=RuffMypyRunner())
            return engine.review(files, pr_context, repo_root=workdir)
    except WorkspaceError as exc:
        logger.warning(
            "checkout of %s@%s failed (%s); reviewing without repository context",
            case.repo,
            case.review_sha[:8],
            exc,
        )
        return ReviewEngine(client, config=review_config).review(files, _EMPTY_CONTEXT)


def to_run_comments(result: ReviewResult) -> tuple[RunComment, ...]:
    return tuple(
        RunComment(
            file_path=c.file_path,
            line=c.line,
            severity=c.severity.value,
            confidence=c.confidence,
            body=c.body,
            category=c.category,
            has_context=c.has_context,
        )
        for c in result.comments
    )


def build_run_result(
    case: EvalCase,
    result: ReviewResult,
    *,
    backend: str,
    model: str,
    cost_usd: float,
) -> RunResult:
    return RunResult(
        repo=case.repo,
        number=case.number,
        review_sha=case.review_sha,
        backend=backend,
        model=model,
        comments=to_run_comments(result),
        cost_usd=cost_usd,
        failures=result.stats.triage_failures + result.stats.review_failures,
    )
