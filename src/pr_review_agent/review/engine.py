"""The agent loop: triage every hunk cheaply, fully review only flagged
hunks with retrieved context, then gate, dedup, and cap the results.

The engine owns no I/O beyond the ModelClient it is given — callers wire in
a CachingModelClient so every call is cached and cost-tracked.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pr_review_agent.config import ReviewConfig
from pr_review_agent.context.models import FileContext, PRContext
from pr_review_agent.diff.models import FileDiff, FileStatus, Hunk, LineKind
from pr_review_agent.models.base import ModelClient, ModelError, ModelMessage
from pr_review_agent.prompts import load_prompt
from pr_review_agent.review.lint import LintRunner
from pr_review_agent.review.models import AgentComment, ReviewResult, ReviewStats, Severity

logger = logging.getLogger(__name__)

_TRIAGE_MAX_TOKENS = 128
_REVIEW_MAX_TOKENS = 1536


class ReviewEngine:
    """Two-tier PR review over parsed diffs and retrieved context."""

    def __init__(
        self,
        client: ModelClient,
        *,
        config: ReviewConfig,
        lint_runner: LintRunner | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._lint = lint_runner
        self._triage_system = load_prompt("triage_system").template
        self._triage_user = load_prompt("triage_user")
        self._review_system = load_prompt("review_system").template
        self._review_user = load_prompt("review_user")

    def review(
        self,
        files: Sequence[FileDiff],
        pr_context: PRContext,
        repo_root: Path | None = None,
    ) -> ReviewResult:
        context_by_file = {fc.file_path: fc for fc in pr_context.files}
        hunks_total = 0
        triage_failures = 0
        flagged: list[tuple[FileDiff, Hunk, str]] = []

        for file_diff in files:
            if file_diff.is_binary or file_diff.status is FileStatus.DELETED:
                continue
            for hunk in file_diff.hunks:
                hunks_total += 1
                verdict = self._triage(file_diff, hunk)
                if verdict is None:
                    triage_failures += 1
                elif verdict[0]:
                    flagged.append((file_diff, hunk, verdict[1]))

        drafts: list[AgentComment] = []
        dropped_invalid_line = 0
        review_failures = 0
        for file_diff, hunk, category in flagged:
            result = self._review_hunk(
                file_diff, hunk, category, context_by_file.get(file_diff.path)
            )
            if result is None:
                review_failures += 1
                continue
            comments, invalid = result
            drafts.extend(comments)
            dropped_invalid_line += invalid

        kept: list[AgentComment] = []
        dropped_low_confidence = 0
        for draft in drafts:
            threshold = (
                self._config.confidence_threshold
                if draft.has_context
                else self._config.no_context_confidence_threshold
            )
            if draft.confidence < threshold:
                dropped_low_confidence += 1
            else:
                kept.append(draft)

        dropped_lint_duplicate = 0
        if self._lint is not None and repo_root is not None and kept:
            lint_hits = self._lint.findings(repo_root, sorted({c.file_path for c in kept}))
            deduped = [c for c in kept if (c.file_path, c.line) not in lint_hits]
            dropped_lint_duplicate = len(kept) - len(deduped)
            kept = deduped

        kept.sort(key=lambda c: (-c.severity.rank, -c.confidence, c.file_path, c.line))
        dropped_over_cap = max(0, len(kept) - self._config.max_comments)
        kept = kept[: self._config.max_comments]
        kept.sort(key=lambda c: (c.file_path, c.line))

        stats = ReviewStats(
            hunks_total=hunks_total,
            hunks_flagged=len(flagged),
            drafts_generated=len(drafts),
            dropped_low_confidence=dropped_low_confidence,
            dropped_invalid_line=dropped_invalid_line,
            dropped_lint_duplicate=dropped_lint_duplicate,
            dropped_over_cap=dropped_over_cap,
            triage_failures=triage_failures,
            review_failures=review_failures,
        )
        return ReviewResult(comments=tuple(kept), stats=stats)

    # -- model passes ---------------------------------------------------------

    def _triage(self, file_diff: FileDiff, hunk: Hunk) -> tuple[bool, str] | None:
        user = self._triage_user.substitute(
            file_path=file_diff.path,
            status=file_diff.status.value,
            hunk=render_hunk(hunk),
        )
        try:
            response = self._client.complete(
                self._triage_system,
                [ModelMessage("user", user)],
                max_tokens=_TRIAGE_MAX_TOKENS,
                purpose="triage",
            )
        except ModelError as exc:
            logger.warning("triage call failed for %s: %s", file_diff.path, exc)
            return None
        payload = _parse_json(response.text)
        if not isinstance(payload, dict) or not isinstance(payload.get("worth_reviewing"), bool):
            logger.warning(
                "unparseable triage verdict for %s: %r", file_diff.path, response.text[:120]
            )
            return None
        category = payload.get("category")
        return bool(payload["worth_reviewing"]), str(category) if category else "other"

    def _review_hunk(
        self,
        file_diff: FileDiff,
        hunk: Hunk,
        category: str,
        file_context: FileContext | None,
    ) -> tuple[list[AgentComment], int] | None:
        has_context = file_context is not None and bool(file_context.symbols)
        user = self._review_user.substitute(
            file_path=file_diff.path,
            category=category,
            context=render_context(file_context),
            hunk=render_hunk(hunk),
        )
        try:
            response = self._client.complete(
                self._review_system,
                [ModelMessage("user", user)],
                max_tokens=_REVIEW_MAX_TOKENS,
                purpose="review",
            )
        except ModelError as exc:
            logger.warning("review call failed for %s: %s", file_diff.path, exc)
            return None
        payload = _parse_json(response.text)
        if not isinstance(payload, list):
            logger.warning(
                "unparseable review output for %s: %r", file_diff.path, response.text[:120]
            )
            return None

        anchorable = {line.new_lineno for line in hunk.lines if line.new_lineno is not None}
        comments: list[AgentComment] = []
        invalid_lines = 0
        for item in payload:
            comment = _comment_from_item(item, file_diff.path, category, has_context)
            if comment is None:
                continue
            if comment.line not in anchorable:
                invalid_lines += 1
                continue
            comments.append(comment)
        return comments, invalid_lines


def render_hunk(hunk: Hunk) -> str:
    """Render a hunk with new-file line numbers as anchors."""
    header = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
    if hunk.section:
        header += f" {hunk.section}"
    lines = [header]
    for line in hunk.lines:
        if line.kind is LineKind.ADDED:
            lines.append(f"{line.new_lineno:>5} + {line.content}")
        elif line.kind is LineKind.REMOVED:
            lines.append(f"{'':>5} - {line.content}")
        else:
            lines.append(f"{line.new_lineno:>5}   {line.content}")
    return "\n".join(lines)


def render_context(file_context: FileContext | None) -> str:
    if file_context is None or not file_context.symbols:
        return "No repository context was retrieved for this file."
    parts = [
        f"# {symbol.module_path}:{symbol.lineno} ({symbol.kind.value})\n{symbol.source}"
        for symbol in file_context.symbols
    ]
    return "\n\n".join(parts)


def _comment_from_item(
    item: object, file_path: str, category: str, has_context: bool
) -> AgentComment | None:
    if not isinstance(item, dict):
        return None
    line = item.get("line")
    body = item.get("comment")
    confidence = item.get("confidence")
    if not isinstance(line, int) or not isinstance(body, str) or not body.strip():
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None
    severity_raw = item.get("severity")
    try:
        severity = Severity(str(severity_raw))
    except ValueError:
        severity = Severity.MINOR  # severity is advisory; keep the finding
    return AgentComment(
        file_path=file_path,
        line=line,
        severity=severity,
        confidence=max(0.0, min(1.0, float(confidence))),
        body=body.strip(),
        category=category,
        has_context=has_context,
    )


def _parse_json(text: str) -> Any:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
