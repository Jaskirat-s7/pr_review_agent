"""Tests for posting helpers: run key, marker, idempotency, comment formatting."""

from __future__ import annotations

from pr_review_agent.github.models import Review
from pr_review_agent.review.models import AgentComment, Severity
from pr_review_agent.review.post import (
    find_existing_review,
    marker,
    review_body,
    run_key,
    to_draft_comments,
)

KEY = run_key("octo/widgets", 7, "abc123")


def _review(author: str, body: str) -> Review:
    return Review(review_id=1, author=author, body=body, state="COMMENTED", commit_id="abc123")


def test_run_key_is_deterministic_per_head() -> None:
    assert KEY == "octo/widgets#7@abc123"
    assert run_key("octo/widgets", 7, "def456") != KEY  # new push → new key


def test_marker_is_an_invisible_html_comment() -> None:
    assert marker(KEY) == "<!-- pr-review-agent:octo/widgets#7@abc123 -->"


def test_existing_review_found_only_when_bot_authored() -> None:
    bot_review = _review("pra-bot", f"Automated review\n\n{marker(KEY)}")
    foreign_copy = _review("mallory", f"hah, copied your marker: {marker(KEY)}")
    bot_other_key = _review("pra-bot", f"older run\n\n{marker(run_key('octo/widgets', 7, 'old'))}")

    assert find_existing_review([foreign_copy, bot_review], KEY, "pra-bot") is bot_review
    assert find_existing_review([foreign_copy], KEY, "pra-bot") is None  # marker copy ignored
    assert find_existing_review([bot_other_key], KEY, "pra-bot") is None  # different head SHA


def test_review_body_contains_marker_and_count() -> None:
    body = review_body(KEY, [])
    assert marker(KEY) in body
    assert "0 comment(s)" in body


def test_draft_comments_carry_severity_and_confidence_footer() -> None:
    comment = AgentComment(
        file_path="app.py",
        line=11,
        severity=Severity.MAJOR,
        confidence=0.9,
        body="KeyError when 'key' is missing.",
        category="bug",
        has_context=True,
    )
    (draft,) = to_draft_comments([comment])
    assert draft.path == "app.py"
    assert draft.line == 11
    assert draft.body.startswith("KeyError when 'key' is missing.")
    assert "severity: major" in draft.body
    assert "confidence: 0.90" in draft.body
