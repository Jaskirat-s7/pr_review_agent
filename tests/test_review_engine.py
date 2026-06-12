"""Tests for the two-tier review engine, using a scripted ModelClient."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pr_review_agent.config import ReviewConfig
from pr_review_agent.context.models import FileContext, PRContext, SymbolDef, SymbolKind
from pr_review_agent.diff.parser import parse_diff
from pr_review_agent.models.base import ModelMessage, ModelResponse
from pr_review_agent.review.engine import ReviewEngine, render_hunk
from pr_review_agent.review.models import Severity

DIFF = """\
diff --git a/app.py b/app.py
index 1111111..2222222 100644
--- a/app.py
+++ b/app.py
@@ -10,2 +10,4 @@ def handler():
     data = load()
+    result = data['key']
+    save(result)
     return data
@@ -30,2 +31,2 @@ def footer():
-    # old comment
+    # new comment
     pass
"""


class ScriptedClient:
    """Returns queued responses per purpose; records the prompts it saw."""

    def __init__(self, triage: list[str], review: list[str]) -> None:
        self._queues = {"triage": list(triage), "review": list(review)}
        self.prompts: dict[str, list[str]] = {"triage": [], "review": []}

    @property
    def model(self) -> str:
        return "scripted"

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        self.prompts[purpose].append(messages[0].content)
        text = self._queues[purpose].pop(0)
        return ModelResponse(text=text, model="scripted", input_tokens=10, output_tokens=5)


def triage_yes(category: str = "bug") -> str:
    return json.dumps({"worth_reviewing": True, "category": category})


def triage_no() -> str:
    return json.dumps({"worth_reviewing": False, "category": "none"})


def review_json(*comments: dict[str, object]) -> str:
    return json.dumps(list(comments))


def context_with_symbols(file_path: str = "app.py") -> PRContext:
    symbol = SymbolDef(
        module_path="lib.py",
        name="save",
        kind=SymbolKind.FUNCTION,
        source="def save(x):\n    ...",
        lineno=1,
        end_lineno=2,
        est_tokens=8,
        reference_count=1,
    )
    return PRContext(
        files=(FileContext(file_path, (symbol,), ()),), total_tokens=8, dropped_symbols=0
    )


def empty_context(file_path: str = "app.py") -> PRContext:
    return PRContext(files=(FileContext(file_path, (), ()),), total_tokens=0, dropped_symbols=0)


def engine(client: ScriptedClient, **overrides: float | int) -> ReviewEngine:
    config = ReviewConfig(
        max_comments=int(overrides.get("max_comments", 3)),
        confidence_threshold=float(overrides.get("confidence_threshold", 0.6)),
        no_context_confidence_threshold=float(
            overrides.get("no_context_confidence_threshold", 0.8)
        ),
    )
    return ReviewEngine(client, config=config)


def test_review_runs_only_on_flagged_hunks_and_gates_confidence() -> None:
    client = ScriptedClient(
        triage=[triage_yes(), triage_no()],
        review=[
            review_json(
                {"line": 11, "severity": "major", "confidence": 0.9, "comment": "KeyError risk."},
                {"line": 12, "severity": "minor", "confidence": 0.3, "comment": "Maybe."},
            )
        ],
    )
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    assert len(client.prompts["review"]) == 1  # second hunk triaged out
    (comment,) = result.comments
    assert comment.line == 11
    assert comment.severity is Severity.MAJOR
    assert comment.has_context
    assert result.stats.hunks_total == 2
    assert result.stats.hunks_flagged == 1
    assert result.stats.dropped_low_confidence == 1


def test_no_context_comments_face_higher_threshold() -> None:
    review = review_json(
        {"line": 11, "severity": "major", "confidence": 0.7, "comment": "Unchecked key."}
    )
    # With context: 0.7 >= 0.6 → kept.
    client = ScriptedClient(triage=[triage_yes(), triage_no()], review=[review])
    with_context = engine(client).review(parse_diff(DIFF), context_with_symbols())
    assert len(with_context.comments) == 1

    # Without context: 0.7 < 0.8 → dropped.
    client = ScriptedClient(triage=[triage_yes(), triage_no()], review=[review])
    without_context = engine(client).review(parse_diff(DIFF), empty_context())
    assert without_context.comments == ()
    assert without_context.stats.dropped_low_confidence == 1
    assert not_has_context_was_tagged(client)


def not_has_context_was_tagged(client: ScriptedClient) -> bool:
    return "No repository context was retrieved" in client.prompts["review"][0]


def test_invalid_line_anchor_is_dropped() -> None:
    client = ScriptedClient(
        triage=[triage_yes(), triage_no()],
        review=[
            review_json(
                {"line": 999, "severity": "major", "confidence": 0.9, "comment": "Ghost line."}
            )
        ],
    )
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    assert result.comments == ()
    assert result.stats.dropped_invalid_line == 1


def test_lint_duplicates_are_dropped() -> None:
    class FakeLint:
        def findings(self, repo_root: Path, files: Sequence[str]) -> set[tuple[str, int]]:
            return {("app.py", 11)}

    client = ScriptedClient(
        triage=[triage_yes(), triage_no()],
        review=[
            review_json(
                {"line": 11, "severity": "major", "confidence": 0.9, "comment": "Linter knows."},
                {"line": 12, "severity": "major", "confidence": 0.9, "comment": "Novel issue."},
            )
        ],
    )
    config = ReviewConfig()
    eng = ReviewEngine(client, config=config, lint_runner=FakeLint())
    result = eng.review(parse_diff(DIFF), context_with_symbols(), repo_root=Path())
    (comment,) = result.comments
    assert comment.line == 12
    assert result.stats.dropped_lint_duplicate == 1


def test_cap_keeps_highest_severity_then_confidence() -> None:
    client = ScriptedClient(
        triage=[triage_yes(), triage_no()],
        review=[
            review_json(
                {"line": 10, "severity": "nit", "confidence": 0.95, "comment": "n"},
                {"line": 11, "severity": "critical", "confidence": 0.85, "comment": "c"},
                {"line": 12, "severity": "major", "confidence": 0.9, "comment": "m1"},
                {"line": 13, "severity": "major", "confidence": 0.7, "comment": "m2"},
            )
        ],
    )
    result = engine(client, max_comments=2).review(parse_diff(DIFF), context_with_symbols())
    assert result.stats.dropped_over_cap == 2
    assert [c.severity for c in result.comments] == [Severity.CRITICAL, Severity.MAJOR]
    assert [c.line for c in result.comments] == [11, 12]  # final order: by file, line


def test_unparseable_triage_counts_as_failure_and_skips_review() -> None:
    client = ScriptedClient(triage=["not json at all", triage_no()], review=[])
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    assert result.stats.triage_failures == 1
    assert client.prompts["review"] == []
    assert result.comments == ()


def test_code_fenced_json_is_parsed() -> None:
    fenced = "```json\n" + triage_yes() + "\n```"
    review_fenced = (
        "```json\n"
        + review_json({"line": 11, "severity": "major", "confidence": 0.9, "comment": "ok"})
        + "\n```"
    )
    client = ScriptedClient(triage=[fenced, triage_no()], review=[review_fenced])
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    assert len(result.comments) == 1


def test_malformed_items_in_valid_array_are_counted_not_silent() -> None:
    # The array parses, but two elements are unusable. In Phase 5 a silent
    # drop would look identical to "the agent found nothing".
    review = json.dumps(
        [
            {"line": 11, "severity": "major", "confidence": 0.9, "comment": "real finding"},
            "just a string, not an object",
            {"line": 12, "severity": "major", "confidence": 0.9},  # missing comment text
        ]
    )
    client = ScriptedClient(triage=[triage_yes(), triage_no()], review=[review])
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    (comment,) = result.comments
    assert comment.line == 11
    assert result.stats.dropped_malformed_item == 2
    assert result.stats.review_failures == 0  # the response itself was fine


def test_unknown_severity_is_coerced_to_minor() -> None:
    client = ScriptedClient(
        triage=[triage_yes(), triage_no()],
        review=[
            review_json(
                {"line": 11, "severity": "blocker", "confidence": 0.9, "comment": "kept anyway"}
            )
        ],
    )
    result = engine(client).review(parse_diff(DIFF), context_with_symbols())
    (comment,) = result.comments
    assert comment.severity is Severity.MINOR


def test_deleted_and_binary_files_are_not_triaged() -> None:
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
    )
    client = ScriptedClient(triage=[], review=[])
    result = engine(client).review(
        parse_diff(diff), PRContext(files=(), total_tokens=0, dropped_symbols=0)
    )
    assert result.stats.hunks_total == 0


def test_render_hunk_shows_anchorable_line_numbers() -> None:
    (file_diff,) = parse_diff(DIFF)[:1]
    rendered = render_hunk(file_diff.hunks[0])
    assert "   11 +     result = data['key']" in rendered
    assert "   10       data = load()" in rendered
    assert rendered.startswith("@@ -10,2 +10,4 @@ def handler():")
