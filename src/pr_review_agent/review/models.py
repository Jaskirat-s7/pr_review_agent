"""Review engine result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    NIT = "nit"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ("nit", "minor", "major", "critical").index(self.value)


@dataclass(frozen=True, slots=True)
class AgentComment:
    """A generated review comment, anchored to a new-file line."""

    file_path: str
    line: int
    severity: Severity
    confidence: float
    body: str
    category: str
    has_context: bool  # whether retrieved context was available (decision #3)


@dataclass(frozen=True, slots=True)
class ReviewStats:
    """What happened at each stage of the pipeline, for reporting."""

    hunks_total: int = 0
    hunks_flagged: int = 0
    drafts_generated: int = 0
    dropped_low_confidence: int = 0
    dropped_invalid_line: int = 0
    dropped_malformed_item: int = 0  # valid JSON array, unusable element
    dropped_lint_duplicate: int = 0
    dropped_over_cap: int = 0
    triage_failures: int = 0  # whole triage response unparseable / call failed
    review_failures: int = 0  # whole review response unparseable / call failed


@dataclass(frozen=True, slots=True)
class ReviewResult:
    comments: tuple[AgentComment, ...]
    stats: ReviewStats
