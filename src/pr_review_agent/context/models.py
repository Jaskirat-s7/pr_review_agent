"""Structured results of context retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SymbolKind(StrEnum):
    """What kind of definition a retrieved symbol is."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async function"
    CLASS = "class"


@dataclass(frozen=True, slots=True)
class SymbolDef:
    """A symbol definition pulled from the repo as review context."""

    module_path: str  # repo-relative posix path of the defining file
    name: str
    kind: SymbolKind
    source: str
    lineno: int
    end_lineno: int
    est_tokens: int
    reference_count: int  # references from changed lines


@dataclass(frozen=True, slots=True)
class FileContext:
    """Retrieved context for one changed file."""

    file_path: str
    symbols: tuple[SymbolDef, ...]
    unresolved: tuple[str, ...]  # referenced but not resolvable inside the repo


@dataclass(frozen=True, slots=True)
class PRContext:
    """All retrieved context for a pull request, within the token budget."""

    files: tuple[FileContext, ...]
    total_tokens: int
    dropped_symbols: int  # resolved but cut by the budget
