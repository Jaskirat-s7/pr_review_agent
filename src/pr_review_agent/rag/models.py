"""Structured units of the RAG index."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ChunkKind(StrEnum):
    """What part of a file a chunk covers."""

    MODULE = "module"  # top-level code (imports, constants), no def bodies
    FUNCTION = "function"
    ASYNC_FUNCTION = "async function"
    CLASS_SKELETON = "class skeleton"  # class header + method signatures, no bodies
    METHOD = "method"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of source, carved out along AST boundaries."""

    id: str  # stable within a repo: "<path>:<qualname>"
    path: str  # repo-relative posix path
    kind: ChunkKind
    name: str  # the def/class name, or the file stem for a module chunk
    qualname: str  # dotted path within the file ("Cls.method", "func", "<module>")
    source: str
    lineno: int
    end_lineno: int
    est_tokens: int
