"""Hybrid retrieval over the RAG index.

Vector and full-text rankings are fused with Reciprocal Rank Fusion, then the
top chunks are mapped to the same ``SymbolDef``/``PRContext`` shapes the AST
resolver produces — so the review engine consumes either retriever unchanged.

The index is reached through the ``ChunkIndex`` protocol, so this module needs
no LanceDB import and is unit-tested against an in-memory fake. The concrete
LanceDB adapter lives next to the index build.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pr_review_agent.context.models import FileContext, PRContext, SymbolDef, SymbolKind
from pr_review_agent.diff.models import FileDiff, FileStatus, LineKind
from pr_review_agent.rag.index import Embedder
from pr_review_agent.rag.models import Chunk, ChunkKind


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked id lists by Reciprocal Rank Fusion.

    Each ranking contributes ``1 / (k + rank)`` per id (rank 0-based). Ids
    appearing in several rankings sum their contributions, so an id ranked by
    both vector and full-text search outranks one ranked by a single retriever.
    Returns ``(id, score)`` sorted by descending score, ties broken by id.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True, slots=True)
class RankedChunk:
    """A chunk and its fused retrieval score."""

    chunk: Chunk
    score: float


class ChunkIndex(Protocol):
    """A searchable index of chunks (see the LanceDB adapter)."""

    def vector_search(self, vector: Sequence[float], *, limit: int) -> list[Chunk]: ...

    def text_search(self, query: str, *, limit: int) -> list[Chunk]: ...


def hybrid_search(
    index: ChunkIndex,
    *,
    query_text: str,
    query_vector: Sequence[float],
    limit: int,
    k: int = 60,
) -> list[RankedChunk]:
    """Run vector + full-text search and fuse the two rankings with RRF."""
    vector_hits = index.vector_search(query_vector, limit=limit)
    text_hits = index.text_search(query_text, limit=limit)
    by_id = {chunk.id: chunk for chunk in (*vector_hits, *text_hits)}
    fused = reciprocal_rank_fusion(
        [[chunk.id for chunk in vector_hits], [chunk.id for chunk in text_hits]], k=k
    )
    return [RankedChunk(by_id[doc_id], score) for doc_id, score in fused[:limit]]


_KIND_MAP = {
    ChunkKind.FUNCTION: SymbolKind.FUNCTION,
    ChunkKind.ASYNC_FUNCTION: SymbolKind.ASYNC_FUNCTION,
    ChunkKind.METHOD: SymbolKind.FUNCTION,
    ChunkKind.CLASS_SKELETON: SymbolKind.CLASS,
    # MODULE chunks (top-level imports/constants) carry no single definition and
    # are skipped — they have no SymbolKind and read poorly as "context".
}


class RagRetriever:
    """Retrieves review context for changed files via hybrid vector+BM25 search.

    Produces a :class:`PRContext` interchangeable with the AST resolver's, so
    the engine and prompts are untouched. Index and embedder are injected.
    """

    def __init__(
        self,
        index: ChunkIndex,
        embedder: Embedder,
        *,
        token_budget: int,
        per_file_limit: int = 12,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._budget = token_budget
        self._limit = per_file_limit

    def retrieve(self, files: Sequence[FileDiff]) -> PRContext:
        per_file: list[tuple[str, list[SymbolDef]]] = []
        seen: set[tuple[str, str]] = set()
        for file_diff in files:
            if (
                file_diff.status is FileStatus.DELETED
                or file_diff.is_binary
                or not file_diff.path.endswith(".py")
            ):
                continue
            per_file.append((file_diff.path, self._file_symbols(file_diff, seen)))

        admitted, total, dropped = _apply_budget(per_file, self._budget)
        file_contexts = tuple(
            FileContext(path, tuple(admitted[path]), ()) for path, _ in per_file
        )
        return PRContext(files=file_contexts, total_tokens=total, dropped_symbols=dropped)

    def _file_symbols(self, file_diff: FileDiff, seen: set[tuple[str, str]]) -> list[SymbolDef]:
        query = _changed_text(file_diff)
        if not query:
            return []
        vector = self._embedder.encode([query])[0]
        ranked = hybrid_search(
            self._index, query_text=query, query_vector=vector, limit=self._limit
        )
        symbols: list[SymbolDef] = []
        for rank, hit in enumerate(ranked):
            chunk = hit.chunk
            kind = _KIND_MAP.get(chunk.kind)
            if kind is None or chunk.path == file_diff.path:
                continue  # skip module chunks and the changed file itself
            key = (chunk.path, chunk.qualname)
            if key in seen:
                continue
            seen.add(key)
            symbols.append(
                SymbolDef(
                    module_path=chunk.path,
                    name=chunk.qualname,
                    kind=kind,
                    source=chunk.source,
                    lineno=chunk.lineno,
                    end_lineno=chunk.end_lineno,
                    est_tokens=chunk.est_tokens,
                    # Fusion rank stands in for AST reference_count: higher-ranked
                    # hits survive the shared token budget first.
                    reference_count=self._limit - rank,
                )
            )
        return symbols


def _changed_text(file_diff: FileDiff) -> str:
    """The added lines of a diff, joined — the query for retrieval."""
    return "\n".join(
        line.content
        for hunk in file_diff.hunks
        for line in hunk.lines
        if line.kind is LineKind.ADDED
    ).strip()


def _apply_budget(
    per_file: list[tuple[str, list[SymbolDef]]],
    budget: int,
) -> tuple[dict[str, list[SymbolDef]], int, int]:
    """Keep the highest-ranked (then smallest) symbols within the token budget."""
    flat = [(path, symbol) for path, symbols in per_file for symbol in symbols]
    ranked = sorted(
        flat,
        key=lambda item: (
            -item[1].reference_count,
            item[1].est_tokens,
            item[1].module_path,
            item[1].name,
        ),
    )
    admitted_keys: set[tuple[str, str, str]] = set()
    total = 0
    dropped = 0
    for path, symbol in ranked:
        if total + symbol.est_tokens <= budget:
            admitted_keys.add((path, symbol.module_path, symbol.name))
            total += symbol.est_tokens
        else:
            dropped += 1
    admitted = {
        path: [s for s in symbols if (path, s.module_path, s.name) in admitted_keys]
        for path, symbols in per_file
    }
    return admitted, total, dropped
