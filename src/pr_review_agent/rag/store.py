"""LanceDB-backed ChunkIndex: vector and BM25 search over a built index.

Kept apart from the pure retrieval logic so that module needs no LanceDB
import. The full-text query is reduced to identifier tokens, since raw diff
text carries punctuation the FTS parser would choke on.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pr_review_agent.rag.index import TABLE_NAME
from pr_review_agent.rag.models import Chunk, ChunkKind

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class LanceChunkIndex:
    """ChunkIndex over an open LanceDB table (see rag.retriever.ChunkIndex)."""

    def __init__(self, table: Any) -> None:
        self._table = table

    def vector_search(self, vector: Sequence[float], *, limit: int) -> list[Chunk]:
        rows = self._table.search(list(vector)).limit(limit).to_list()
        return [_row_to_chunk(row) for row in rows]

    def text_search(self, query: str, *, limit: int) -> list[Chunk]:
        terms = list(dict.fromkeys(_IDENTIFIER.findall(query)))
        if not terms:
            return []
        rows = self._table.search(" ".join(terms), query_type="fts").limit(limit).to_list()
        return [_row_to_chunk(row) for row in rows]


def open_index(dest: Path) -> LanceChunkIndex:
    """Open the LanceDB chunk table built at ``dest``."""
    import lancedb

    table = lancedb.connect(str(dest)).open_table(TABLE_NAME)
    return LanceChunkIndex(table)


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        path=row["path"],
        kind=ChunkKind(row["kind"]),
        name=row["name"],
        qualname=row["qualname"],
        source=row["source"],
        lineno=row["lineno"],
        end_lineno=row["end_lineno"],
        est_tokens=row["est_tokens"],
    )
