"""Tests for rank fusion and RAG retrieval into a PRContext."""

from __future__ import annotations

from collections.abc import Sequence

from pr_review_agent.context.models import SymbolKind
from pr_review_agent.diff.models import DiffLine, FileDiff, FileStatus, Hunk, LineKind
from pr_review_agent.rag.models import Chunk, ChunkKind
from pr_review_agent.rag.retriever import (
    RagRetriever,
    hybrid_search,
    reciprocal_rank_fusion,
)


def test_doc_in_both_rankings_ranks_first() -> None:
    vector = ["a", "shared", "b"]
    full_text = ["c", "shared", "d"]

    fused = reciprocal_rank_fusion([vector, full_text])
    ids = [doc_id for doc_id, _ in fused]

    # "shared" appears in both lists, so its summed score beats any id that
    # appears in only one.
    assert ids[0] == "shared"
    assert set(ids) == {"a", "b", "c", "d", "shared"}


def test_empty_input_is_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def _chunk(
    path: str,
    qualname: str,
    *,
    kind: ChunkKind = ChunkKind.FUNCTION,
    est_tokens: int = 10,
) -> Chunk:
    return Chunk(
        id=f"{path}:{qualname}",
        path=path,
        kind=kind,
        name=qualname.split(".")[-1],
        qualname=qualname,
        source=f"def {qualname}():\n    ...\n",
        lineno=1,
        end_lineno=2,
        est_tokens=est_tokens,
    )


def _changed_file(path: str, content: str = "x = helper()") -> FileDiff:
    line = DiffLine(LineKind.ADDED, content, None, 1)
    hunk = Hunk(0, 0, 1, 1, "", (line,))
    return FileDiff(None, path, FileStatus.MODIFIED, False, (hunk,))


class _FakeIndex:
    def __init__(self, vector_hits: Sequence[Chunk], text_hits: Sequence[Chunk]) -> None:
        self._vector = list(vector_hits)
        self._text = list(text_hits)

    def vector_search(self, vector: Sequence[float], *, limit: int) -> list[Chunk]:
        return self._vector[:limit]

    def text_search(self, query: str, *, limit: int) -> list[Chunk]:
        return self._text[:limit]


class _FakeEmbedder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def test_hybrid_search_fuses_vector_and_text() -> None:
    shared = _chunk("pkg/h.py", "shared")
    index = _FakeIndex(
        vector_hits=[_chunk("pkg/a.py", "a"), shared],
        text_hits=[_chunk("pkg/b.py", "b"), shared],
    )
    ranked = hybrid_search(index, query_text="q", query_vector=[0.0, 1.0], limit=5)
    assert ranked[0].chunk.id == shared.id  # in both rankings → highest fused score


def test_rag_retriever_skips_same_file_and_module_chunks() -> None:
    good = _chunk("pkg/helper.py", "helper")
    klass = _chunk("pkg/b.py", "B", kind=ChunkKind.CLASS_SKELETON)
    index = _FakeIndex(
        vector_hits=[_chunk("pkg/a.py", "a"), good],  # a.py is the changed file
        text_hits=[_chunk("pkg/a.py", "mod", kind=ChunkKind.MODULE), klass, good],
    )
    retriever = RagRetriever(index, _FakeEmbedder(), token_budget=1000)

    context = retriever.retrieve([_changed_file("pkg/a.py")])

    assert len(context.files) == 1
    symbols = context.files[0].symbols
    assert {s.name for s in symbols} == {"helper", "B"}
    assert {s.kind for s in symbols} == {SymbolKind.FUNCTION, SymbolKind.CLASS}


def test_rag_retriever_respects_token_budget() -> None:
    top = _chunk("pkg/helper.py", "helper", est_tokens=10)
    extra = _chunk("pkg/b.py", "B", est_tokens=10)
    index = _FakeIndex(vector_hits=[top, extra], text_hits=[top])
    retriever = RagRetriever(index, _FakeEmbedder(), token_budget=10)

    context = retriever.retrieve([_changed_file("pkg/a.py")])

    assert context.total_tokens == 10
    assert context.dropped_symbols == 1
    assert [s.name for s in context.files[0].symbols] == ["helper"]
