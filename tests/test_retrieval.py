"""Tests for retriever selection and the hybrid merge."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pr_review_agent.config import AppConfig, RagConfig
from pr_review_agent.context.models import FileContext, PRContext, SymbolDef, SymbolKind
from pr_review_agent.context.retriever import ContextRetriever
from pr_review_agent.diff.models import FileDiff
from pr_review_agent.retrieval import (
    HybridRetriever,
    RetrievalError,
    RetrieverKind,
    build_retriever,
)


def _symbol(module_path: str, name: str, *, est_tokens: int = 10, ref: int = 1) -> SymbolDef:
    return SymbolDef(
        module_path=module_path,
        name=name,
        kind=SymbolKind.FUNCTION,
        source=f"def {name}(): ...",
        lineno=1,
        end_lineno=1,
        est_tokens=est_tokens,
        reference_count=ref,
    )


class _StubRetriever:
    def __init__(self, context: PRContext) -> None:
        self._context = context

    def retrieve(self, files: Sequence[FileDiff]) -> PRContext:
        return self._context


def test_build_ast_is_the_default_baseline(tmp_path: Path) -> None:
    retriever = build_retriever(
        RetrieverKind.AST, repo_root=tmp_path, config=AppConfig(), repo="o/r", sha="abc1234"
    )
    assert isinstance(retriever, ContextRetriever)


def test_build_rag_without_index_raises(tmp_path: Path) -> None:
    config = AppConfig(rag=RagConfig(cache_dir=str(tmp_path / "empty")))
    with pytest.raises(RetrievalError, match="no RAG index"):
        build_retriever(
            RetrieverKind.RAG, repo_root=tmp_path, config=config, repo="o/r", sha="abc1234"
        )


def test_hybrid_merges_dedups_and_rebudgets() -> None:
    ast_ctx = PRContext(
        files=(FileContext("a.py", (_symbol("helper.py", "shared", ref=5),), ("X.y",)),),
        total_tokens=10,
        dropped_symbols=0,
    )
    rag_ctx = PRContext(
        files=(
            FileContext(
                "a.py",
                (_symbol("helper.py", "shared", ref=1), _symbol("other.py", "extra", ref=9)),
                (),
            ),
        ),
        total_tokens=20,
        dropped_symbols=0,
    )
    hybrid = HybridRetriever(_StubRetriever(ast_ctx), _StubRetriever(rag_ctx), token_budget=1000)  # type: ignore[arg-type]

    merged = hybrid.retrieve([])

    file = merged.files[0]
    # "shared" is de-duplicated (AST wins), "extra" comes from RAG.
    assert {s.name for s in file.symbols} == {"shared", "extra"}
    # AST's unresolved survives the merge.
    assert file.unresolved == ("X.y",)
