"""Selecting and constructing the retrieval backend.

The AST resolver (``context.ContextRetriever``) is the default baseline. The
RAG path (``rag.RagRetriever`` over a prebuilt LanceDB index) and a hybrid that
merges both produce the same ``PRContext``, so the review engine is agnostic to
the choice. RAG/hybrid need the ``rag`` extra and a built index; the heavy
imports stay lazy so the AST path costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pr_review_agent.config import AppConfig
from pr_review_agent.context.models import FileContext, PRContext, SymbolDef
from pr_review_agent.context.retriever import ContextRetriever
from pr_review_agent.diff.models import FileDiff
from pr_review_agent.rag.retriever import RagRetriever, apply_token_budget


class RetrieverKind(StrEnum):
    """Which retrieval backend supplies review context."""

    AST = "ast"
    RAG = "rag"
    HYBRID = "hybrid"


class RetrievalError(Exception):
    """A retriever could not be built (e.g. no RAG index for the commit)."""


class Retriever(Protocol):
    """Anything that turns changed files into a PRContext."""

    def retrieve(self, files: Sequence[FileDiff]) -> PRContext: ...


def build_retriever(
    kind: RetrieverKind,
    *,
    repo_root: Path,
    config: AppConfig,
    repo: str,
    sha: str,
) -> Retriever:
    """Construct the retriever for ``kind`` against a checkout at ``repo_root``.

    Raises :class:`RetrievalError` if a RAG index is needed but missing, or
    ``ImportError`` if the ``rag`` extra is not installed.
    """
    budget = config.context.token_budget
    ast = ContextRetriever(repo_root, token_budget=budget)
    if kind is RetrieverKind.AST:
        return ast
    rag = _build_rag(config, repo, sha, budget)
    if kind is RetrieverKind.RAG:
        return rag
    return HybridRetriever(ast, rag, token_budget=budget)


def _build_rag(config: AppConfig, repo: str, sha: str, budget: int) -> RagRetriever:
    from pr_review_agent.rag.embeddings import CodeEmbedder
    from pr_review_agent.rag.index import index_dir, index_exists
    from pr_review_agent.rag.store import open_index

    dest = index_dir(Path(config.rag.cache_dir), repo, sha)
    if not index_exists(dest):
        raise RetrievalError(
            f"no RAG index for {repo}@{sha[:8]}; build it first: "
            f"pra index {repo} --ref {sha}"
        )
    embedder = CodeEmbedder(config.rag.embedding_model, device=config.rag.device)
    return RagRetriever(open_index(dest), embedder, token_budget=budget)


class HybridRetriever:
    """Union of the AST and RAG retrievers, re-budgeted as one context.

    Per file, symbols from both are merged and de-duplicated by
    ``(module_path, name)`` — AST wins ties — then the shared token budget
    keeps the highest-ranked across the merged set.
    """

    def __init__(
        self, ast: ContextRetriever, rag: RagRetriever, *, token_budget: int
    ) -> None:
        self._ast = ast
        self._rag = rag
        self._budget = token_budget

    def retrieve(self, files: Sequence[FileDiff]) -> PRContext:
        ast_by = {fc.file_path: fc for fc in self._ast.retrieve(files).files}
        rag_by = {fc.file_path: fc for fc in self._rag.retrieve(files).files}
        per_file: list[tuple[str, list[SymbolDef]]] = []
        for path in dict.fromkeys([*ast_by, *rag_by]):
            merged: dict[tuple[str, str], SymbolDef] = {}
            for source in (ast_by.get(path), rag_by.get(path)):
                if source is None:
                    continue
                for symbol in source.symbols:
                    merged.setdefault((symbol.module_path, symbol.name), symbol)
            per_file.append((path, list(merged.values())))

        admitted, total, dropped = apply_token_budget(per_file, self._budget)
        # AST carries the real unresolved list (RAG's is always empty); list it
        # last so it wins the merge.
        unresolved_by = {
            fc.file_path: fc.unresolved for fc in (*rag_by.values(), *ast_by.values())
        }
        file_contexts = tuple(
            FileContext(path, tuple(admitted[path]), unresolved_by.get(path, ()))
            for path, _ in per_file
        )
        return PRContext(files=file_contexts, total_tokens=total, dropped_symbols=dropped)
