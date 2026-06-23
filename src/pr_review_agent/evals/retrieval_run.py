"""Run a retriever over gold cases and tabulate retrieval metrics.

A gold case pins, for one PR, the set of repo symbols that *should* be
retrieved for each changed file (``module_path::name`` ids). Given a built
retriever and the PR's parsed diff, we read the ranking it produces per file
and score it against the gold ids. The orchestration that fetches diffs and
builds indexes lives in the CLI; everything here is pure and testable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pr_review_agent.context.models import FileContext, PRContext, SymbolDef
from pr_review_agent.diff.models import FileDiff
from pr_review_agent.evals.retrieval_metrics import RetrievalScore, score_queries


@dataclass(frozen=True, slots=True)
class GoldQuery:
    """Gold relevance for one changed file."""

    file: str
    relevant: frozenset[str]  # symbol ids: "module_path::name"


@dataclass(frozen=True, slots=True)
class GoldCase:
    """Gold relevance for one PR (matched to its diff by number)."""

    repo: str
    number: int
    sha: str
    queries: tuple[GoldQuery, ...]


def symbol_id(symbol: SymbolDef) -> str:
    """Canonical id shared by AST and RAG outputs and by gold labels."""
    return f"{symbol.module_path}::{symbol.name}"


def ranked_ids(file_context: FileContext | None) -> list[str]:
    """The retriever's per-file ranking, most relevant first.

    ``reference_count`` is the rank signal for both retrievers (the RAG path
    stores the fusion rank there); ties break by size then id for determinism.
    """
    if file_context is None:
        return []
    ordered = sorted(
        file_context.symbols,
        key=lambda s: (-s.reference_count, s.est_tokens, s.module_path, s.name),
    )
    return [symbol_id(symbol) for symbol in ordered]


def collect_results(
    context: PRContext,
    queries: Sequence[GoldQuery],
) -> list[tuple[list[str], set[str]]]:
    """Pair each query's ranking with its gold set, ready for scoring."""
    by_file = {fc.file_path: fc for fc in context.files}
    return [(ranked_ids(by_file.get(q.file)), set(q.relevant)) for q in queries]


def score_retriever(
    cases: Sequence[tuple[PRContext, GoldCase]],
    *,
    recall_ks: Sequence[int] = (5, 10),
    ndcg_ks: Sequence[int] = (10,),
) -> RetrievalScore:
    """Score one retriever across all cases (its retrieved PRContext per case)."""
    per_query: list[tuple[list[str], set[str]]] = []
    for context, case in cases:
        per_query.extend(collect_results(context, case.queries))
    return score_queries(per_query, recall_ks=recall_ks, ndcg_ks=ndcg_ks)


def load_gold(path: Path) -> list[GoldCase]:
    """Load gold cases from a JSONL file."""
    cases: list[GoldCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        queries = tuple(
            GoldQuery(file=q["file"], relevant=frozenset(q["relevant"]))
            for q in raw["queries"]
        )
        cases.append(
            GoldCase(repo=raw["repo"], number=int(raw["number"]), sha=raw["sha"], queries=queries)
        )
    return cases


def gold_queries_for(case: GoldCase, files: Sequence[FileDiff]) -> tuple[GoldQuery, ...]:
    """Keep only gold queries whose file is actually present in the diff."""
    changed = {file_diff.path for file_diff in files}
    return tuple(q for q in case.queries if q.file in changed)


def format_markdown(
    results: Mapping[str, RetrievalScore],
    *,
    recall_ks: Sequence[int] = (5, 10),
    ndcg_ks: Sequence[int] = (10,),
) -> str:
    """Render a results table, one row per retriever."""
    recall_cols = [f"Recall@{k}" for k in recall_ks]
    ndcg_cols = [f"nDCG@{k}" for k in ndcg_ks]
    headers = ["Retriever", "Queries", *recall_cols, "MRR", *ndcg_cols]
    aligns = ["---", "---:", *["---:"] * (len(headers) - 2)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for name, score in results.items():
        cells = [
            name,
            str(score.queries),
            *[f"{score.recall_at_k[k]:.3f}" for k in recall_ks],
            f"{score.mrr:.3f}",
            *[f"{score.ndcg_at_k[k]:.3f}" for k in ndcg_ks],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
