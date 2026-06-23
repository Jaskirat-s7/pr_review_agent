"""Retrieval-quality metrics for the AST-vs-RAG benchmark.

Binary relevance: each query (one changed file) has a gold set of symbol ids
that *should* be retrieved. Metrics run over the ranked list of retrieved ids.
Queries with an empty gold set are skipped by the aggregator — they carry no
signal — so the per-query functions reject one rather than guess.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of gold ids found in the top ``k`` retrieved ids."""
    _require_relevant(relevant)
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant id (1-based), or 0 if none ranks."""
    _require_relevant(relevant)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised DCG at ``k`` under binary relevance (gain 1 per hit)."""
    _require_relevant(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked[:k])
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """Aggregated metrics across the scored queries."""

    queries: int  # queries actually scored (empty-gold ones excluded)
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


def score_queries(
    per_query: Sequence[tuple[Sequence[str], set[str]]],
    *,
    recall_ks: Sequence[int] = (5, 10),
    ndcg_ks: Sequence[int] = (10,),
) -> RetrievalScore:
    """Average each metric over queries; queries with no gold ids are skipped."""
    rows = [(ranked, relevant) for ranked, relevant in per_query if relevant]
    n = len(rows)
    if n == 0:
        return RetrievalScore(
            queries=0,
            recall_at_k=dict.fromkeys(recall_ks, 0.0),
            mrr=0.0,
            ndcg_at_k=dict.fromkeys(ndcg_ks, 0.0),
        )
    return RetrievalScore(
        queries=n,
        recall_at_k={
            k: sum(recall_at_k(r, g, k) for r, g in rows) / n for k in recall_ks
        },
        mrr=sum(reciprocal_rank(r, g) for r, g in rows) / n,
        ndcg_at_k={k: sum(ndcg_at_k(r, g, k) for r, g in rows) / n for k in ndcg_ks},
    )


def _require_relevant(relevant: set[str]) -> None:
    if not relevant:
        raise ValueError("metric is undefined for an empty relevant set")
