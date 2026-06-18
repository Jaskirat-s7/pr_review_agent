"""Hybrid retrieval over the RAG index.

Only the rank-fusion step lands here for now; the vector + full-text search,
cross-encoder rerank, and token-budgeted assembly are wired in a later PR. The
fusion function is pure and dependency-free so it can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Sequence


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
