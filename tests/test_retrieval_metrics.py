"""Tests for retrieval metrics."""

from __future__ import annotations

import math

import pytest

from pr_review_agent.evals.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_queries,
)


def test_recall_at_k_counts_hits_in_top_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "d"}
    assert recall_at_k(ranked, relevant, 2) == 0.5  # only "a" in top 2
    assert recall_at_k(ranked, relevant, 4) == 1.0


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(["x", "y", "hit"], {"hit"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"hit"}) == 0.0


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["hit", "x", "y"], {"hit"}, 3)
    late = ndcg_at_k(["x", "y", "hit"], {"hit"}, 3)
    assert early == 1.0  # single relevant ranked first == ideal
    assert late == pytest.approx(1 / math.log2(4))
    assert late < early


def test_metrics_reject_empty_relevant_set() -> None:
    for call in (
        lambda: recall_at_k(["a"], set(), 1),
        lambda: reciprocal_rank(["a"], set()),
        lambda: ndcg_at_k(["a"], set(), 1),
    ):
        with pytest.raises(ValueError, match="empty relevant set"):
            call()


def test_score_queries_averages_and_skips_empty_gold() -> None:
    per_query = [
        (["a", "b"], {"a"}),  # recall@5=1, rr=1, ndcg=1
        (["x", "hit"], {"hit"}),  # recall@5=1, rr=0.5, ndcg=1/log2(3)
        (["z"], set()),  # skipped: no gold
    ]
    score = score_queries(per_query, recall_ks=(5,), ndcg_ks=(2,))
    assert score.queries == 2
    assert score.recall_at_k[5] == 1.0
    assert score.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert score.ndcg_at_k[2] == pytest.approx((1.0 + 1 / math.log2(3)) / 2)


def test_score_queries_handles_all_empty() -> None:
    score = score_queries([([], set())], recall_ks=(5,), ndcg_ks=(10,))
    assert score.queries == 0
    assert score.mrr == 0.0
    assert score.recall_at_k[5] == 0.0
