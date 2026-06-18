"""Tests for rank fusion."""

from __future__ import annotations

from pr_review_agent.rag.retriever import reciprocal_rank_fusion


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
