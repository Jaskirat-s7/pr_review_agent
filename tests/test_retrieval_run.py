"""Tests for the retrieval eval runner."""

from __future__ import annotations

from pathlib import Path

from pr_review_agent.context.models import FileContext, PRContext, SymbolDef, SymbolKind
from pr_review_agent.evals.retrieval_metrics import RetrievalScore
from pr_review_agent.evals.retrieval_run import (
    GoldCase,
    GoldQuery,
    format_markdown,
    load_gold,
    ranked_ids,
    score_retriever,
)


def _symbol(module_path: str, name: str, *, ref: int) -> SymbolDef:
    return SymbolDef(
        module_path=module_path,
        name=name,
        kind=SymbolKind.FUNCTION,
        source="...",
        lineno=1,
        end_lineno=1,
        est_tokens=5,
        reference_count=ref,
    )


def test_ranked_ids_orders_by_reference_count() -> None:
    fc = FileContext(
        "a.py",
        (_symbol("x.py", "low", ref=1), _symbol("y.py", "high", ref=9)),
        (),
    )
    assert ranked_ids(fc) == ["y.py::high", "x.py::low"]
    assert ranked_ids(None) == []


def test_score_retriever_matches_gold() -> None:
    symbols = (_symbol("y.py", "high", ref=9), _symbol("x.py", "low", ref=1))
    context = PRContext(
        files=(FileContext("a.py", symbols, ()),),
        total_tokens=10,
        dropped_symbols=0,
    )
    case = GoldCase(
        repo="o/r",
        number=1,
        sha="abc",
        queries=(GoldQuery(file="a.py", relevant=frozenset({"y.py::high"})),),
    )
    score = score_retriever([(context, case)], recall_ks=(1,), ndcg_ks=(2,))
    assert score.queries == 1
    assert score.recall_at_k[1] == 1.0  # the one gold id is ranked first
    assert score.mrr == 1.0


def test_load_gold_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"repo": "o/r", "number": 7, "sha": "deadbeef", '
        '"queries": [{"file": "a.py", "relevant": ["x.py::f"]}]}\n\n',
        encoding="utf-8",
    )
    cases = load_gold(path)
    assert len(cases) == 1
    assert cases[0].number == 7
    assert cases[0].queries[0].relevant == frozenset({"x.py::f"})


def test_format_markdown_has_a_row_per_retriever() -> None:
    results = {
        "ast": RetrievalScore(queries=3, recall_at_k={5: 0.5}, mrr=0.4, ndcg_at_k={10: 0.6}),
        "rag": RetrievalScore(queries=3, recall_at_k={5: 0.7}, mrr=0.6, ndcg_at_k={10: 0.8}),
    }
    table = format_markdown(results, recall_ks=(5,), ndcg_ks=(10,))
    lines = table.splitlines()
    assert lines[0].startswith("| Retriever | Queries | Recall@5 | MRR | nDCG@10 |")
    assert "| ast | 3 | 0.500 | 0.400 | 0.600 |" in table
    assert "| rag | 3 | 0.700 | 0.600 | 0.800 |" in table
