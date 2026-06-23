"""Round-trip test for the LanceDB-backed ChunkIndex (needs the rag extra)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest


class _FakeEmbedder:
    dim = 8

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float((len(t) >> i) & 1) for i in range(self.dim)] for t in texts]


def test_lance_index_vector_and_text_search(tmp_path: Path) -> None:
    pytest.importorskip("lancedb")
    from pr_review_agent.rag.index import index_repo
    from pr_review_agent.rag.store import open_index

    repo_root = tmp_path / "repo"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "svc.py").write_text(
        "def handle_payment(request):\n    return request.payload\n", encoding="utf-8"
    )
    stats = index_repo(
        repo_root, "octo/widgets", "cafe", _FakeEmbedder(), cache_root=tmp_path / "cache"
    )

    index = open_index(stats.dest)

    text_hits = index.text_search("handle_payment", limit=5)
    assert any("handle_payment" in chunk.source for chunk in text_hits)

    vector_hits = index.vector_search(_FakeEmbedder().encode(["handle_payment"])[0], limit=5)
    assert vector_hits  # nearest-neighbour search returns the indexed chunk(s)

    # Punctuation-only queries reduce to no FTS terms rather than erroring.
    assert index.text_search("()=:", limit=5) == []
