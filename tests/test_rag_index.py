"""Tests for index location, cache reuse, and the LanceDB build."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pr_review_agent.rag.index import (
    TABLE_NAME,
    index_dir,
    index_exists,
    index_repo,
)


class FakeEmbedder:
    """Deterministic, model-free embedder for tests."""

    dim = 8

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float((len(t) >> i) & 1) for i in range(self.dim)] for t in texts]


class ExplodingEmbedder:
    """Fails if asked to embed — used to prove a build was skipped."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise AssertionError("embed should not be called on a cache hit")


def test_index_dir_layout(tmp_path: Path) -> None:
    dest = index_dir(tmp_path, "octo/widgets", "abc123")
    assert dest == tmp_path / "octo" / "widgets" / "abc123"


def test_index_exists_checks_table_marker(tmp_path: Path) -> None:
    dest = tmp_path / "idx"
    assert not index_exists(dest)
    (dest / f"{TABLE_NAME}.lance").mkdir(parents=True)
    assert index_exists(dest)


def test_index_repo_reuses_cached_index(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    dest = index_dir(cache_root, "octo/widgets", "deadbeef")
    (dest / f"{TABLE_NAME}.lance").mkdir(parents=True)

    stats = index_repo(
        repo_root, "octo/widgets", "deadbeef", ExplodingEmbedder(), cache_root=cache_root
    )
    assert stats.reused
    assert stats.chunks == 0


def test_build_and_search_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lancedb")
    repo_root = tmp_path / "repo"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "svc.py").write_text(
        "def handle(request):\n    return request.payload\n", encoding="utf-8"
    )
    cache_root = tmp_path / "cache"
    embedder = FakeEmbedder()

    stats = index_repo(repo_root, "octo/widgets", "cafe", embedder, cache_root=cache_root)
    assert stats.reused is False
    assert stats.files == 1
    assert stats.chunks >= 1
    assert index_exists(stats.dest)

    import lancedb

    table = lancedb.connect(str(stats.dest)).open_table(TABLE_NAME)
    assert table.count_rows() == stats.chunks

    hits = table.search("payload", query_type="fts").limit(5).to_list()
    assert any("payload" in row["source"] for row in hits)
