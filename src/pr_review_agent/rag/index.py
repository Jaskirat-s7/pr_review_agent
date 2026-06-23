"""Building and locating the per-commit LanceDB index.

An index is a LanceDB table of chunks carrying both a dense embedding (vector
search) and a full-text index on the source (BM25). It is cached on disk keyed
by repo and commit SHA, so re-indexing the same commit is a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pr_review_agent.rag.chunker import chunk_source
from pr_review_agent.rag.models import Chunk
from pr_review_agent.rag.walk import walk_python_files

TABLE_NAME = "chunks"


class Embedder(Protocol):
    """Anything that turns text into dense vectors (see CodeEmbedder)."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Outcome of an index build."""

    dest: Path
    files: int
    chunks: int
    reused: bool  # True when an existing index was found and left untouched


def index_dir(cache_root: Path, repo: str, sha: str) -> Path:
    """Cache location for one repo at one commit: ``<cache_root>/<repo>/<sha>``."""
    return cache_root / repo / sha


def index_exists(dest: Path) -> bool:
    """Whether a built index already lives at ``dest`` (cheap, no lancedb import)."""
    return (dest / f"{TABLE_NAME}.lance").exists()


def build_index(chunks: Sequence[Chunk], embedder: Embedder, dest: Path) -> int:
    """Embed ``chunks`` and write the LanceDB table at ``dest``; return row count.

    Overwrites any existing table at ``dest``. Returns 0 without creating a
    table when there are no chunks.
    """
    if not chunks:
        return 0
    import lancedb

    vectors = embedder.encode([chunk.source for chunk in chunks])
    rows = [
        {
            "id": chunk.id,
            "path": chunk.path,
            "kind": chunk.kind.value,
            "name": chunk.name,
            "qualname": chunk.qualname,
            "source": chunk.source,
            "lineno": chunk.lineno,
            "end_lineno": chunk.end_lineno,
            "est_tokens": chunk.est_tokens,
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    dest.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(dest))
    table = db.create_table(TABLE_NAME, data=rows, mode="overwrite")
    table.create_fts_index("source", replace=True)
    return len(rows)


def index_repo(
    root: Path,
    repo: str,
    sha: str,
    embedder: Embedder,
    *,
    cache_root: Path,
    force: bool = False,
) -> IndexStats:
    """Walk, chunk, embed, and index ``root`` (a checkout of ``repo`` at ``sha``).

    A cached index for the same repo+SHA is reused unless ``force`` is set.
    """
    dest = index_dir(cache_root, repo, sha)
    if index_exists(dest) and not force:
        return IndexStats(dest=dest, files=0, chunks=0, reused=True)

    chunks: list[Chunk] = []
    files = 0
    for path in walk_python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        file_chunks = chunk_source(path.relative_to(root).as_posix(), source)
        if file_chunks:
            files += 1
            chunks.extend(file_chunks)

    count = build_index(chunks, embedder, dest)
    return IndexStats(dest=dest, files=files, chunks=count, reused=False)
