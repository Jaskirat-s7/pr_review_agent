"""Tests for AST-aware chunking."""

from __future__ import annotations

from pr_review_agent.rag.chunker import chunk_source
from pr_review_agent.rag.models import ChunkKind

SOURCE = '''\
import os

CONST = 1


def top_level(x):
    return x + 1


class Service:
    """A service."""

    attr = 3

    def handle(self, request):
        secret_body_token = request.payload
        return secret_body_token
'''


def test_class_skeleton_excludes_method_bodies() -> None:
    chunks = {c.kind: c for c in chunk_source("pkg/svc.py", SOURCE)}

    skeleton = chunks[ChunkKind.CLASS_SKELETON]
    method = chunks[ChunkKind.METHOD]

    # The skeleton keeps the class header, class-level attrs, and the method
    # signature, but drops the body.
    assert "class Service:" in skeleton.source
    assert "attr = 3" in skeleton.source
    assert "def handle(self, request):" in skeleton.source
    assert "..." in skeleton.source
    assert "secret_body_token" not in skeleton.source

    # The body lives in the method chunk instead.
    assert "secret_body_token = request.payload" in method.source
    assert method.qualname == "Service.handle"


def test_top_level_and_module_chunks() -> None:
    chunks = chunk_source("pkg/svc.py", SOURCE)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.FUNCTION in kinds
    assert ChunkKind.MODULE in kinds

    module = next(c for c in chunks if c.kind is ChunkKind.MODULE)
    assert "import os" in module.source
    assert "CONST = 1" in module.source
    assert "def top_level" not in module.source


def test_invalid_python_yields_no_chunks() -> None:
    assert chunk_source("bad.py", "def oops(:\n") == []
