"""AST-aware chunking of a Python file into retrievable units.

A file becomes: one chunk per top-level function, one chunk per method, one
"skeleton" chunk per class (header plus method signatures, method bodies
elided), and one "module" chunk for the remaining top-level code (imports,
constants). Chunking on definition boundaries keeps each chunk self-contained
and stops a method's body from being split mid-statement.
"""

from __future__ import annotations

import ast

from pr_review_agent.estimate import estimate_tokens
from pr_review_agent.rag.models import Chunk, ChunkKind

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef
_ELLIPSIS = "..."


def chunk_source(path: str, source: str) -> list[Chunk]:
    """Carve ``source`` into chunks tagged with ``path`` (repo-relative posix).

    Returns an empty list when the source is not valid Python.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()

    chunks: list[Chunk] = []
    module_stmts: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, _FuncDef):
            chunks.append(_def_chunk(path, lines, node, qualname=node.name))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_class_chunks(path, lines, node))
        else:
            module_stmts.append(node)

    module = _module_chunk(path, lines, module_stmts)
    if module is not None:
        chunks.insert(0, module)
    return chunks


def _class_chunks(path: str, lines: list[str], node: ast.ClassDef) -> list[Chunk]:
    chunks: list[Chunk] = [_skeleton_chunk(path, lines, node)]
    for stmt in node.body:
        if isinstance(stmt, _FuncDef):
            chunks.append(
                _def_chunk(path, lines, stmt, qualname=f"{node.name}.{stmt.name}", method=True)
            )
    return chunks


def _skeleton_chunk(path: str, lines: list[str], node: ast.ClassDef) -> Chunk:
    start = _def_start(node)
    end = node.end_lineno or node.lineno
    header_end = node.body[0].lineno - 1 if node.body else end
    out = lines[start - 1 : header_end]
    for stmt in node.body:
        if isinstance(stmt, _FuncDef):
            sig_start = _def_start(stmt)
            sig_end = stmt.body[0].lineno - 1
            out.extend(lines[sig_start - 1 : sig_end])
            out.append(" " * (stmt.col_offset + 4) + _ELLIPSIS)
        else:
            out.extend(lines[stmt.lineno - 1 : (stmt.end_lineno or stmt.lineno)])
    segment = "\n".join(out)
    return Chunk(
        id=f"{path}:{node.name}",
        path=path,
        kind=ChunkKind.CLASS_SKELETON,
        name=node.name,
        qualname=node.name,
        source=segment,
        lineno=start,
        end_lineno=end,
        est_tokens=estimate_tokens(segment),
    )


def _def_chunk(
    path: str,
    lines: list[str],
    node: _FuncDef,
    *,
    qualname: str,
    method: bool = False,
) -> Chunk:
    start = _def_start(node)
    end = node.end_lineno or node.lineno
    segment = "\n".join(lines[start - 1 : end])
    if method:
        kind = ChunkKind.METHOD
    elif isinstance(node, ast.AsyncFunctionDef):
        kind = ChunkKind.ASYNC_FUNCTION
    else:
        kind = ChunkKind.FUNCTION
    return Chunk(
        id=f"{path}:{qualname}",
        path=path,
        kind=kind,
        name=node.name,
        qualname=qualname,
        source=segment,
        lineno=start,
        end_lineno=end,
        est_tokens=estimate_tokens(segment),
    )


def _module_chunk(path: str, lines: list[str], stmts: list[ast.stmt]) -> Chunk | None:
    if not stmts:
        return None
    spans = [(s.lineno, s.end_lineno or s.lineno) for s in stmts]
    out = [line for start, end in spans for line in lines[start - 1 : end]]
    segment = "\n".join(out)
    return Chunk(
        id=f"{path}:<module>",
        path=path,
        kind=ChunkKind.MODULE,
        name=path.rsplit("/", 1)[-1].removesuffix(".py"),
        qualname="<module>",
        source=segment,
        lineno=spans[0][0],
        end_lineno=spans[-1][1],
        est_tokens=estimate_tokens(segment),
    )


def _def_start(node: _FuncDef | ast.ClassDef) -> int:
    """First source line of a definition, decorators included."""
    return min([node.lineno, *(dec.lineno for dec in node.decorator_list)])
