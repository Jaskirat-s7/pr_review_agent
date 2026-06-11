"""Resolving modules to repo files and extracting symbol definitions."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from pr_review_agent.context.models import SymbolDef, SymbolKind
from pr_review_agent.estimate import estimate_tokens

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def package_roots(repo_root: Path) -> tuple[Path, ...]:
    """Directories that can anchor absolute imports (repo root and src/)."""
    src = repo_root / "src"
    return (repo_root, src) if src.is_dir() else (repo_root,)


def module_to_file(module: str, roots: Sequence[Path]) -> Path | None:
    """Map a dotted module to a file under one of the roots, if it exists."""
    rel = Path(*module.split("."))
    for root in roots:
        module_file = (root / rel).with_suffix(".py")
        if module_file.is_file():
            return module_file
        init_file = root / rel / "__init__.py"
        if init_file.is_file():
            return init_file
    return None


def file_package(file_path: Path) -> str:
    """The dotted package containing a file ("" for a top-level module)."""
    parts: list[str] = []
    current = file_path.parent
    while (current / "__init__.py").is_file():
        parts.append(current.name)
        current = current.parent
    return ".".join(reversed(parts))


def find_definition(tree: ast.Module, name: str) -> Definition | None:
    """Find a top-level function/class definition by name."""
    for node in tree.body:
        if isinstance(node, Definition) and node.name == name:
            return node
    return None


def extract_symbol(
    source: str,
    node: Definition,
    *,
    module_path: str,
    reference_count: int,
) -> SymbolDef:
    """Cut a definition's full source (decorators included) into a SymbolDef."""
    start = min([node.lineno, *(dec.lineno for dec in node.decorator_list)])
    end = node.end_lineno if node.end_lineno is not None else node.lineno
    segment = "\n".join(source.splitlines()[start - 1 : end])
    if isinstance(node, ast.ClassDef):
        kind = SymbolKind.CLASS
    elif isinstance(node, ast.AsyncFunctionDef):
        kind = SymbolKind.ASYNC_FUNCTION
    else:
        kind = SymbolKind.FUNCTION
    return SymbolDef(
        module_path=module_path,
        name=node.name,
        kind=kind,
        source=segment,
        lineno=start,
        end_lineno=end,
        est_tokens=estimate_tokens(segment),
        reference_count=reference_count,
    )
