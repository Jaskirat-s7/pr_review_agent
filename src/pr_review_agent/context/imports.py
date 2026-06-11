"""AST-based analysis of a changed file: its imports and which names its
changed lines reference."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Set as AbstractSet
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImportedName:
    """One name bound by an import statement.

    ``symbol`` is set for ``from m import x`` (the imported symbol) and
    ``None`` for plain ``import m`` / ``import m as alias``.
    """

    local_name: str
    module: str  # absolute dotted module path
    symbol: str | None
    aliased: bool


def collect_imports(tree: ast.Module, *, current_package: str) -> list[ImportedName]:
    """Collect all import bindings in the file, including function-local ones.

    Relative imports are resolved against ``current_package`` (the dotted
    package containing the file); unresolvable ones are skipped.
    """
    out: list[ImportedName] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.partition(".")[0]
                out.append(ImportedName(local, alias.name, None, alias.asname is not None))
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node.module, node.level, current_package)
            if module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                out.append(ImportedName(local, module, alias.name, alias.asname is not None))
    return out


def collect_star_imports(tree: ast.Module, *, current_package: str) -> list[str]:
    """Modules pulled in via ``from m import *``, as absolute dotted paths."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            module = _absolute_module(node.module, node.level, current_package)
            if module is not None:
                out.append(module)
    return out


def referenced_dotted_names(tree: ast.Module, changed_lines: AbstractSet[int]) -> Counter[str]:
    """Count dotted-name references (``x``, ``pkg.mod.f``) on changed lines.

    Attribute chains are counted once at their full length, never per link.
    """
    collector = _RefCollector(changed_lines)
    collector.visit(tree)
    return collector.counts


def resolve_usage(dotted: str, imports: list[ImportedName]) -> tuple[str, str] | None:
    """Map a dotted usage to ``(module, symbol)`` via the file's imports.

    Returns ``None`` when the usage does not go through an import or names a
    bare module with no attribute (nothing to extract).
    """
    for imp in imports:
        if imp.symbol is not None:
            if dotted == imp.local_name or dotted.startswith(imp.local_name + "."):
                return imp.module, imp.symbol
        else:
            # For unaliased `import x.y` the binding is `x` but attribute
            # access goes through the full dotted module path.
            prefix = imp.local_name if imp.aliased else imp.module
            if dotted.startswith(prefix + "."):
                remainder = dotted[len(prefix) + 1 :]
                return imp.module, remainder.partition(".")[0]
    return None


def _absolute_module(module: str | None, level: int, current_package: str) -> str | None:
    if level == 0:
        return module
    parts = current_package.split(".") if current_package else []
    if level - 1 > len(parts):
        return None  # relative import escapes the repo's package tree
    base = parts[: len(parts) - (level - 1)]
    if module:
        base = [*base, *module.split(".")]
    return ".".join(base) if base else None


class _RefCollector(ast.NodeVisitor):
    def __init__(self, changed_lines: AbstractSet[int]) -> None:
        self._lines = changed_lines
        self.counts: Counter[str] = Counter()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        dotted = _dotted_chain(node)
        if dotted is not None:
            if isinstance(node.ctx, ast.Load):  # skip assignment/deletion targets
                self._record(node, dotted)
            return  # the whole chain is consumed; don't count sub-chains
        self.generic_visit(node)  # e.g. `f(x).attr`: still visit `f(x)`

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):  # skip assignment/deletion targets
            self._record(node, node.id)

    def _record(self, node: ast.expr, dotted: str) -> None:
        end = node.end_lineno if node.end_lineno is not None else node.lineno
        if any(line in self._lines for line in range(node.lineno, end + 1)):
            self.counts[dotted] += 1


def _dotted_chain(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))
