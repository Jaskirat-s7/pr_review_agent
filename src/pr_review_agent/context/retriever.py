"""Token-budgeted retrieval of symbol definitions referenced by changed code.

For each changed Python file: parse it, find which imported names its added
lines reference, resolve those imports to files inside the repo, and extract
just the referenced function/class definitions. A global token budget then
keeps the most-referenced symbols and drops the rest.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from pathlib import Path

from pr_review_agent.context.imports import (
    collect_imports,
    collect_star_imports,
    referenced_dotted_names,
    resolve_usage,
)
from pr_review_agent.context.models import FileContext, PRContext, SymbolDef
from pr_review_agent.context.resolve import (
    Definition,
    extract_symbol,
    file_package,
    find_definition,
    module_to_file,
    package_roots,
)
from pr_review_agent.diff.models import FileDiff, FileStatus, LineKind


class ContextRetriever:
    """Retrieves review context from a checked-out PR head."""

    def __init__(self, repo_root: Path, *, token_budget: int) -> None:
        self._root = repo_root
        self._budget = token_budget
        self._cache: dict[Path, tuple[str, ast.Module] | None] = {}

    def retrieve(self, files: Sequence[FileDiff]) -> PRContext:
        roots = package_roots(self._root)
        per_file: list[tuple[str, list[SymbolDef], list[str]]] = []
        seen: set[tuple[str, str]] = set()
        for file_diff in files:
            if (
                file_diff.status is FileStatus.DELETED
                or file_diff.is_binary
                or not file_diff.path.endswith(".py")
            ):
                continue
            symbols, unresolved = self._file_candidates(file_diff, roots, seen)
            per_file.append((file_diff.path, symbols, unresolved))
        admitted, total, dropped = _apply_budget(per_file, self._budget)
        file_contexts = tuple(
            FileContext(path, tuple(admitted[path]), tuple(unresolved))
            for path, _, unresolved in per_file
        )
        return PRContext(files=file_contexts, total_tokens=total, dropped_symbols=dropped)

    def _file_candidates(
        self,
        file_diff: FileDiff,
        roots: tuple[Path, ...],
        seen: set[tuple[str, str]],
    ) -> tuple[list[SymbolDef], list[str]]:
        abs_path = self._root / file_diff.path
        parsed = self._parse(abs_path)
        if parsed is None:
            return [], [f"{file_diff.path} (unreadable or invalid Python)"]
        _, tree = parsed
        changed_lines = {
            line.new_lineno
            for hunk in file_diff.hunks
            for line in hunk.lines
            if line.kind is LineKind.ADDED and line.new_lineno is not None
        }
        if not changed_lines:
            return [], []

        imports = collect_imports(tree, current_package=file_package(abs_path))
        targets: dict[tuple[str, str], int] = {}
        for dotted, count in referenced_dotted_names(tree, changed_lines).items():
            resolved = resolve_usage(dotted, imports)
            if resolved is not None:
                targets[resolved] = targets.get(resolved, 0) + count

        symbols: list[SymbolDef] = []
        unresolved: list[str] = []
        for (module, symbol), count in sorted(targets.items()):
            if module.partition(".")[0] in sys.stdlib_module_names:
                continue
            located = self._locate_definition(module, symbol, roots)
            if located is None:
                unresolved.append(f"{module}.{symbol}")
                continue
            target_file, target_source, node = located
            if target_file == abs_path:
                continue  # defined in the changed file itself; the diff shows it
            module_path = target_file.relative_to(self._root).as_posix()
            if (module_path, node.name) in seen:
                continue  # already retrieved for an earlier changed file
            seen.add((module_path, node.name))
            symbols.append(
                extract_symbol(target_source, node, module_path=module_path, reference_count=count)
            )
        return symbols, unresolved

    def _locate_definition(
        self,
        module: str,
        symbol: str,
        roots: tuple[Path, ...],
        *,
        follow_reexport: bool = True,
    ) -> tuple[Path, str, Definition] | None:
        """Find where ``module.symbol`` is actually defined.

        Follows at most one re-export hop (``pkg/__init__.py`` doing
        ``from .impl import X``), which covers the ubiquitous public-API
        pattern without risking import cycles.
        """
        target_file = module_to_file(module, roots)
        if target_file is None:
            return None
        parsed = self._parse(target_file)
        if parsed is None:
            return None
        source, tree = parsed
        node = find_definition(tree, symbol)
        if node is not None:
            return target_file, source, node
        if not follow_reexport:
            return None
        package = file_package(target_file)
        reexports = collect_imports(tree, current_package=package)
        hop = resolve_usage(symbol, reexports)
        if hop is not None and hop[0].partition(".")[0] not in sys.stdlib_module_names:
            return self._locate_definition(hop[0], hop[1], roots, follow_reexport=False)
        for star_module in collect_star_imports(tree, current_package=package):
            if star_module.partition(".")[0] in sys.stdlib_module_names:
                continue
            located = self._locate_definition(star_module, symbol, roots, follow_reexport=False)
            if located is not None:
                return located
        return None

    def _parse(self, path: Path) -> tuple[str, ast.Module] | None:
        if path not in self._cache:
            try:
                source = path.read_text(encoding="utf-8")
                self._cache[path] = (source, ast.parse(source))
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
                self._cache[path] = None
        return self._cache[path]


def _apply_budget(
    per_file: list[tuple[str, list[SymbolDef], list[str]]],
    budget: int,
) -> tuple[dict[str, list[SymbolDef]], int, int]:
    """Keep the most-referenced (then smallest) symbols within the budget."""
    flat = [(path, symbol) for path, symbols, _ in per_file for symbol in symbols]
    ranked = sorted(
        flat,
        key=lambda item: (
            -item[1].reference_count,
            item[1].est_tokens,
            item[1].module_path,
            item[1].name,
        ),
    )
    admitted_keys: set[tuple[str, str, str]] = set()
    total = 0
    dropped = 0
    for path, symbol in ranked:
        if total + symbol.est_tokens <= budget:
            admitted_keys.add((path, symbol.module_path, symbol.name))
            total += symbol.est_tokens
        else:
            dropped += 1
    admitted = {
        path: [s for s in symbols if (path, s.module_path, s.name) in admitted_keys]
        for path, symbols, _ in per_file
    }
    return admitted, total, dropped
