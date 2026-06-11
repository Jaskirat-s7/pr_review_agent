"""Tests for module-to-file resolution and symbol extraction."""

from __future__ import annotations

import ast
from pathlib import Path

from pr_review_agent.context.models import SymbolKind
from pr_review_agent.context.resolve import (
    extract_symbol,
    file_package,
    find_definition,
    module_to_file,
    package_roots,
)


def test_package_roots_include_src_when_present(tmp_path: Path) -> None:
    assert package_roots(tmp_path) == (tmp_path,)
    (tmp_path / "src").mkdir()
    assert package_roots(tmp_path) == (tmp_path, tmp_path / "src")


def test_module_to_file_prefers_module_then_package(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text("class Model: ...\n", encoding="utf-8")
    roots = (tmp_path,)
    assert module_to_file("pkg.models", roots) == pkg / "models.py"
    assert module_to_file("pkg", roots) == pkg / "__init__.py"
    assert module_to_file("missing", roots) is None


def test_module_to_file_in_src_layout(tmp_path: Path) -> None:
    lib = tmp_path / "src" / "lib"
    lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("", encoding="utf-8")
    (lib / "x.py").write_text("def f(): ...\n", encoding="utf-8")
    assert module_to_file("lib.x", package_roots(tmp_path)) == lib / "x.py"


def test_file_package_walks_init_chain(tmp_path: Path) -> None:
    sub = tmp_path / "pkg" / "sub"
    sub.mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (sub / "__init__.py").write_text("", encoding="utf-8")
    module = sub / "mod.py"
    module.write_text("", encoding="utf-8")
    assert file_package(module) == "pkg.sub"
    assert file_package(tmp_path / "top.py") == ""


def test_extract_function_with_decorator() -> None:
    source = "import functools\n\n\n@functools.cache\ndef slow(n: int) -> int:\n    return n\n"
    tree = ast.parse(source)
    node = find_definition(tree, "slow")
    assert node is not None
    symbol = extract_symbol(source, node, module_path="m.py", reference_count=2)
    assert symbol.kind is SymbolKind.FUNCTION
    assert symbol.source.startswith("@functools.cache\ndef slow")
    assert (symbol.lineno, symbol.end_lineno) == (4, 6)
    assert symbol.reference_count == 2
    assert symbol.est_tokens > 0


def test_extract_class_and_async_function() -> None:
    source = "class Model:\n    x = 1\n\n\nasync def fetch() -> None:\n    pass\n"
    tree = ast.parse(source)
    model = find_definition(tree, "Model")
    fetch = find_definition(tree, "fetch")
    assert model is not None and fetch is not None
    assert (
        extract_symbol(source, model, module_path="m.py", reference_count=1).kind
        is SymbolKind.CLASS
    )
    assert (
        extract_symbol(source, fetch, module_path="m.py", reference_count=1).kind
        is SymbolKind.ASYNC_FUNCTION
    )


def test_find_definition_is_top_level_only() -> None:
    tree = ast.parse("def outer():\n    def inner(): ...\n")
    assert find_definition(tree, "outer") is not None
    assert find_definition(tree, "inner") is None
