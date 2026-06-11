"""Tests for import collection and changed-line reference analysis."""

from __future__ import annotations

import ast

from pr_review_agent.context.imports import (
    ImportedName,
    collect_imports,
    referenced_dotted_names,
    resolve_usage,
)


def _imports(source: str, package: str = "") -> list[ImportedName]:
    return collect_imports(ast.parse(source), current_package=package)


def test_plain_and_aliased_imports() -> None:
    found = _imports("import json\nimport numpy as np\nimport os.path\n")
    assert found == [
        ImportedName("json", "json", None, False),
        ImportedName("np", "numpy", None, True),
        ImportedName("os", "os.path", None, False),
    ]


def test_from_imports_with_alias() -> None:
    found = _imports("from helpers import greet\nfrom pkg.models import Model as M\n")
    assert found == [
        ImportedName("greet", "helpers", "greet", False),
        ImportedName("M", "pkg.models", "Model", True),
    ]


def test_relative_imports_resolve_against_package() -> None:
    found = _imports(
        "from .utils import helper\nfrom ..core import thing\nfrom . import sibling\n",
        package="pkg.sub",
    )
    assert found == [
        ImportedName("helper", "pkg.sub.utils", "helper", False),
        ImportedName("thing", "pkg.core", "thing", False),
        ImportedName("sibling", "pkg.sub", "sibling", False),
    ]


def test_star_imports_and_escaping_relatives_are_skipped() -> None:
    assert _imports("from helpers import *\n") == []
    assert _imports("from ...beyond import x\n", package="pkg") == []


def test_function_local_imports_are_collected() -> None:
    found = _imports("def f():\n    from helpers import greet\n    return greet\n")
    assert found == [ImportedName("greet", "helpers", "greet", False)]


def test_referenced_names_only_count_changed_lines() -> None:
    tree = ast.parse("a = greet('x')\nb = Model()\nc = greet('y')\n")
    counts = referenced_dotted_names(tree, {1, 3})
    assert counts == {"greet": 2}


def test_attribute_chain_counted_once_at_full_length() -> None:
    tree = ast.parse("helpers.api.shout('x')\n")
    counts = referenced_dotted_names(tree, {1})
    assert counts == {"helpers.api.shout": 1}


def test_chain_behind_call_still_counts_inner_names() -> None:
    tree = ast.parse("get_client().fetch(payload)\n")
    counts = referenced_dotted_names(tree, {1})
    assert counts == {"get_client": 1, "payload": 1}


def test_resolve_usage_through_import_forms() -> None:
    imports = [
        ImportedName("greet", "helpers", "greet", False),
        ImportedName("np", "numpy", None, True),
        ImportedName("os", "os.path", None, False),
    ]
    assert resolve_usage("greet", imports) == ("helpers", "greet")
    assert resolve_usage("greet.value", imports) == ("helpers", "greet")
    assert resolve_usage("np.array", imports) == ("numpy", "array")
    assert resolve_usage("os.path.join", imports) == ("os.path", "join")
    assert resolve_usage("np", imports) is None  # bare module: nothing to extract
    assert resolve_usage("unrelated", imports) is None
