"""End-to-end tests for token-budgeted context retrieval on a fake repo."""

from __future__ import annotations

from pathlib import Path

import pytest

from pr_review_agent.context.models import PRContext, SymbolKind
from pr_review_agent.context.retriever import ContextRetriever
from pr_review_agent.diff.parser import parse_diff

APP_SOURCE = """\
import json

import requests

import helpers
from helpers import greet
from pkg.models import Model


def run() -> None:
    greet("one")
    greet("two")
    helpers.shout("x")
    model = Model()
    print(model, requests.get, json.dumps({}))
"""

HELPERS_SOURCE = """\
def greet(name: str) -> str:
    return f"hi {name}"


def shout(text: str) -> str:
    return text.upper()


def unused() -> None:
    return None
"""

MODELS_SOURCE = """\
class Model:
    \"\"\"A model.\"\"\"

    def __init__(self) -> None:
        self.x = 1
"""


def new_file_diff(path: str, content: str) -> str:
    lines = content.splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}\n"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (tmp_path / "helpers.py").write_text(HELPERS_SOURCE, encoding="utf-8")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "models.py").write_text(MODELS_SOURCE, encoding="utf-8")
    return tmp_path


def retrieve(repo: Path, diff_text: str, *, budget: int = 100_000) -> PRContext:
    retriever = ContextRetriever(repo, token_budget=budget)
    return retriever.retrieve(parse_diff(diff_text))


def test_retrieves_referenced_symbols_only(repo: Path) -> None:
    result = retrieve(repo, new_file_diff("app.py", APP_SOURCE))
    (file_context,) = result.files
    assert file_context.file_path == "app.py"
    by_name = {symbol.name: symbol for symbol in file_context.symbols}
    assert set(by_name) == {"greet", "shout", "Model"}  # "unused" not pulled

    greet = by_name["greet"]
    assert greet.module_path == "helpers.py"
    assert greet.kind is SymbolKind.FUNCTION
    assert greet.reference_count == 2
    assert greet.source.startswith("def greet(name: str)")

    model = by_name["Model"]
    assert model.module_path == "pkg/models.py"
    assert model.kind is SymbolKind.CLASS
    assert model.source.startswith("class Model:")

    assert result.total_tokens == sum(symbol.est_tokens for symbol in file_context.symbols)
    assert result.dropped_symbols == 0


def test_stdlib_skipped_and_third_party_reported_unresolved(repo: Path) -> None:
    result = retrieve(repo, new_file_diff("app.py", APP_SOURCE))
    (file_context,) = result.files
    assert file_context.unresolved == ("requests.get",)  # json (stdlib) is silent


def test_only_changed_lines_drive_retrieval(repo: Path) -> None:
    diff = (
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -11 +11 @@ def run() -> None:\n"
        '-    greet("zero")\n'
        '+    greet("one")\n'
    )
    result = retrieve(repo, diff)
    (file_context,) = result.files
    assert [symbol.name for symbol in file_context.symbols] == ["greet"]
    assert file_context.unresolved == ()


def test_budget_keeps_most_referenced_and_counts_dropped(repo: Path) -> None:
    full = retrieve(repo, new_file_diff("app.py", APP_SOURCE))
    greet_tokens = next(
        symbol.est_tokens for symbol in full.files[0].symbols if symbol.name == "greet"
    )
    result = retrieve(repo, new_file_diff("app.py", APP_SOURCE), budget=greet_tokens)
    (file_context,) = result.files
    assert [symbol.name for symbol in file_context.symbols] == ["greet"]  # 2 refs ranks first
    assert result.dropped_symbols == 2
    assert result.total_tokens == greet_tokens


def test_non_python_deleted_and_binary_files_are_skipped(repo: Path) -> None:
    diff = (
        new_file_diff("README.md", "# hi") + "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
    )
    result = retrieve(repo, diff)
    assert result.files == ()


def test_unparseable_changed_file_is_reported(repo: Path) -> None:
    (repo / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    result = retrieve(repo, new_file_diff("bad.py", "def broken(:"))
    (file_context,) = result.files
    assert file_context.symbols == ()
    assert file_context.unresolved == ("bad.py (unreadable or invalid Python)",)


def test_reexported_symbol_followed_one_hop(repo: Path) -> None:
    (repo / "pkg" / "__init__.py").write_text(
        "from pkg.models import Model as PublicModel\n", encoding="utf-8"
    )
    consumer = "import pkg\n\n\ndef use() -> None:\n    pkg.PublicModel()\n"
    (repo / "consumer.py").write_text(consumer, encoding="utf-8")
    result = retrieve(repo, new_file_diff("consumer.py", consumer))
    (file_context,) = result.files
    (symbol,) = file_context.symbols
    assert symbol.name == "Model"  # the real definition, behind the rename
    assert symbol.module_path == "pkg/models.py"
    assert file_context.unresolved == ()


def test_star_reexport_followed_one_hop(repo: Path) -> None:
    (repo / "pkg" / "__init__.py").write_text("from pkg.models import *\n", encoding="utf-8")
    consumer = "import pkg\n\n\ndef use() -> None:\n    pkg.Model()\n"
    (repo / "consumer.py").write_text(consumer, encoding="utf-8")
    result = retrieve(repo, new_file_diff("consumer.py", consumer))
    (file_context,) = result.files
    (symbol,) = file_context.symbols
    assert symbol.name == "Model"
    assert symbol.module_path == "pkg/models.py"
    assert file_context.unresolved == ()


def test_reexport_chase_stops_after_one_hop(repo: Path) -> None:
    (repo / "pkg" / "inner.py").write_text("from pkg.models import Model\n", encoding="utf-8")
    (repo / "pkg" / "__init__.py").write_text("from pkg.inner import Model\n", encoding="utf-8")
    consumer = "import pkg\n\n\ndef use() -> None:\n    pkg.Model()\n"
    (repo / "consumer.py").write_text(consumer, encoding="utf-8")
    result = retrieve(repo, new_file_diff("consumer.py", consumer))
    (file_context,) = result.files
    assert file_context.symbols == ()  # two hops needed: reported, not chased
    assert file_context.unresolved == ("pkg.Model",)


def test_symbol_deduped_across_changed_files(repo: Path) -> None:
    second = 'from helpers import greet\n\n\ndef other() -> None:\n    greet("again")\n'
    (repo / "other.py").write_text(second, encoding="utf-8")
    diff = new_file_diff("app.py", APP_SOURCE) + new_file_diff("other.py", second)
    result = retrieve(repo, diff)
    greet_holders = [
        file_context.file_path
        for file_context in result.files
        for symbol in file_context.symbols
        if symbol.name == "greet"
    ]
    assert greet_holders == ["app.py"]  # retrieved once, for the first referencing file
