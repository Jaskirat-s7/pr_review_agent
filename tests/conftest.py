"""Shared test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(*parts: str) -> str:
    return FIXTURES_DIR.joinpath(*parts).read_text(encoding="utf-8")
