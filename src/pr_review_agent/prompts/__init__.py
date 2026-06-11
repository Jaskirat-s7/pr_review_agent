"""Versioned prompt templates, loaded as package resources."""

from __future__ import annotations

import string
from importlib import resources


def load_prompt(name: str) -> string.Template:
    """Load ``prompts/<name>.md`` as a string.Template ($placeholders)."""
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    return string.Template(text)
