"""Configuration loading.

Settings live in ``config.toml``; secrets come exclusively from environment
variables and are never written to disk or logged.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_CONFIG_FILENAME = "config.toml"

_T = TypeVar("_T")


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or has bad values."""


@dataclass(frozen=True, slots=True)
class GitHubConfig:
    """Settings for the GitHub REST client."""

    base_url: str = "https://api.github.com"
    clone_base_url: str = "https://github.com"
    timeout_seconds: float = 30.0
    min_rate_limit_remaining: int = 25
    max_retries: int = 3
    max_sleep_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Settings for changed-code context retrieval."""

    token_budget: int = 6000


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    github: GitHubConfig = field(default_factory=GitHubConfig)
    context: ContextConfig = field(default_factory=ContextConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from ``path``, or ``./config.toml`` if present.

    A missing default file yields built-in defaults; an explicitly provided
    path must exist.
    """
    if path is None:
        default = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if not default.is_file():
            return AppConfig()
        path = default
    elif not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return AppConfig(
        github=_github_config(raw.get("github", {}), source=path),
        context=_context_config(raw.get("context", {}), source=path),
    )


def github_token() -> str | None:
    """Return the GitHub token from the environment, if set and non-empty."""
    return os.environ.get("GITHUB_TOKEN") or None


def _github_config(raw: object, *, source: Path) -> GitHubConfig:
    table = _expect_table(raw, "github", source, GitHubConfig)
    defaults = GitHubConfig()
    section = "github"
    return GitHubConfig(
        base_url=_expect(table, "base_url", str, defaults.base_url, source, section),
        clone_base_url=_expect(
            table, "clone_base_url", str, defaults.clone_base_url, source, section
        ),
        timeout_seconds=_expect_number(
            table, "timeout_seconds", defaults.timeout_seconds, source, section
        ),
        min_rate_limit_remaining=_expect(
            table,
            "min_rate_limit_remaining",
            int,
            defaults.min_rate_limit_remaining,
            source,
            section,
        ),
        max_retries=_expect(table, "max_retries", int, defaults.max_retries, source, section),
        max_sleep_seconds=_expect_number(
            table, "max_sleep_seconds", defaults.max_sleep_seconds, source, section
        ),
    )


def _context_config(raw: object, *, source: Path) -> ContextConfig:
    table = _expect_table(raw, "context", source, ContextConfig)
    defaults = ContextConfig()
    return ContextConfig(
        token_budget=_expect(table, "token_budget", int, defaults.token_budget, source, "context"),
    )


def _expect_table(raw: object, section: str, source: Path, cls: type) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table in {source}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown keys in [{section}] of {source}: {', '.join(unknown)}")
    return raw


def _expect(
    raw: dict[str, Any], key: str, typ: type[_T], default: _T, source: Path, section: str
) -> _T:
    value = raw.get(key, default)
    if (isinstance(value, bool) and typ is not bool) or not isinstance(value, typ):
        raise ConfigError(f"'{key}' in [{section}] of {source} must be a {typ.__name__}")
    return value


def _expect_number(
    raw: dict[str, Any], key: str, default: float, source: Path, section: str
) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{key}' in [{section}] of {source} must be a number")
    return float(value)
