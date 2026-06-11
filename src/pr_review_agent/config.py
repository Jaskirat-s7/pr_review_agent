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
    timeout_seconds: float = 30.0
    min_rate_limit_remaining: int = 25
    max_retries: int = 3
    max_sleep_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    github: GitHubConfig = field(default_factory=GitHubConfig)


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
    return AppConfig(github=_github_config(raw.get("github", {}), source=path))


def github_token() -> str | None:
    """Return the GitHub token from the environment, if set and non-empty."""
    return os.environ.get("GITHUB_TOKEN") or None


def _github_config(raw: object, *, source: Path) -> GitHubConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"[github] must be a table in {source}")
    known = {f.name for f in fields(GitHubConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown keys in [github] of {source}: {', '.join(unknown)}")
    defaults = GitHubConfig()
    return GitHubConfig(
        base_url=_expect(raw, "base_url", str, defaults.base_url, source),
        timeout_seconds=_expect_number(raw, "timeout_seconds", defaults.timeout_seconds, source),
        min_rate_limit_remaining=_expect(
            raw, "min_rate_limit_remaining", int, defaults.min_rate_limit_remaining, source
        ),
        max_retries=_expect(raw, "max_retries", int, defaults.max_retries, source),
        max_sleep_seconds=_expect_number(
            raw, "max_sleep_seconds", defaults.max_sleep_seconds, source
        ),
    )


def _expect(raw: dict[str, Any], key: str, typ: type[_T], default: _T, source: Path) -> _T:
    value = raw.get(key, default)
    if isinstance(value, bool) and typ is not bool:
        raise ConfigError(f"'{key}' in [github] of {source} must be a {typ.__name__}")
    if not isinstance(value, typ):
        raise ConfigError(f"'{key}' in [github] of {source} must be a {typ.__name__}")
    return value


def _expect_number(raw: dict[str, Any], key: str, default: float, source: Path) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{key}' in [github] of {source} must be a number")
    return float(value)
