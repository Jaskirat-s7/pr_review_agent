"""Configuration loading.

Settings live in ``config.toml``; secrets come exclusively from environment
variables and are never written to disk or logged.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
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
class RagConfig:
    """Settings for the RAG retrieval path (index location and local models)."""

    cache_dir: str = ".pra/rag"
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    device: str = "cpu"  # passed to sentence-transformers; e.g. "cuda", "mps"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-million-token prices for one model."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


# Defaults for the shipped backend models (see prices_as_of). Local models
# are free. Override or extend via [models.pricing] in config.toml.
DEFAULT_PRICING: Mapping[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(input_per_mtok=0.30, output_per_mtok=2.50),
    "gemini-2.5-pro": ModelPricing(input_per_mtok=1.25, output_per_mtok=10.00),
    "claude-opus-4-8": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    # Claude Code judge runs on a subscription, not metered API spend.
    "claude-code": ModelPricing(input_per_mtok=0.0, output_per_mtok=0.0),
    "qwen2.5-coder:7b": ModelPricing(input_per_mtok=0.0, output_per_mtok=0.0),
    # Cerebras: free-tier actual spend is $0, but these list prices keep the
    # cost-per-PR column API-list-equivalent (the cost-at-scale story). Confirm
    # against current Cerebras pricing when bumping prices_as_of.
    "qwen-3-32b": ModelPricing(input_per_mtok=0.40, output_per_mtok=0.80),
    "llama-4-scout": ModelPricing(input_per_mtok=0.65, output_per_mtok=0.85),
}

DEFAULT_CEREBRAS_MODELS: tuple[str, ...] = ("qwen-3-32b", "llama-4-scout")


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Model backends, call database, and pricing."""

    backend: str = "gemini"  # agent-loop backend: gemini | anthropic | ollama | claude-code
    db_path: str = "pra.sqlite3"
    gemini_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro"  # eval ceiling baseline (same free tier)
    anthropic_model: str = "claude-opus-4-8"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    # Discovery preference order; the first model the API offers is used.
    cerebras_models: tuple[str, ...] = DEFAULT_CEREBRAS_MODELS
    # Free-tier context window (prompt + completion), in tokens.
    cerebras_context_limit: int = 8192
    # Empty = let the claude CLI pick the plan's default model.
    claude_code_model: str = ""
    judge_backend: str = "claude-code"  # eval judge backend
    ceiling_backend: str = "gemini-pro"  # run label marked "(ceiling)" in the report
    prices_as_of: str = "2026-06"
    pricing: Mapping[str, ModelPricing] = field(default_factory=lambda: dict(DEFAULT_PRICING))


@dataclass(frozen=True, slots=True)
class ReviewConfig:
    """Settings for the review engine."""

    max_comments: int = 3
    confidence_threshold: float = 0.6
    # Comments on hunks with no retrieved context face a higher bar:
    # hallucination risk is higher when the model only sees the hunk.
    no_context_confidence_threshold: float = 0.8


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    github: GitHubConfig = field(default_factory=GitHubConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    rag: RagConfig = field(default_factory=RagConfig)


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
        models=_models_config(raw.get("models", {}), source=path),
        review=_review_config(raw.get("review", {}), source=path),
        rag=_rag_config(raw.get("rag", {}), source=path),
    )


def github_token() -> str | None:
    """Return the GitHub token from the environment, if set and non-empty."""
    return os.environ.get("GITHUB_TOKEN") or None


def gemini_api_key() -> str | None:
    """Return the Gemini API key from the environment, if set and non-empty."""
    return os.environ.get("GEMINI_API_KEY") or None


def anthropic_api_key() -> str | None:
    """Return the Anthropic API key from the environment, if set and non-empty."""
    return os.environ.get("ANTHROPIC_API_KEY") or None


def cerebras_api_key() -> str | None:
    """Return the Cerebras API key from the environment, if set and non-empty."""
    return os.environ.get("CEREBRAS_API_KEY") or None


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


def _rag_config(raw: object, *, source: Path) -> RagConfig:
    table = _expect_table(raw, "rag", source, RagConfig)
    defaults = RagConfig()
    section = "rag"
    return RagConfig(
        cache_dir=_expect(table, "cache_dir", str, defaults.cache_dir, source, section),
        embedding_model=_expect(
            table, "embedding_model", str, defaults.embedding_model, source, section
        ),
        device=_expect(table, "device", str, defaults.device, source, section),
    )


def _models_config(raw: object, *, source: Path) -> ModelsConfig:
    table = _expect_table(raw, "models", source, ModelsConfig)
    defaults = ModelsConfig()
    section = "models"
    pricing = dict(DEFAULT_PRICING)
    pricing.update(_pricing_table(table.get("pricing", {}), source=source))
    return ModelsConfig(
        backend=_expect(table, "backend", str, defaults.backend, source, section),
        db_path=_expect(table, "db_path", str, defaults.db_path, source, section),
        gemini_model=_expect(table, "gemini_model", str, defaults.gemini_model, source, section),
        gemini_pro_model=_expect(
            table, "gemini_pro_model", str, defaults.gemini_pro_model, source, section
        ),
        anthropic_model=_expect(
            table, "anthropic_model", str, defaults.anthropic_model, source, section
        ),
        ollama_base_url=_expect(
            table, "ollama_base_url", str, defaults.ollama_base_url, source, section
        ),
        ollama_model=_expect(table, "ollama_model", str, defaults.ollama_model, source, section),
        cerebras_base_url=_expect(
            table, "cerebras_base_url", str, defaults.cerebras_base_url, source, section
        ),
        cerebras_models=_expect_str_tuple(
            table, "cerebras_models", defaults.cerebras_models, source, section
        ),
        cerebras_context_limit=_expect(
            table, "cerebras_context_limit", int, defaults.cerebras_context_limit, source, section
        ),
        claude_code_model=_expect(
            table, "claude_code_model", str, defaults.claude_code_model, source, section
        ),
        judge_backend=_expect(table, "judge_backend", str, defaults.judge_backend, source, section),
        ceiling_backend=_expect(
            table, "ceiling_backend", str, defaults.ceiling_backend, source, section
        ),
        prices_as_of=_expect(table, "prices_as_of", str, defaults.prices_as_of, source, section),
        pricing=pricing,
    )


def _pricing_table(raw: object, *, source: Path) -> dict[str, ModelPricing]:
    if not isinstance(raw, dict):
        raise ConfigError(f"[models.pricing] must be a table in {source}")
    pricing: dict[str, ModelPricing] = {}
    for model, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"[models.pricing.{model}] must be a table in {source}")
        unknown = sorted(set(entry) - {"input_per_mtok", "output_per_mtok"})
        if unknown:
            raise ConfigError(
                f"unknown keys in [models.pricing.{model}] of {source}: {', '.join(unknown)}"
            )
        section = f"models.pricing.{model}"
        pricing[str(model)] = ModelPricing(
            input_per_mtok=_expect_number(entry, "input_per_mtok", 0.0, source, section),
            output_per_mtok=_expect_number(entry, "output_per_mtok", 0.0, source, section),
        )
    return pricing


def _review_config(raw: object, *, source: Path) -> ReviewConfig:
    table = _expect_table(raw, "review", source, ReviewConfig)
    defaults = ReviewConfig()
    section = "review"
    return ReviewConfig(
        max_comments=_expect(table, "max_comments", int, defaults.max_comments, source, section),
        confidence_threshold=_expect_number(
            table, "confidence_threshold", defaults.confidence_threshold, source, section
        ),
        no_context_confidence_threshold=_expect_number(
            table,
            "no_context_confidence_threshold",
            defaults.no_context_confidence_threshold,
            source,
            section,
        ),
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


def _expect_str_tuple(
    raw: dict[str, Any], key: str, default: tuple[str, ...], source: Path, section: str
) -> tuple[str, ...]:
    value = raw.get(key, default)
    if isinstance(value, tuple):
        return value
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{key}' in [{section}] of {source} must be a list of strings")
    if not value:
        raise ConfigError(f"'{key}' in [{section}] of {source} must not be empty")
    return tuple(value)


def _expect_number(
    raw: dict[str, Any], key: str, default: float, source: Path, section: str
) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{key}' in [{section}] of {source} must be a number")
    return float(value)
