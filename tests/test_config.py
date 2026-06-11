"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from pr_review_agent.config import (
    AppConfig,
    ConfigError,
    GitHubConfig,
    ModelPricing,
    anthropic_api_key,
    gemini_api_key,
    github_token,
    load_config,
)

REPO_ROOT = Path(__file__).parent.parent


def test_defaults_when_no_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config() == AppConfig()


def test_explicit_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_example_config_parses_to_defaults() -> None:
    config = load_config(REPO_ROOT / "config.toml.example")
    assert config == AppConfig(github=GitHubConfig())


def test_values_are_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[github]\nbase_url = "https://ghe.example.com/api/v3"\n'
        "timeout_seconds = 10\nmax_retries = 1\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.github.base_url == "https://ghe.example.com/api/v3"
    assert config.github.timeout_seconds == 10.0
    assert config.github.max_retries == 1
    # unspecified keys keep defaults
    assert config.github.min_rate_limit_remaining == GitHubConfig().min_rate_limit_remaining


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[github]\nmin_rate_limit = 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown keys.*min_rate_limit"):
        load_config(path)


def test_wrong_value_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[github]\nmax_retries = "three"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="max_retries"):
        load_config(path)


def test_invalid_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[github\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_models_section_with_pricing_override(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[models]\nbackend = "ollama"\nprices_as_of = "2027-01"\n'
        '[models.pricing."custom-model"]\ninput_per_mtok = 1.5\noutput_per_mtok = 6\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.models.backend == "ollama"
    assert config.models.prices_as_of == "2027-01"
    assert config.models.pricing["custom-model"] == ModelPricing(1.5, 6.0)
    # built-in defaults are preserved alongside overrides
    assert "gemini-2.5-flash" in config.models.pricing


def test_models_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[models]\nmodle = "gemini"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown keys.*modle"):
        load_config(path)


def test_pricing_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[models.pricing."m"]\ninput_cost = 1.0\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"models\.pricing\.m"):
        load_config(path)


def test_review_section_parses(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[review]\nmax_comments = 5\nconfidence_threshold = 0.5\n", encoding="utf-8")
    config = load_config(path)
    assert config.review.max_comments == 5
    assert config.review.confidence_threshold == 0.5
    assert config.review.no_context_confidence_threshold == 0.8  # default kept


def test_model_api_keys_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert gemini_api_key() == "g-key"
    assert anthropic_api_key() == "a-key"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert gemini_api_key() is None
    assert anthropic_api_key() is None


def test_github_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    assert github_token() == "tok123"
    monkeypatch.setenv("GITHUB_TOKEN", "")
    assert github_token() is None
    monkeypatch.delenv("GITHUB_TOKEN")
    assert github_token() is None
