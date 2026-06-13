"""Build a ModelClient from config; secrets come from the environment only."""

from __future__ import annotations

from pr_review_agent.config import ConfigError, ModelsConfig, anthropic_api_key, gemini_api_key
from pr_review_agent.models.anthropic_client import AnthropicClient
from pr_review_agent.models.base import ModelClient
from pr_review_agent.models.claude_code import ClaudeCodeClient
from pr_review_agent.models.gemini import GeminiClient
from pr_review_agent.models.ollama import OllamaClient

BACKENDS = ("gemini", "gemini-pro", "anthropic", "ollama", "claude-code")


def build_model_client(backend: str, config: ModelsConfig) -> ModelClient:
    """Construct the named backend, reading its API key from the environment."""
    if backend in ("gemini", "gemini-pro"):
        key = gemini_api_key()
        if key is None:
            raise ConfigError("GEMINI_API_KEY is not set")
        model = config.gemini_pro_model if backend == "gemini-pro" else config.gemini_model
        return GeminiClient(key, model)
    if backend == "anthropic":
        key = anthropic_api_key()
        if key is None:
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        return AnthropicClient(key, config.anthropic_model)
    if backend == "ollama":
        return OllamaClient(config.ollama_base_url, config.ollama_model)
    if backend == "claude-code":
        return ClaudeCodeClient(claude_model=config.claude_code_model)
    raise ConfigError(f"unknown model backend {backend!r}; expected one of {', '.join(BACKENDS)}")
