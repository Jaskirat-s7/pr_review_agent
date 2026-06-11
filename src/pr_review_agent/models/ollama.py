"""Ollama backend (local HTTP) — for offline development."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from pr_review_agent.models.base import ModelError, ModelMessage, ModelResponse


class OllamaClient:
    """ModelClient backed by a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise ModelError(f"Ollama request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ModelError(
                f"Ollama request failed with HTTP {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise ModelError("Ollama returned a non-object JSON response")
        message = data.get("message")
        text = message.get("content", "") if isinstance(message, dict) else ""
        return ModelResponse(
            text=str(text),
            model=self._model,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
