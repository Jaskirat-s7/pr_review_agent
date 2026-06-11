"""Anthropic backend — used by the eval judge and ceiling-baseline runs.

No sampling parameters are sent: ``temperature``/``top_p``/``top_k`` are
removed on Opus 4.7+ and return HTTP 400.
"""

from __future__ import annotations

from collections.abc import Sequence

import anthropic
from anthropic.types import MessageParam

from pr_review_agent.models.base import ModelError, ModelMessage, ModelResponse


class AnthropicClient:
    """ModelClient backed by the Anthropic API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-8",
        *,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        messages: Sequence[ModelMessage],
        *,
        max_tokens: int = 1024,
        purpose: str = "",
    ) -> ModelResponse:
        message_params: list[MessageParam] = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=message_params,
            )
        except anthropic.AnthropicError as exc:
            raise ModelError(f"Anthropic request failed: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResponse(
            text=text,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
