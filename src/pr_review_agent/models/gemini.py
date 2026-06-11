"""Gemini backend (google-genai SDK) — the default for the agent loop."""

from __future__ import annotations

from collections.abc import Sequence

from google import genai
from google.genai import types as genai_types

from pr_review_agent.models.base import ModelError, ModelMessage, ModelResponse


class GeminiClient:
    """ModelClient backed by the Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
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
        contents = [
            genai_types.Content(
                role="user" if message.role == "user" else "model",
                parts=[genai_types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
        except Exception as exc:  # the SDK raises a wide family of errors
            raise ModelError(f"Gemini request failed: {exc}") from exc
        usage = response.usage_metadata
        return ModelResponse(
            text=response.text or "",
            model=self._model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )
