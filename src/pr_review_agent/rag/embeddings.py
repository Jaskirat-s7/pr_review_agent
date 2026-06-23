"""Local code embeddings via sentence-transformers.

The model (jina code embeddings by default) is downloaded and held in memory on
first use. sentence-transformers and torch are imported lazily so the package
imports without the ``rag`` extra installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class CodeEmbedder:
    """Encodes text into dense vectors with a local sentence-transformers model."""

    def __init__(self, model_name: str, *, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, device=self._device, trust_remote_code=True
            )
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
