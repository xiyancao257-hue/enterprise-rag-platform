from __future__ import annotations

from typing import Protocol


class EmbeddingModel(Protocol):
    def embed(self, text: str) -> list[float]:
        """Embed text into a dense vector."""
