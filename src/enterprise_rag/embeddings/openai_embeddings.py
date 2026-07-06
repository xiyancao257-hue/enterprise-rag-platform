from __future__ import annotations


class OpenAIEmbeddingNotConfiguredError(RuntimeError):
    pass


class OpenAIEmbeddingModel:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    def embed(self, text: str) -> list[float]:
        raise OpenAIEmbeddingNotConfiguredError(
            "OpenAIEmbeddingModel is an adapter stub. Install and configure the OpenAI SDK before using it."
        )
