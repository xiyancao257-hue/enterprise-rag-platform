from __future__ import annotations


class OpenAIEmbeddingNotConfiguredError(RuntimeError):
    pass


class OpenAIEmbeddingModel:
    def __init__(self, model: str = "text-embedding-3-small", client: object | None = None) -> None:
        self.model = model
        self.client = client

    def embed(self, text: str) -> list[float]:
        client = self.client or _default_openai_client()
        response = client.embeddings.create(model=self.model, input=text)
        try:
            return list(response.data[0].embedding)
        except (AttributeError, IndexError, TypeError) as exc:
            raise OpenAIEmbeddingNotConfiguredError("OpenAI embedding response did not include an embedding.") from exc


def _default_openai_client() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIEmbeddingNotConfiguredError(
            "OpenAI SDK is not installed. Install the optional OpenAI dependency before using provider=openai."
        ) from exc
    return OpenAI()
