import pytest

from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.embeddings.cached import CachedEmbeddingModel
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.embeddings.openai_embeddings import OpenAIEmbeddingModel, OpenAIEmbeddingNotConfiguredError


class CountingEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text))]


def test_hashing_embedding_model_is_deterministic_and_normalized() -> None:
    model = HashingEmbeddingModel(dimensions=16)

    first = model.embed("hybrid retrieval")
    second = model.embed("hybrid retrieval")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_hashing_embedding_model_returns_zero_vector_for_empty_text() -> None:
    model = HashingEmbeddingModel(dimensions=8)

    assert model.embed("") == [0.0] * 8


def test_cached_embedding_model_reuses_cached_vector() -> None:
    base_model = CountingEmbeddingModel()
    cache = InMemoryCache()
    model = CachedEmbeddingModel(base_model, cache=cache, model_id="counting")

    first = model.embed("hybrid retrieval")
    second = model.embed("hybrid retrieval")

    assert first == [16.0]
    assert second == [16.0]
    assert base_model.calls == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_openai_embedding_model_stub_raises_clear_error() -> None:
    model = OpenAIEmbeddingModel(model="example-embedding-model")

    assert model.model == "example-embedding-model"
    with pytest.raises(OpenAIEmbeddingNotConfiguredError, match="adapter stub"):
        model.embed("hello")
