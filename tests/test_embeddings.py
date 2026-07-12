import pytest

from enterprise_rag.cache.in_memory import InMemoryCache
from enterprise_rag.config import EmbeddingConfig
from enterprise_rag.embeddings.cached import CachedEmbeddingModel
from enterprise_rag.embeddings.factory import create_embedding_model
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.embeddings.openai_embeddings import OpenAIEmbeddingModel, OpenAIEmbeddingNotConfiguredError
from enterprise_rag.providers.resilience import ProviderResiliencePolicy


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
    class FakeEmbeddings:
        def create(self, **kwargs):
            return object()

    class FakeClient:
        embeddings = FakeEmbeddings()

    model = OpenAIEmbeddingModel(model="example-embedding-model", client=FakeClient())

    assert model.model == "example-embedding-model"
    with pytest.raises(OpenAIEmbeddingNotConfiguredError, match="did not include an embedding"):
        model.embed("hello")


def test_openai_embedding_model_calls_embeddings_api() -> None:
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            data = [type("EmbeddingData", (), {"embedding": [0.1, 0.2, 0.3]})()]
            return type("EmbeddingResponse", (), {"data": data})()

    class FakeClient:
        embeddings = FakeEmbeddings()

    model = OpenAIEmbeddingModel(model="embedding-test", client=FakeClient())

    assert model.embed("hybrid retrieval") == [0.1, 0.2, 0.3]
    assert calls == [{"model": "embedding-test", "input": "hybrid retrieval"}]


def test_openai_embedding_model_uses_resilience_policy_for_transient_failures() -> None:
    calls = 0

    class FakeEmbeddings:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary outage")
            data = [type("EmbeddingData", (), {"embedding": [0.4, 0.5]})()]
            return type("EmbeddingResponse", (), {"data": data})()

    class FakeClient:
        embeddings = FakeEmbeddings()

    model = OpenAIEmbeddingModel(
        model="embedding-test",
        client=FakeClient(),
        resilience=ProviderResiliencePolicy(max_retries=1),
    )

    assert model.embed("hybrid retrieval") == [0.4, 0.5]
    assert calls == 2


def test_embedding_factory_builds_hashing_model() -> None:
    model = create_embedding_model(EmbeddingConfig(provider="hashing", dimensions=12))

    assert isinstance(model, HashingEmbeddingModel)
    assert len(model.embed("hybrid retrieval")) == 12


def test_embedding_factory_builds_openai_model() -> None:
    model = create_embedding_model(
        EmbeddingConfig(
            provider="openai",
            model="embedding-test",
            timeout_seconds=8.0,
            max_retries=1,
            retry_backoff_seconds=0.25,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_reset_seconds=20.0,
        )
    )

    assert isinstance(model, OpenAIEmbeddingModel)
    assert model.model == "embedding-test"
    assert model.timeout_seconds == 8.0
    assert model.resilience.max_retries == 1
    assert model.resilience.retry_backoff_seconds == 0.25
    assert model.resilience.circuit_breaker_failure_threshold == 2
    assert model.resilience.circuit_breaker_reset_seconds == 20.0


def test_embedding_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedding_model(EmbeddingConfig(provider="unknown"))
