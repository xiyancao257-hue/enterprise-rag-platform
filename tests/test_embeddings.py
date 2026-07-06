import pytest

from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.embeddings.openai_embeddings import OpenAIEmbeddingModel, OpenAIEmbeddingNotConfiguredError


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


def test_openai_embedding_model_stub_raises_clear_error() -> None:
    model = OpenAIEmbeddingModel(model="example-embedding-model")

    assert model.model == "example-embedding-model"
    with pytest.raises(OpenAIEmbeddingNotConfiguredError, match="adapter stub"):
        model.embed("hello")

