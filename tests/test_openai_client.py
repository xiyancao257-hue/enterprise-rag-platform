import pytest

from enterprise_rag.config import LLMConfig
from enterprise_rag.llm.factory import StubLLMClient, create_llm_client
from enterprise_rag.llm.openai_client import OpenAIClient, OpenAIClientNotConfiguredError
from enterprise_rag.rag.answer_generation import LLMAnswerGenerator


def test_openai_client_stores_model_name() -> None:
    client = OpenAIClient(model="example-model")

    assert client.model == "example-model"


def test_openai_client_stub_raises_clear_configuration_error() -> None:
    client = OpenAIClient()

    with pytest.raises(OpenAIClientNotConfiguredError, match="adapter stub"):
        client.complete("prompt")


def test_openai_client_matches_llm_answer_generator_client_interface() -> None:
    generator = LLMAnswerGenerator(OpenAIClient())

    assert generator.client.model == "gpt-4.1-mini"


def test_create_llm_client_builds_stub_client() -> None:
    client = create_llm_client(LLMConfig(provider="stub", model="local-test"))

    assert isinstance(client, StubLLMClient)
    assert client.model == "local-test"


def test_create_llm_client_builds_openai_client() -> None:
    client = create_llm_client(LLMConfig(provider="openai", model="gpt-test"))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-test"


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(LLMConfig(provider="unknown"))
