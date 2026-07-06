import pytest

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

