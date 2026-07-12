import pytest

from enterprise_rag.config import LLMConfig
from enterprise_rag.llm.factory import StubLLMClient, create_llm_client
from enterprise_rag.llm.openai_client import OpenAIClient, OpenAIClientNotConfiguredError
from enterprise_rag.providers.resilience import ProviderResiliencePolicy
from enterprise_rag.rag.answer_generation import LLMAnswerGenerator


def test_openai_client_stores_model_name() -> None:
    client = OpenAIClient(model="example-model")

    assert client.model == "example-model"


def test_openai_client_stub_raises_clear_configuration_error() -> None:
    class FakeResponses:
        def create(self, **kwargs):
            return object()

    class FakeClient:
        responses = FakeResponses()

    client = OpenAIClient(client=FakeClient())

    with pytest.raises(OpenAIClientNotConfiguredError, match="output_text"):
        client.complete("prompt")


def test_openai_client_calls_responses_api() -> None:
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return type("Response", (), {"output_text": "grounded answer"})()

    class FakeClient:
        responses = FakeResponses()

    client = OpenAIClient(model="gpt-test", client=FakeClient())

    assert client.complete("prompt text") == "grounded answer"
    assert calls == [{"model": "gpt-test", "input": "prompt text"}]


def test_openai_client_uses_resilience_policy_for_transient_failures() -> None:
    calls = 0

    class FakeResponses:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary outage")
            return type("Response", (), {"output_text": "recovered answer"})()

    class FakeClient:
        responses = FakeResponses()

    client = OpenAIClient(
        model="gpt-test",
        client=FakeClient(),
        resilience=ProviderResiliencePolicy(max_retries=1),
    )

    assert client.complete("prompt text") == "recovered answer"
    assert calls == 2


def test_openai_client_matches_llm_answer_generator_client_interface() -> None:
    generator = LLMAnswerGenerator(OpenAIClient())

    assert generator.client.model == "gpt-4.1-mini"


def test_create_llm_client_builds_stub_client() -> None:
    client = create_llm_client(LLMConfig(provider="stub", model="local-test"))

    assert isinstance(client, StubLLMClient)
    assert client.model == "local-test"


def test_create_llm_client_builds_openai_client() -> None:
    client = create_llm_client(
        LLMConfig(
            provider="openai",
            model="gpt-test",
            timeout_seconds=12.0,
            max_retries=2,
            retry_backoff_seconds=0.5,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_reset_seconds=45.0,
        )
    )

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-test"
    assert client.timeout_seconds == 12.0
    assert client.resilience.max_retries == 2
    assert client.resilience.retry_backoff_seconds == 0.5
    assert client.resilience.circuit_breaker_failure_threshold == 3
    assert client.resilience.circuit_breaker_reset_seconds == 45.0


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(LLMConfig(provider="unknown"))
