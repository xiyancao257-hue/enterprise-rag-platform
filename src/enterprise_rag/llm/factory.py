from __future__ import annotations

from enterprise_rag.config import LLMConfig
from enterprise_rag.llm.base import LLMClient
from enterprise_rag.llm.openai_client import OpenAIClient
from enterprise_rag.providers.resilience import ProviderResiliencePolicy


class StubLLMClient:
    def __init__(self, model: str = "stub") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        return "Stub LLM client is configured; replace it with a real provider for generation."


def create_llm_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.lower()
    if provider == "stub":
        return StubLLMClient(model=config.model)
    if provider == "openai":
        return OpenAIClient(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            resilience=ProviderResiliencePolicy(
                max_retries=config.max_retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_reset_seconds=config.circuit_breaker_reset_seconds,
            ),
        )
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
