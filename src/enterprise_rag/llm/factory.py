from __future__ import annotations

from enterprise_rag.config import LLMConfig
from enterprise_rag.llm.base import LLMClient
from enterprise_rag.llm.openai_client import OpenAIClient


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
        return OpenAIClient(model=config.model)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
