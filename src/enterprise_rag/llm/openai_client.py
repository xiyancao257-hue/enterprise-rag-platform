from __future__ import annotations

from enterprise_rag.providers.resilience import ProviderResiliencePolicy


class OpenAIClientNotConfiguredError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        client: object | None = None,
        timeout_seconds: float = 30.0,
        resilience: ProviderResiliencePolicy | None = None,
    ) -> None:
        self.model = model
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.resilience = resilience or ProviderResiliencePolicy()

    def complete(self, prompt: str) -> str:
        return self.resilience.call(lambda: self._complete_once(prompt), provider_name="openai_llm")

    def _complete_once(self, prompt: str) -> str:
        client = self.client or _default_openai_client(timeout_seconds=self.timeout_seconds)
        response = client.responses.create(model=self.model, input=prompt)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        raise OpenAIClientNotConfiguredError("OpenAI response did not include `output_text`.")


def _default_openai_client(timeout_seconds: float) -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIClientNotConfiguredError(
            "OpenAI SDK is not installed. Install the optional OpenAI dependency before using provider=openai."
        ) from exc
    return OpenAI(timeout=timeout_seconds)
