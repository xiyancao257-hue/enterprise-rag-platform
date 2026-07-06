from __future__ import annotations


class OpenAIClientNotConfiguredError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def complete(self, prompt: str) -> str:
        raise OpenAIClientNotConfiguredError(
            "OpenAIClient is an adapter stub. Install and configure the OpenAI SDK before using it."
        )
