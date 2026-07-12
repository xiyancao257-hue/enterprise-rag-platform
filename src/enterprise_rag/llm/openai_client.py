from __future__ import annotations


class OpenAIClientNotConfiguredError(RuntimeError):
    pass


class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini", client: object | None = None) -> None:
        self.model = model
        self.client = client

    def complete(self, prompt: str) -> str:
        client = self.client or _default_openai_client()
        response = client.responses.create(model=self.model, input=prompt)
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        raise OpenAIClientNotConfiguredError("OpenAI response did not include `output_text`.")


def _default_openai_client() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIClientNotConfiguredError(
            "OpenAI SDK is not installed. Install the optional OpenAI dependency before using provider=openai."
        ) from exc
    return OpenAI()
