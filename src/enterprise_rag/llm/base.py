from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str) -> str:
        """Return a model completion for the rendered prompt."""
