from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.config import LLMConfig
from enterprise_rag.text import tokenize


@dataclass(frozen=True)
class EstimatedLLMCost:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class LLMCostEstimator:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def estimate(self, prompt: str, completion: str) -> EstimatedLLMCost:
        input_tokens = len(tokenize(prompt))
        output_tokens = len(tokenize(completion))
        cost = (input_tokens / 1000 * self.config.input_cost_per_1k_tokens) + (
            output_tokens / 1000 * self.config.output_cost_per_1k_tokens
        )
        return EstimatedLLMCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 8),
        )
