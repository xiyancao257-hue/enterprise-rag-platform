from enterprise_rag.config import LLMConfig
from enterprise_rag.observability.costs import LLMCostEstimator


def test_llm_cost_estimator_counts_tokens_and_estimates_cost() -> None:
    estimator = LLMCostEstimator(
        LLMConfig(
            input_cost_per_1k_tokens=0.10,
            output_cost_per_1k_tokens=0.20,
        )
    )

    cost = estimator.estimate("hybrid retrieval question", "hybrid answer")

    assert cost.input_tokens == 3
    assert cost.output_tokens == 2
    assert cost.estimated_cost_usd == 0.0007
