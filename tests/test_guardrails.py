from enterprise_rag.config import GuardrailsConfig
from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.observability.costs import EstimatedLLMCost
from enterprise_rag.rag.guardrails import QueryGuardrailPolicy


def test_query_guardrail_policy_allows_strong_low_cost_evidence() -> None:
    decision = QueryGuardrailPolicy(GuardrailsConfig()).evaluate(
        "hybrid retrieval",
        (
            SearchHit(
                chunk=Chunk(
                    id="chunk1",
                    document_id="doc1",
                    text="Hybrid retrieval combines BM25 keyword search with vector search.",
                ),
                score=0.8,
                retriever="hybrid",
                rank=1,
            ),
        ),
        EstimatedLLMCost(input_tokens=10, output_tokens=5, estimated_cost_usd=0.001),
        latency_ms=25.0,
    )

    assert decision.needs_human_review is False
    assert decision.reasons == ()


def test_query_guardrail_policy_flags_human_review_reasons() -> None:
    decision = QueryGuardrailPolicy(
        GuardrailsConfig(
            min_citations=2,
            min_top_score=0.5,
            min_evidence_tokens=10,
            max_estimated_cost_usd=0.001,
            max_latency_ms=10,
            sensitive_terms=("legal",),
        )
    ).evaluate(
        "legal policy",
        (
            SearchHit(
                chunk=Chunk(id="chunk1", document_id="doc1", text="short"),
                score=0.2,
                retriever="hybrid",
                rank=1,
            ),
        ),
        EstimatedLLMCost(input_tokens=100, output_tokens=50, estimated_cost_usd=0.002),
        latency_ms=20.0,
    )

    assert decision.needs_human_review is True
    assert decision.reasons == (
        "low_evidence",
        "low_confidence",
        "insufficient_context",
        "cost_budget_exceeded",
        "latency_budget_exceeded",
        "sensitive_topic",
    )
