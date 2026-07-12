from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.config import GuardrailsConfig
from enterprise_rag.models import SearchHit
from enterprise_rag.observability.costs import EstimatedLLMCost
from enterprise_rag.text import tokenize


@dataclass(frozen=True)
class QueryGuardrailDecision:
    needs_human_review: bool
    reasons: tuple[str, ...]


class QueryGuardrailPolicy:
    def __init__(self, config: GuardrailsConfig) -> None:
        self.config = config

    def evaluate(
        self,
        query: str,
        citations: tuple[SearchHit, ...],
        cost: EstimatedLLMCost,
        latency_ms: float,
    ) -> QueryGuardrailDecision:
        reasons = []
        if len(citations) < self.config.min_citations:
            reasons.append("low_evidence")
        if citations and citations[0].score < self.config.min_top_score:
            reasons.append("low_confidence")
        if _total_evidence_tokens(citations) < self.config.min_evidence_tokens:
            reasons.append("insufficient_context")
        if self.config.max_estimated_cost_usd > 0 and cost.estimated_cost_usd > self.config.max_estimated_cost_usd:
            reasons.append("cost_budget_exceeded")
        if self.config.max_latency_ms > 0 and latency_ms > self.config.max_latency_ms:
            reasons.append("latency_budget_exceeded")
        if _contains_sensitive_topic(query, self.config.sensitive_terms):
            reasons.append("sensitive_topic")
        return QueryGuardrailDecision(needs_human_review=bool(reasons), reasons=tuple(reasons))


def _total_evidence_tokens(citations: tuple[SearchHit, ...]) -> int:
    return sum(len(tokenize(hit.chunk.text)) for hit in citations)


def _contains_sensitive_topic(query: str, sensitive_terms: tuple[str, ...]) -> bool:
    query_tokens = set(tokenize(query))
    return any(term.lower() in query_tokens for term in sensitive_terms)
