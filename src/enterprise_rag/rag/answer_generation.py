from __future__ import annotations

from typing import Protocol

from enterprise_rag.llm.base import LLMClient
from enterprise_rag.models import SearchHit
from enterprise_rag.rag.prompts import GroundedQAPromptTemplate
from enterprise_rag.text import tokenize

INSUFFICIENT_EVIDENCE_MESSAGE = "I could not find enough grounded context to answer this query."


class AnswerGenerator(Protocol):
    def generate(self, query: str, hits: list[SearchHit]) -> str:
        """Generate an answer from retrieved evidence."""


class EvidenceSufficiencyPolicy:
    def __init__(self, min_hits: int = 1, min_top_score: float = 0.01, min_total_tokens: int = 5) -> None:
        self.min_hits = min_hits
        self.min_top_score = min_top_score
        self.min_total_tokens = min_total_tokens

    def is_sufficient(self, hits: list[SearchHit]) -> bool:
        if len(hits) < self.min_hits:
            return False
        if max(hit.score for hit in hits) < self.min_top_score:
            return False
        total_tokens = sum(len(tokenize(hit.chunk.text)) for hit in hits)
        return total_tokens >= self.min_total_tokens


class DeterministicAnswerGenerator:
    def __init__(self, sufficiency_policy: EvidenceSufficiencyPolicy | None = None) -> None:
        self.sufficiency_policy = sufficiency_policy or EvidenceSufficiencyPolicy()

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        if not self.sufficiency_policy.is_sufficient(hits):
            return INSUFFICIENT_EVIDENCE_MESSAGE
        evidence = "\n".join(f"- {hit.chunk.text}" for hit in hits[:3])
        return (
            "Grounded draft answer based on retrieved evidence:\n"
            f"{evidence}\n\n"
            "Next step: replace this draft generator with an LLM call that only uses the cited context."
        )


class LLMAnswerGenerator:
    def __init__(
        self,
        client: LLMClient,
        prompt_template: GroundedQAPromptTemplate | None = None,
        sufficiency_policy: EvidenceSufficiencyPolicy | None = None,
    ) -> None:
        self.client = client
        self.prompt_template = prompt_template or GroundedQAPromptTemplate()
        self.sufficiency_policy = sufficiency_policy or EvidenceSufficiencyPolicy()

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        if not self.sufficiency_policy.is_sufficient(hits):
            return INSUFFICIENT_EVIDENCE_MESSAGE
        prompt = self.prompt_template.render(query, hits)
        return self.client.complete(prompt).strip()
