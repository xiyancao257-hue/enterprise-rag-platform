from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.rag.answer_generation import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    DeterministicAnswerGenerator,
    EvidenceSufficiencyPolicy,
    LLMAnswerGenerator,
)


def test_answer_generator_refuses_when_no_evidence() -> None:
    answer = DeterministicAnswerGenerator().generate("unknown policy", [])

    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE


def test_answer_generator_refuses_when_evidence_score_is_too_low() -> None:
    hits = [
        SearchHit(
            chunk=Chunk(id="chunk_1", document_id="doc1", text="Relevant looking evidence with enough words."),
            score=0.001,
            retriever="test",
            rank=1,
        )
    ]

    answer = DeterministicAnswerGenerator().generate("query", hits)

    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE


def test_answer_generator_refuses_when_evidence_text_is_too_short() -> None:
    hits = [
        SearchHit(
            chunk=Chunk(id="chunk_1", document_id="doc1", text="tiny"),
            score=0.5,
            retriever="test",
            rank=1,
        )
    ]

    answer = DeterministicAnswerGenerator().generate("query", hits)

    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE


def test_evidence_sufficiency_policy_can_be_tuned() -> None:
    strict_policy = EvidenceSufficiencyPolicy(min_hits=2, min_top_score=0.5, min_total_tokens=4)
    hits = [
        SearchHit(
            chunk=Chunk(id="chunk_1", document_id="doc1", text="one useful evidence sentence"),
            score=0.7,
            retriever="test",
            rank=1,
        )
    ]

    assert not strict_policy.is_sufficient(hits)


def test_answer_generator_uses_top_three_hits() -> None:
    hits = [
        SearchHit(
            chunk=Chunk(id=f"chunk_{index}", document_id="doc1", text=f"Evidence {index}"),
            score=1.0 / index,
            retriever="test",
            rank=index,
        )
        for index in range(1, 5)
    ]

    answer = DeterministicAnswerGenerator().generate("query", hits)

    assert "Evidence 1" in answer
    assert "Evidence 2" in answer
    assert "Evidence 3" in answer
    assert "Evidence 4" not in answer


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_llm_answer_generator_renders_prompt_and_returns_client_response() -> None:
    client = FakeLLMClient("Hybrid retrieval combines BM25 and vector search [1].")
    hits = [
        SearchHit(
            chunk=Chunk(
                id="chunk_1",
                document_id="doc1",
                text="Hybrid retrieval combines BM25 keyword search with vector search.",
                metadata={"source_path": "memory.md"},
            ),
            score=0.5,
            retriever="bm25+vector+rerank",
            rank=1,
        )
    ]

    answer = LLMAnswerGenerator(client).generate("What is hybrid retrieval?", hits)

    assert answer == "Hybrid retrieval combines BM25 and vector search [1]."
    assert len(client.prompts) == 1
    assert "Question:\nWhat is hybrid retrieval?" in client.prompts[0]
    assert "[1] Hybrid retrieval combines BM25 keyword search with vector search." in client.prompts[0]
    assert "Cite every factual claim" in client.prompts[0]


def test_llm_answer_generator_does_not_call_client_when_evidence_is_insufficient() -> None:
    client = FakeLLMClient("This should not be used.")

    answer = LLMAnswerGenerator(client).generate("Unknown?", [])

    assert answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert client.prompts == []
