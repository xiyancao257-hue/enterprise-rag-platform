from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.rag.prompts import GroundedQAPromptTemplate


def test_grounded_qa_prompt_includes_question_evidence_and_citations() -> None:
    hit = SearchHit(
        chunk=Chunk(
            id="chunk_1",
            document_id="doc1",
            text="Hybrid retrieval combines BM25 keyword search with vector search.",
            heading_path=("Enterprise RAG Notes", "Hybrid Retrieval"),
            metadata={"source_path": "data/raw/enterprise_rag_notes.md"},
        ),
        score=0.42,
        retriever="bm25+vector+rerank",
        rank=1,
    )

    prompt = GroundedQAPromptTemplate().render("What is hybrid retrieval?", [hit])

    assert "You are an enterprise RAG assistant." in prompt
    assert "using only the provided evidence" in prompt
    assert "If the evidence is insufficient" in prompt
    assert "Do not invent facts" in prompt
    assert "Cite every factual claim" in prompt
    assert "Question:\nWhat is hybrid retrieval?" in prompt
    assert "[1] Hybrid retrieval combines BM25 keyword search with vector search." in prompt
    assert (
        "[1] data/raw/enterprise_rag_notes.md - Enterprise RAG Notes > Hybrid Retrieval "
        "(chunk=chunk_1, score=0.4200, retriever=bm25+vector+rerank)"
    ) in prompt
    assert prompt.endswith("Answer:")


def test_grounded_qa_prompt_handles_empty_evidence() -> None:
    prompt = GroundedQAPromptTemplate().render("Unknown question?", [])

    assert "Evidence:\nNo evidence was retrieved." in prompt
    assert "Citations:\nNo citations.\n\nAnswer:" in prompt
