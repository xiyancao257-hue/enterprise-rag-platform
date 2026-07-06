from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.rag.citations import CitationFormatter


def test_citation_formatter_includes_source_heading_chunk_score_and_retriever() -> None:
    hit = SearchHit(
        chunk=Chunk(
            id="chunk_1",
            document_id="doc1",
            text="Hybrid retrieval combines BM25 and vector search.",
            heading_path=("Enterprise RAG Notes", "Hybrid Retrieval"),
            metadata={"source_path": "data/raw/enterprise_rag_notes.md"},
        ),
        score=0.123456,
        retriever="bm25+vector+rerank",
        rank=1,
    )

    citation = CitationFormatter().format(hit)

    assert citation == (
        "[1] data/raw/enterprise_rag_notes.md - Enterprise RAG Notes > Hybrid Retrieval "
        "(chunk=chunk_1, score=0.1235, retriever=bm25+vector+rerank)"
    )


def test_citation_formatter_handles_missing_source_and_heading() -> None:
    hit = SearchHit(
        chunk=Chunk(id="chunk_2", document_id="doc1", text="Evidence."),
        score=0.2,
        retriever="bm25",
        rank=2,
    )

    citation = CitationFormatter().format(hit)

    assert citation == "[2] unknown (chunk=chunk_2, score=0.2000, retriever=bm25)"
