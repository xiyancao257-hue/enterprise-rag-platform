from enterprise_rag.models import Chunk
from enterprise_rag.observability.tracing import format_query_trace
from enterprise_rag.rag.pipeline import RagPipeline


def test_pipeline_returns_query_trace() -> None:
    chunks = [
        Chunk(
            id="hybrid",
            document_id="doc1",
            text="Hybrid retrieval combines BM25 and vector search.",
            metadata={"source_path": "memory.md"},
        )
    ]

    answer, trace = RagPipeline(chunks).answer_for_user_with_trace("hybrid retrieval", top_k=1)

    assert answer.citations
    assert trace.original_query == "hybrid retrieval"
    assert trace.normalized_query == "hybrid retrieval"
    assert trace.retrieved[0].chunk_id == "hybrid"
    assert trace.reranked[0].chunk_id == "hybrid"
    assert trace.final_context[0].chunk_id == "hybrid"
    assert trace.final_context[0].source_path == "memory.md"


def test_format_query_trace_includes_stage_summaries() -> None:
    chunks = [
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
            heading_path=("Service Dependency Notes",),
        )
    ]
    _answer, trace = RagPipeline(chunks).answer_for_user_with_trace("AUTH-429", top_k=1)

    formatted = format_query_trace(trace)

    assert "Trace" in formatted
    assert "Retrieved" in formatted
    assert "Reranked" in formatted
    assert "Final context" in formatted
    assert "chunk=policy" in formatted
    assert "Service Dependency Notes" in formatted
