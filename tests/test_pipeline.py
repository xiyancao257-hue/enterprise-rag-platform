from enterprise_rag.processing.chunking import StructureAwareChunker
from enterprise_rag.processing.cleaning import DirtyDataCleaner
from enterprise_rag.processing.parser import StructureParser
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.models import Chunk, Document


def test_pipeline_retrieves_hybrid_context() -> None:
    document = Document(
        id="doc1",
        source_path="memory.md",
        text=(
            "# Enterprise RAG\n\n"
            "Hybrid retrieval combines BM25 keyword search with vector search.\n\n"
            "Reranking improves precision after broad recall.\n"
        ),
    )
    cleaned = DirtyDataCleaner().clean(document)
    assert cleaned is not None
    blocks = StructureParser().parse(cleaned)
    chunks = StructureAwareChunker(target_tokens=20, max_tokens=40).chunk(blocks)

    answer = RagPipeline(chunks).answer("hybrid retrival", top_k=2)

    assert answer.citations
    assert answer.query_plan.corrections["retrival"] == "retrieval"
    assert "BM25" in answer.answer


def test_pipeline_can_enable_graph_retrieval() -> None:
    chunks = [
        Chunk(
            id="product",
            document_id="doc1",
            text="Product Atlas depends on Auth Service.",
        ),
        Chunk(
            id="service",
            document_id="doc1",
            text="Auth Service uses Rate Limit Policy.",
        ),
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
        ),
    ]

    answer = RagPipeline(chunks, enable_graph=True, graph_max_hops=3).answer(
        "Which product is affected by AUTH-429?",
        top_k=3,
    )

    assert {hit.chunk.id for hit in answer.citations} == {"product", "service", "policy"}
    assert any("graph" in hit.retriever for hit in answer.citations)


def test_pipeline_applies_query_metadata_filters() -> None:
    chunks = [
        Chunk(
            id="security",
            document_id="doc1",
            text="Retention policy for security logs.",
            metadata={"department": "security"},
        ),
        Chunk(
            id="finance",
            document_id="doc1",
            text="Retention policy for finance records.",
            metadata={"department": "finance"},
        ),
    ]

    answer = RagPipeline(chunks).answer("department:security retention policy", top_k=5)

    assert answer.query_plan.metadata_filters == {"department": "security"}
    assert {hit.chunk.id for hit in answer.citations} == {"security"}


def test_pipeline_applies_user_group_acl_filters() -> None:
    chunks = [
        Chunk(
            id="security",
            document_id="doc1",
            text="Retention policy for security logs.",
            metadata={"allowed_groups": "security"},
        ),
        Chunk(
            id="finance",
            document_id="doc1",
            text="Retention policy for finance records.",
            metadata={"allowed_groups": "finance"},
        ),
    ]

    answer = RagPipeline(chunks).answer_for_user("retention policy", top_k=5, user_groups={"security"})

    assert {hit.chunk.id for hit in answer.citations} == {"security"}
