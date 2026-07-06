from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from enterprise_rag.embeddings.hashing import HashingEmbeddingModel
from enterprise_rag.rag.compression import ContextCompressor
from enterprise_rag.retrieval.bm25 import BM25Retriever
from enterprise_rag.retrieval.filters import MetadataFilter
from enterprise_rag.retrieval.graph import GraphRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.rerank import LightweightReranker
from enterprise_rag.retrieval.vector import HashingVectorRetriever
from enterprise_rag.vector_index.in_memory import InMemoryVectorIndex


def make_chunk(chunk_id: str, text: str, heading_path: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        text=text,
        heading_path=heading_path,
        metadata={"source_path": "memory.md"},
    )


def test_metadata_filter_matches_chunks_and_hits() -> None:
    public_chunk = Chunk(id="public", document_id="doc1", text="Public policy.", metadata={"department": "security"})
    private_chunk = Chunk(id="private", document_id="doc1", text="Private policy.", metadata={"department": "finance"})
    metadata_filter = MetadataFilter({"department": "security"})

    assert metadata_filter.apply_chunks([public_chunk, private_chunk]) == [public_chunk]
    assert metadata_filter.apply_hits(
        [
            SearchHit(chunk=public_chunk, score=1.0, retriever="test", rank=1),
            SearchHit(chunk=private_chunk, score=0.5, retriever="test", rank=2),
        ]
    )[0].chunk.id == "public"


def test_metadata_filter_enforces_allowed_groups_acl() -> None:
    public_chunk = Chunk(id="public", document_id="doc1", text="Public policy.")
    security_chunk = Chunk(
        id="security",
        document_id="doc1",
        text="Security policy.",
        metadata={"allowed_groups": "security,platform"},
    )
    finance_chunk = Chunk(
        id="finance",
        document_id="doc1",
        text="Finance policy.",
        metadata={"allowed_groups": "finance"},
    )

    metadata_filter = MetadataFilter(user_groups={"security"})

    assert metadata_filter.apply_chunks([public_chunk, security_chunk, finance_chunk]) == [public_chunk, security_chunk]


def test_bm25_retriever_finds_exact_keyword_match() -> None:
    chunks = [
        make_chunk("auth", "Error code AUTH-429 means the authentication service rate limit was exceeded."),
        make_chunk("rag", "Hybrid retrieval combines BM25 keyword search with vector search."),
    ]

    hits = BM25Retriever(chunks).search("AUTH-429", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk.id == "auth"
    assert hits[0].retriever == "bm25"
    assert hits[0].rank == 1


def test_vector_retriever_returns_ranked_hits() -> None:
    chunks = [
        make_chunk("rag", "Hybrid retrieval combines BM25 keyword search with vector search."),
        make_chunk("cleaning", "Dirty data cleaning removes repeated headers and OCR noise."),
    ]

    hits = HashingVectorRetriever(chunks).search("hybrid retrieval", top_k=2)

    assert hits
    assert hits[0].retriever == "vector"
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_vector_retriever_accepts_embedding_model_interface() -> None:
    chunks = [
        make_chunk("rag", "Hybrid retrieval combines BM25 keyword search with vector search."),
    ]

    hits = HashingVectorRetriever(chunks, embedding_model=HashingEmbeddingModel(dimensions=32)).search("hybrid retrieval")

    assert hits[0].chunk.id == "rag"


def test_vector_retriever_accepts_vector_index_interface() -> None:
    chunks = [
        make_chunk("rag", "Hybrid retrieval combines BM25 keyword search with vector search."),
    ]

    hits = HashingVectorRetriever(chunks, vector_index=InMemoryVectorIndex()).search("hybrid retrieval")

    assert hits[0].chunk.id == "rag"


def test_hybrid_retriever_fuses_bm25_and_vector_hits() -> None:
    chunks = [
        make_chunk("rag", "Hybrid retrieval combines BM25 keyword search with vector search."),
        make_chunk("cleaning", "Dirty data cleaning removes repeated headers and OCR noise."),
    ]

    hits = HybridRetriever(chunks).search(["hybrid retrieval"], top_k=2)

    assert hits
    assert hits[0].chunk.id == "rag"
    assert "bm25" in hits[0].retriever
    assert "vector" in hits[0].retriever


def test_hybrid_retriever_can_include_graph_hits() -> None:
    chunks = [
        make_chunk("product", "Product Atlas depends on Auth Service."),
        make_chunk("service", "Auth Service uses Rate Limit Policy."),
        make_chunk("policy", "Rate Limit Policy defines AUTH-429."),
    ]
    graph = KnowledgeGraphBuilder().build(chunks)
    graph_retriever = GraphRetriever(graph, max_hops=3)

    hits = HybridRetriever(chunks, extra_retrievers=[graph_retriever]).search(
        ["Which product is affected by AUTH-429?"],
        top_k=3,
    )

    assert {hit.chunk.id for hit in hits} == {"policy", "service", "product"}
    assert any("graph" in hit.retriever for hit in hits)


def test_hybrid_retriever_applies_metadata_filters() -> None:
    chunks = [
        Chunk(id="security", document_id="doc1", text="Retention policy for security logs.", metadata={"department": "security"}),
        Chunk(id="finance", document_id="doc1", text="Retention policy for finance records.", metadata={"department": "finance"}),
    ]

    hits = HybridRetriever(chunks).search(["retention policy"], top_k=5, metadata_filters={"department": "security"})

    assert hits
    assert {hit.chunk.id for hit in hits} == {"security"}


def test_hybrid_retriever_applies_acl_filters() -> None:
    chunks = [
        Chunk(id="security", document_id="doc1", text="Retention policy for security logs.", metadata={"allowed_groups": "security"}),
        Chunk(id="finance", document_id="doc1", text="Retention policy for finance records.", metadata={"allowed_groups": "finance"}),
    ]

    hits = HybridRetriever(chunks).search(["retention policy"], top_k=5, user_groups={"security"})

    assert hits
    assert {hit.chunk.id for hit in hits} == {"security"}


def test_reranker_promotes_query_overlap_and_heading_match() -> None:
    weak_hit = SearchHit(
        chunk=make_chunk("weak", "General notes about enterprise document ingestion."),
        score=0.20,
        retriever="vector",
        rank=1,
    )
    strong_hit = SearchHit(
        chunk=make_chunk(
            "strong",
            "BM25 and vector search are combined for hybrid retrieval.",
            heading_path=("Enterprise RAG", "Hybrid Retrieval"),
        ),
        score=0.10,
        retriever="bm25",
        rank=2,
    )

    hits = LightweightReranker().rerank("hybrid retrieval", [weak_hit, strong_hit], top_k=2)

    assert hits[0].chunk.id == "strong"
    assert hits[0].retriever == "bm25+rerank"
    assert hits[0].rank == 1


def test_context_compressor_keeps_query_relevant_sentences() -> None:
    hit = SearchHit(
        chunk=make_chunk(
            "rag",
            (
                "Hybrid retrieval combines BM25 keyword search with vector search. "
                "Dirty data cleaning removes repeated headers and OCR noise. "
                "BM25 is useful for exact terms, product names, and error codes."
            ),
        ),
        score=0.5,
        retriever="bm25+vector+rerank",
        rank=1,
    )

    compressed = ContextCompressor().compress("hybrid retrieval BM25", [hit], max_sentences_per_hit=2)

    assert len(compressed) == 1
    assert "Hybrid retrieval combines BM25" in compressed[0].chunk.text
    assert "BM25 is useful for exact terms" in compressed[0].chunk.text
    assert "Dirty data cleaning" not in compressed[0].chunk.text
    assert compressed[0].chunk.id == "rag"
    assert compressed[0].retriever == "bm25+vector+rerank"
