from enterprise_rag.graph.entity_extraction import RuleBasedEntityExtractor
from enterprise_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from enterprise_rag.models import Chunk
from enterprise_rag.retrieval.graph import GraphRetriever


def test_rule_based_entity_extractor_finds_codes_acronyms_and_title_case_entities() -> None:
    extractor = RuleBasedEntityExtractor()

    entities = extractor.extract("Product Atlas depends on Auth Service. Rate Limit Policy defines AUTH-429 for RAG.")

    assert "Product Atlas" in entities
    assert "Auth Service" in entities
    assert "Rate Limit Policy" in entities
    assert "AUTH-429" in entities
    assert "RAG" in entities


def test_knowledge_graph_builder_extracts_relationships_and_entity_chunk_index() -> None:
    chunks = [
        Chunk(id="product", document_id="doc1", text="Product Atlas depends on Auth Service."),
        Chunk(id="service", document_id="doc1", text="Auth Service uses Rate Limit Policy."),
        Chunk(id="policy", document_id="doc1", text="Rate Limit Policy defines AUTH-429."),
    ]

    graph = KnowledgeGraphBuilder().build(chunks)

    assert "product" in graph.entity_to_chunks["product atlas"]
    assert "service" in graph.entity_to_chunks["auth service"]
    assert "policy" in graph.entity_to_chunks["auth-429"]
    assert ("Product Atlas", "depends_on", "Auth Service", "product") in {
        (rel.source, rel.predicate, rel.target, rel.chunk_id) for rel in graph.relationships
    }
    assert "auth service" in graph.related_entities("AUTH-429", max_hops=2)


def test_graph_retriever_expands_query_entities_to_related_chunks() -> None:
    chunks = [
        Chunk(id="product", document_id="doc1", text="Product Atlas depends on Auth Service."),
        Chunk(id="service", document_id="doc1", text="Auth Service uses Rate Limit Policy."),
        Chunk(id="policy", document_id="doc1", text="Rate Limit Policy defines AUTH-429."),
    ]
    graph = KnowledgeGraphBuilder().build(chunks)

    hits = GraphRetriever(graph, max_hops=2).search("Which product is affected by AUTH-429?", top_k=10)

    hit_ids = {hit.chunk.id for hit in hits}
    assert {"policy", "service", "product"} <= hit_ids
    assert all(hit.retriever == "graph" for hit in hits)
