from __future__ import annotations

from enterprise_rag.graph.entity_extraction import RuleBasedEntityExtractor
from enterprise_rag.graph.knowledge_graph import KnowledgeGraph
from enterprise_rag.models import SearchHit


class GraphRetriever:
    def __init__(
        self,
        graph: KnowledgeGraph,
        extractor: RuleBasedEntityExtractor | None = None,
        max_hops: int = 2,
    ) -> None:
        self.graph = graph
        self.extractor = extractor or RuleBasedEntityExtractor()
        self.max_hops = max_hops

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query_entities = self.extractor.extract(query)
        expanded_entities = set(query_entities)
        for entity in query_entities:
            expanded_entities.update(self.graph.related_entities(entity, max_hops=self.max_hops))

        chunks = self.graph.chunks_for_entities(expanded_entities)
        return [
            SearchHit(
                chunk=chunk,
                score=1 / rank,
                retriever="graph",
                rank=rank,
            )
            for rank, chunk in enumerate(chunks[:top_k], start=1)
        ]

