from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re

from enterprise_rag.graph.entity_extraction import RuleBasedEntityExtractor
from enterprise_rag.models import Chunk


@dataclass(frozen=True)
class Relationship:
    source: str
    predicate: str
    target: str
    chunk_id: str


class KnowledgeGraph:
    def __init__(
        self,
        entity_to_chunks: dict[str, set[str]],
        relationships: list[Relationship],
        chunks_by_id: dict[str, Chunk],
    ) -> None:
        self.entity_to_chunks = entity_to_chunks
        self.relationships = relationships
        self.chunks_by_id = chunks_by_id
        self.neighbors_by_entity = self._build_neighbors(relationships)

    def related_entities(self, entity: str, max_hops: int = 1) -> set[str]:
        normalized = self._normalize(entity)
        related = set()
        queue = deque([(normalized, 0)])
        visited = {normalized}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self.neighbors_by_entity.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                related.add(neighbor)
                queue.append((neighbor, depth + 1))
        return related

    def chunks_for_entities(self, entities: set[str]) -> list[Chunk]:
        chunk_ids = set()
        for entity in entities:
            chunk_ids.update(self.entity_to_chunks.get(self._normalize(entity), set()))
        return [self.chunks_by_id[chunk_id] for chunk_id in sorted(chunk_ids) if chunk_id in self.chunks_by_id]

    def _build_neighbors(self, relationships: list[Relationship]) -> dict[str, set[str]]:
        neighbors: dict[str, set[str]] = defaultdict(set)
        for relationship in relationships:
            source = self._normalize(relationship.source)
            target = self._normalize(relationship.target)
            neighbors[source].add(target)
            neighbors[target].add(source)
        return neighbors

    def _normalize(self, entity: str) -> str:
        return " ".join(entity.lower().split())


class KnowledgeGraphBuilder:
    RELATION_PATTERNS = (
        (re.compile(r"\b(?P<source>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\s+depends on\s+(?P<target>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\b"), "depends_on"),
        (re.compile(r"\b(?P<source>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\s+uses\s+(?P<target>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\b"), "uses"),
        (re.compile(r"\b(?P<source>[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4})\s+defines\s+(?P<target>[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)?)\b"), "defines"),
    )

    def __init__(self, extractor: RuleBasedEntityExtractor | None = None) -> None:
        self.extractor = extractor or RuleBasedEntityExtractor()

    def build(self, chunks: list[Chunk]) -> KnowledgeGraph:
        entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        relationships = []

        for chunk in chunks:
            for entity in self.extractor.extract(chunk.text):
                entity_to_chunks[self.extractor.normalize(entity)].add(chunk.id)
            relationships.extend(self._extract_relationships(chunk))

        for relationship in relationships:
            entity_to_chunks[self.extractor.normalize(relationship.source)].add(relationship.chunk_id)
            entity_to_chunks[self.extractor.normalize(relationship.target)].add(relationship.chunk_id)

        return KnowledgeGraph(dict(entity_to_chunks), relationships, chunks_by_id)

    def _extract_relationships(self, chunk: Chunk) -> list[Relationship]:
        relationships = []
        for pattern, predicate in self.RELATION_PATTERNS:
            for match in pattern.finditer(chunk.text):
                relationships.append(
                    Relationship(
                        source=match.group("source").strip(),
                        predicate=predicate,
                        target=match.group("target").strip(),
                        chunk_id=chunk.id,
                    )
                )
        return relationships

