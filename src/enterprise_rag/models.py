from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE_OCR = "image_ocr"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Document:
    id: str
    source_path: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentBlock:
    id: str
    document_id: str
    block_type: BlockType
    text: str
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    heading_path: tuple[str, ...] = ()
    source_blocks: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    retriever: str
    rank: int


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    rewritten_queries: tuple[str, ...]
    ambiguity_notes: tuple[str, ...] = ()
    corrections: dict[str, str] = field(default_factory=dict)
    metadata_filters: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    citations: tuple[SearchHit, ...]
    query_plan: QueryPlan
