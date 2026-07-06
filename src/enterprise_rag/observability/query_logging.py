from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from enterprise_rag.models import RagAnswer
from enterprise_rag.observability.tracing import QueryTrace
from enterprise_rag.rag.answer_generation import INSUFFICIENT_EVIDENCE_MESSAGE


@dataclass(frozen=True)
class QueryLogRecord:
    timestamp: str
    query: str
    normalized_query: str
    rewritten_queries: tuple[str, ...]
    metadata_filters: dict[str, str]
    top_k: int
    enable_graph: bool
    graph_max_hops: int
    user_groups: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    final_chunk_ids: tuple[str, ...]
    insufficient_evidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "rewritten_queries": list(self.rewritten_queries),
            "metadata_filters": self.metadata_filters,
            "top_k": self.top_k,
            "enable_graph": self.enable_graph,
            "graph_max_hops": self.graph_max_hops,
            "user_groups": list(self.user_groups),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "final_chunk_ids": list(self.final_chunk_ids),
            "insufficient_evidence": self.insufficient_evidence,
        }


class QueryLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, record: QueryLogRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def build_query_log_record(
    answer: RagAnswer,
    trace: QueryTrace,
    top_k: int,
    enable_graph: bool,
    graph_max_hops: int,
    user_groups: set[str] | None = None,
) -> QueryLogRecord:
    return QueryLogRecord(
        timestamp=datetime.now(UTC).isoformat(),
        query=answer.query,
        normalized_query=trace.normalized_query,
        rewritten_queries=trace.rewritten_queries,
        metadata_filters=trace.metadata_filters,
        top_k=top_k,
        enable_graph=enable_graph,
        graph_max_hops=graph_max_hops,
        user_groups=tuple(sorted(user_groups or set())),
        retrieved_chunk_ids=tuple(hit.chunk_id for hit in trace.retrieved),
        final_chunk_ids=tuple(hit.chunk_id for hit in trace.final_context),
        insufficient_evidence=answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE,
    )
