from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LogAnalysisReport:
    total_queries: int
    insufficient_evidence_count: int
    no_retrieval_hits_count: int
    no_final_context_count: int
    graph_disabled_count: int
    candidate_eval_queries: tuple[str, ...]
    recommendations: tuple[str, ...]


def analyze_query_log(path: Path) -> LogAnalysisReport:
    records = load_query_log_records(path)
    insufficient = [record for record in records if record.get("insufficient_evidence") is True]
    no_retrieval_hits = [record for record in records if not record.get("retrieved_chunk_ids", [])]
    no_final_context = [record for record in records if not record.get("final_chunk_ids", [])]
    graph_disabled = [record for record in records if record.get("enable_graph") is False]

    candidate_queries = _candidate_eval_queries(insufficient + no_retrieval_hits + no_final_context)
    recommendations = _recommendations(insufficient, no_retrieval_hits, no_final_context, graph_disabled)

    return LogAnalysisReport(
        total_queries=len(records),
        insufficient_evidence_count=len(insufficient),
        no_retrieval_hits_count=len(no_retrieval_hits),
        no_final_context_count=len(no_final_context),
        graph_disabled_count=len(graph_disabled),
        candidate_eval_queries=candidate_queries,
        recommendations=recommendations,
    )


def load_query_log_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query log file does not exist: {path}")

    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"Log line {line_number} must contain a JSON object.")
        records.append(data)
    return records


def format_log_analysis_report(report: LogAnalysisReport) -> str:
    lines = [
        "Log Analysis",
        f"- total queries: {report.total_queries}",
        f"- insufficient evidence: {report.insufficient_evidence_count}",
        f"- no retrieval hits: {report.no_retrieval_hits_count}",
        f"- no final context: {report.no_final_context_count}",
        f"- graph disabled queries: {report.graph_disabled_count}",
    ]

    lines.append("")
    lines.append("Candidate eval queries")
    if report.candidate_eval_queries:
        lines.extend(f"- {query}" for query in report.candidate_eval_queries)
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Recommendations")
    if report.recommendations:
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    else:
        lines.append("- No obvious self-healing actions found from this log.")

    return "\n".join(lines)


def _candidate_eval_queries(records: list[dict[str, Any]], limit: int = 10) -> tuple[str, ...]:
    queries = []
    seen = set()
    for record in records:
        query = str(record.get("query", "")).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        queries.append(query)
        if len(queries) >= limit:
            break
    return tuple(queries)


def _recommendations(
    insufficient: list[dict[str, Any]],
    no_retrieval_hits: list[dict[str, Any]],
    no_final_context: list[dict[str, Any]],
    graph_disabled: list[dict[str, Any]],
) -> tuple[str, ...]:
    recommendations = []
    if insufficient:
        recommendations.append("Add insufficient-evidence queries to the retrieval eval set.")
    if no_retrieval_hits:
        recommendations.append("Inspect ingestion, chunking, and metadata filters for no-hit queries.")
    if no_final_context:
        recommendations.append("Review reranking and compression thresholds for queries with empty final context.")
    if graph_disabled:
        recommendations.append("Try graph retrieval for entity-heavy failed queries.")
    return tuple(recommendations)
