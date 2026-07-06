from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from enterprise_rag.evaluation.diagnostics import RetrievalDiagnostics, format_diagnostics
from enterprise_rag.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from enterprise_rag.graph.knowledge_graph import KnowledgeGraphBuilder
from enterprise_rag.models import Chunk, SearchHit
from enterprise_rag.query.engine import QueryEngine
from enterprise_rag.retrieval.graph import GraphRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.text import tokenize


@dataclass(frozen=True)
class RetrievalEvalCase:
    id: str
    query: str
    expected_chunk_ids: set[str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    query: str
    expected_chunk_ids: set[str]
    retrieved_chunk_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.recall_at_k > 0


@dataclass(frozen=True)
class RetrievalEvalReport:
    k: int
    case_results: list[RetrievalCaseResult]

    @property
    def recall_at_k(self) -> float:
        return self._average([result.recall_at_k for result in self.case_results])

    @property
    def precision_at_k(self) -> float:
        return self._average([result.precision_at_k for result in self.case_results])

    @property
    def mrr(self) -> float:
        return self._average([result.reciprocal_rank for result in self.case_results])

    @property
    def failures(self) -> list[RetrievalCaseResult]:
        return [result for result in self.case_results if not result.passed]

    def _average(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)


def evaluate_retrieval_results(
    cases: list[RetrievalEvalCase],
    retrieved_by_case_id: dict[str, list[SearchHit]],
    k: int = 5,
) -> RetrievalEvalReport:
    case_results = []
    for case in cases:
        hits = retrieved_by_case_id.get(case.id, [])
        retrieved_chunk_ids = [hit.chunk.id for hit in hits]
        case_results.append(
            RetrievalCaseResult(
                case_id=case.id,
                query=case.query,
                expected_chunk_ids=case.expected_chunk_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                recall_at_k=recall_at_k(case.expected_chunk_ids, retrieved_chunk_ids, k),
                precision_at_k=precision_at_k(case.expected_chunk_ids, retrieved_chunk_ids, k),
                reciprocal_rank=reciprocal_rank(case.expected_chunk_ids, retrieved_chunk_ids),
                warnings=case.warnings,
            )
        )
    return RetrievalEvalReport(k=k, case_results=case_results)


def load_retrieval_eval_cases(path: Path, chunks: list[Chunk]) -> list[RetrievalEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in payload:
        expected_chunk_ids = set(item.get("expected_chunk_ids", []))
        warnings = []
        for expected_text in item.get("expected_text_contains", []):
            matched_chunk_ids = [chunk.id for chunk in chunks if expected_text.lower() in chunk.text.lower()]
            if not matched_chunk_ids:
                warnings.append(f"expected_text_contains did not match any chunk: {expected_text}")
            expected_chunk_ids.update(matched_chunk_ids)
        cases.append(
            RetrievalEvalCase(
                id=item["id"],
                query=item["query"],
                expected_chunk_ids=expected_chunk_ids,
                warnings=tuple(warnings),
            )
        )
    return cases


def run_retrieval_eval(
    cases: list[RetrievalEvalCase],
    chunks: list[Chunk],
    k: int = 5,
    enable_graph: bool = False,
    graph_max_hops: int = 2,
) -> RetrievalEvalReport:
    vocabulary = {token for chunk in chunks for token in tokenize(chunk.text)}
    query_engine = QueryEngine(vocabulary=vocabulary)
    extra_retrievers = []
    if enable_graph:
        graph = KnowledgeGraphBuilder().build(chunks)
        extra_retrievers.append(GraphRetriever(graph, max_hops=graph_max_hops))
    retriever = HybridRetriever(chunks, extra_retrievers=extra_retrievers)
    retrieved_by_case_id = {}
    for case in cases:
        plan = query_engine.plan(case.query)
        retrieved_by_case_id[case.id] = retriever.search(list(plan.rewritten_queries), top_k=k)
    return evaluate_retrieval_results(cases, retrieved_by_case_id, k=k)


def format_retrieval_eval_report(report: RetrievalEvalReport) -> str:
    lines = [
        "Retrieval Eval Report",
        f"cases: {len(report.case_results)}",
        f"Recall@{report.k}: {report.recall_at_k:.2f}",
        f"Precision@{report.k}: {report.precision_at_k:.2f}",
        f"MRR: {report.mrr:.2f}",
    ]
    if any(result.warnings for result in report.case_results):
        lines.extend(["", "Warnings:"])
        for result in report.case_results:
            for warning in result.warnings:
                lines.append(f"- {result.case_id}: {warning}")
    lines.extend(["", "Failures:"])
    if not report.failures:
        lines.append("- none")
    else:
        for failure in report.failures:
            lines.extend(
                [
                    f"- {failure.case_id}",
                    f"  query: {failure.query}",
                    f"  expected: {', '.join(sorted(failure.expected_chunk_ids)) or 'none'}",
                    f"  retrieved: {', '.join(failure.retrieved_chunk_ids) or 'none'}",
                ]
            )
    lines.extend(["", format_diagnostics(RetrievalDiagnostics().diagnose(report))])
    return "\n".join(lines)
