from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enterprise_rag.evaluation.retrieval_eval import RetrievalCaseResult, RetrievalEvalReport


@dataclass(frozen=True)
class DiagnosticFinding:
    case_id: str
    category: str
    message: str
    suggestion: str


class RetrievalDiagnostics:
    def diagnose(self, report: RetrievalEvalReport) -> list[DiagnosticFinding]:
        findings = []
        for result in report.case_results:
            findings.extend(self._diagnose_case(result))
        return findings

    def _diagnose_case(self, result: RetrievalCaseResult) -> list[DiagnosticFinding]:
        findings = []
        for warning in result.warnings:
            findings.append(
                DiagnosticFinding(
                    case_id=result.case_id,
                    category="eval_data_or_ingestion",
                    message=warning,
                    suggestion="Check the eval label text, source document ingestion, cleaning, and chunking output.",
                )
            )

        if result.warnings:
            return findings

        if not result.expected_chunk_ids:
            findings.append(
                DiagnosticFinding(
                    case_id=result.case_id,
                    category="eval_data",
                    message="No expected chunks were configured for this eval case.",
                    suggestion="Add expected_chunk_ids or expected_text_contains for this query.",
                )
            )
            return findings

        if result.recall_at_k == 0:
            findings.append(
                DiagnosticFinding(
                    case_id=result.case_id,
                    category="retrieval_recall",
                    message="Expected evidence exists in the index but was not retrieved in the top-k results.",
                    suggestion=(
                        "Review query rewriting, BM25/vector retrieval, fusion, metadata filters, or increase k."
                    ),
                )
            )
        elif result.reciprocal_rank < 1:
            findings.append(
                DiagnosticFinding(
                    case_id=result.case_id,
                    category="ranking",
                    message="Expected evidence was retrieved, but it was not ranked first.",
                    suggestion="Review reranking, score fusion, heading boosts, or chunk quality.",
                )
            )
        return findings


def format_diagnostics(findings: list[DiagnosticFinding]) -> str:
    lines = ["Diagnostics:"]
    if not findings:
        lines.append("- none")
        return "\n".join(lines)
    for finding in findings:
        lines.extend(
            [
                f"- {finding.case_id} [{finding.category}]",
                f"  issue: {finding.message}",
                f"  suggestion: {finding.suggestion}",
            ]
        )
    return "\n".join(lines)
