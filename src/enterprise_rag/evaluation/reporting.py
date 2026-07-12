from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.evaluation.readiness import ReadinessReport
from enterprise_rag.evaluation.retrieval_eval import RetrievalEvalReport


@dataclass(frozen=True)
class EvaluationMarkdownReport:
    title: str
    retrieval: RetrievalEvalReport
    readiness: ReadinessReport | None = None


def format_evaluation_markdown_report(report: EvaluationMarkdownReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        "## Retrieval Metrics",
        "",
        f"- Cases: {len(report.retrieval.case_results)}",
        f"- Recall@{report.retrieval.k}: {report.retrieval.recall_at_k:.2f}",
        f"- Precision@{report.retrieval.k}: {report.retrieval.precision_at_k:.2f}",
        f"- MRR: {report.retrieval.mrr:.2f}",
        f"- Failures: {len(report.retrieval.failures)}",
        "",
        "## Failed Cases",
        "",
    ]
    if not report.retrieval.failures:
        lines.append("- None")
    else:
        for failure in report.retrieval.failures:
            lines.extend(
                [
                    f"### {failure.case_id}",
                    "",
                    f"- Query: `{failure.query}`",
                    f"- Expected chunks: {', '.join(sorted(failure.expected_chunk_ids)) or 'none'}",
                    f"- Retrieved chunks: {', '.join(failure.retrieved_chunk_ids) or 'none'}",
                    f"- Recall@{report.retrieval.k}: {failure.recall_at_k:.2f}",
                    f"- Precision@{report.retrieval.k}: {failure.precision_at_k:.2f}",
                    f"- MRR contribution: {failure.reciprocal_rank:.2f}",
                    "",
                ]
            )
    warnings = [(result.case_id, warning) for result in report.retrieval.case_results for warning in result.warnings]
    lines.extend(["", "## Eval Warnings", ""])
    if not warnings:
        lines.append("- None")
    else:
        lines.extend(f"- {case_id}: {warning}" for case_id, warning in warnings)

    if report.readiness is not None:
        lines.extend(
            [
                "",
                "## Readiness Snapshot",
                "",
                f"- Chunks: {report.readiness.chunk_count}",
                f"- Eval cases: {report.readiness.eval_case_count}",
                f"- Query log present: {_yes_no(report.readiness.query_log_present)}",
                f"- Self-healing suggestions present: {_yes_no(report.readiness.self_healing_suggestions_present)}",
                "",
                "## Enterprise Checks",
                "",
            ]
        )
        lines.extend(
            f"- {check.name}: **{check.status}** - {check.detail}" for check in report.readiness.enterprise_checks
        )
        lines.extend(["", "## Recommendations", ""])
        if report.readiness.recommendations:
            lines.extend(f"- {recommendation}" for recommendation in report.readiness.recommendations)
        else:
            lines.append("- No readiness recommendations.")

    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
