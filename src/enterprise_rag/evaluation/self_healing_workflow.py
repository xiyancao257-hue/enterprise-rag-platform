from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.evaluation.eval_generation import generate_eval_cases_from_logs, write_generated_eval_cases
from enterprise_rag.evaluation.evidence_suggestion import suggest_evidence_for_eval_draft
from enterprise_rag.models import Chunk
from enterprise_rag.observability.log_analysis import LogAnalysisReport, analyze_query_log


@dataclass(frozen=True)
class SelfHealingWorkflowReport:
    analysis: LogAnalysisReport
    draft_path: Path
    suggestions_path: Path
    generated_case_count: int


def run_self_healing_workflow(
    log_path: Path,
    chunks: list[Chunk],
    workdir: Path,
    limit: int = 20,
    suggestion_top_k: int = 3,
) -> SelfHealingWorkflowReport:
    workdir.mkdir(parents=True, exist_ok=True)
    draft_path = workdir / "generated_from_logs.json"
    suggestions_path = workdir / "generated_with_suggestions.json"

    analysis = analyze_query_log(log_path)
    cases = generate_eval_cases_from_logs(log_path, limit=limit)
    write_generated_eval_cases(cases, draft_path)
    suggest_evidence_for_eval_draft(draft_path, chunks, suggestions_path, top_k=suggestion_top_k)

    return SelfHealingWorkflowReport(
        analysis=analysis,
        draft_path=draft_path,
        suggestions_path=suggestions_path,
        generated_case_count=len(cases),
    )


def format_self_healing_workflow_report(report: SelfHealingWorkflowReport) -> str:
    lines = [
        "Self-Healing Workflow Report",
        f"- total queries: {report.analysis.total_queries}",
        f"- insufficient evidence: {report.analysis.insufficient_evidence_count}",
        f"- no retrieval hits: {report.analysis.no_retrieval_hits_count}",
        f"- no final context: {report.analysis.no_final_context_count}",
        f"- generated draft cases: {report.generated_case_count}",
        f"- draft path: {report.draft_path}",
        f"- suggestions path: {report.suggestions_path}",
        "",
        "Human review next",
        f"- Open {report.suggestions_path}",
        "- Choose suggested evidence for each useful case.",
        "- Run approve-suggested-evidence for approved suggestions.",
        "- Run promote-eval-draft to create a formal regression eval file.",
        "- Run eval on the promoted regression eval file.",
    ]
    return "\n".join(lines)
