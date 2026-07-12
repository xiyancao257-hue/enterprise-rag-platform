from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.config import AppConfig
from enterprise_rag.evaluation.retrieval_eval import load_retrieval_eval_cases, run_retrieval_eval
from enterprise_rag.models import Chunk
from enterprise_rag.observability.log_analysis import LogAnalysisReport, analyze_query_log


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    index_present: bool
    chunk_count: int
    eval_present: bool
    eval_case_count: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    query_log_present: bool
    log_analysis: LogAnalysisReport | None
    self_healing_draft_present: bool
    self_healing_suggestions_present: bool
    enterprise_checks: tuple[ReadinessCheck, ...]
    recommendations: tuple[str, ...]


def build_readiness_report(
    chunks: list[Chunk],
    index_path: Path,
    eval_path: Path | None = None,
    query_log_path: Path | None = None,
    self_healing_dir: Path | None = None,
    config: AppConfig | None = None,
    k: int = 5,
) -> ReadinessReport:
    eval_present = eval_path is not None and eval_path.exists()
    eval_case_count = 0
    recall_at_k = None
    precision_at_k = None
    mrr = None

    if eval_present and eval_path is not None:
        cases = load_retrieval_eval_cases(eval_path, chunks)
        eval_case_count = len(cases)
        eval_report = run_retrieval_eval(cases, chunks, k=k)
        recall_at_k = eval_report.recall_at_k
        precision_at_k = eval_report.precision_at_k
        mrr = eval_report.mrr

    query_log_present = query_log_path is not None and query_log_path.exists()
    log_analysis = analyze_query_log(query_log_path) if query_log_present and query_log_path is not None else None

    draft_present = _artifact_present(self_healing_dir, "generated_from_logs.json")
    suggestions_present = _artifact_present(self_healing_dir, "generated_with_suggestions.json")
    enterprise_checks = _enterprise_checks(
        config or AppConfig(),
        chunk_count=len(chunks),
        eval_present=eval_present,
        eval_case_count=eval_case_count,
        query_log_present=query_log_present,
        self_healing_suggestions_present=suggestions_present,
    )

    recommendations = _recommendations(
        chunk_count=len(chunks),
        eval_present=eval_present,
        recall_at_k=recall_at_k,
        query_log_present=query_log_present,
        log_analysis=log_analysis,
        self_healing_suggestions_present=suggestions_present,
        enterprise_checks=enterprise_checks,
    )

    return ReadinessReport(
        index_present=index_path.exists(),
        chunk_count=len(chunks),
        eval_present=eval_present,
        eval_case_count=eval_case_count,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mrr=mrr,
        query_log_present=query_log_present,
        log_analysis=log_analysis,
        self_healing_draft_present=draft_present,
        self_healing_suggestions_present=suggestions_present,
        enterprise_checks=enterprise_checks,
        recommendations=recommendations,
    )


def format_readiness_report(report: ReadinessReport, k: int = 5) -> str:
    lines = [
        "Readiness Report",
        f"- index: {_present(report.index_present)}",
        f"- chunks: {report.chunk_count}",
        f"- eval file: {_present(report.eval_present)}",
        f"- eval cases: {report.eval_case_count}",
        f"- Recall@{k}: {_metric(report.recall_at_k)}",
        f"- Precision@{k}: {_metric(report.precision_at_k)}",
        f"- MRR: {_metric(report.mrr)}",
        f"- query log: {_present(report.query_log_present)}",
    ]
    if report.log_analysis is not None:
        lines.extend(
            [
                f"- logged queries: {report.log_analysis.total_queries}",
                f"- insufficient evidence: {report.log_analysis.insufficient_evidence_count}",
            ]
        )
    lines.extend(
        [
            f"- self-healing draft: {_present(report.self_healing_draft_present)}",
            f"- self-healing suggestions: {_present(report.self_healing_suggestions_present)}",
            "",
            "Enterprise Checks",
        ]
    )
    lines.extend(f"- {check.name}: {check.status} - {check.detail}" for check in report.enterprise_checks)
    lines.extend(
        [
            "",
            "Recommendations",
        ]
    )
    if report.recommendations:
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    else:
        lines.append("- System looks ready for the current local demo scope.")
    return "\n".join(lines)


def count_eval_cases(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Eval file must contain a JSON array.")
    return len(payload)


def _artifact_present(directory: Path | None, filename: str) -> bool:
    return directory is not None and (directory / filename).exists()


def _present(value: bool) -> str:
    return "present" if value else "missing"


def _metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _recommendations(
    chunk_count: int,
    eval_present: bool,
    recall_at_k: float | None,
    query_log_present: bool,
    log_analysis: LogAnalysisReport | None,
    self_healing_suggestions_present: bool,
    enterprise_checks: tuple[ReadinessCheck, ...],
) -> tuple[str, ...]:
    recommendations = []
    if chunk_count == 0:
        recommendations.append("Run ingestion before demo or deployment.")
    if not eval_present:
        recommendations.append("Add or provide a retrieval eval file.")
    if recall_at_k is not None and recall_at_k < 0.9:
        recommendations.append("Improve retrieval recall before treating this system as production-ready.")
    if not query_log_present:
        recommendations.append("Enable query logging to support self-healing analysis.")
    if log_analysis is not None and log_analysis.insufficient_evidence_count > 0:
        recommendations.append("Review insufficient-evidence queries and turn them into regression eval cases.")
    if not self_healing_suggestions_present:
        recommendations.append("Run self-healing-report to generate evidence suggestions for failed queries.")
    if any(check.status == "fail" for check in enterprise_checks):
        recommendations.append("Resolve failing enterprise readiness checks before production rollout.")
    if any(check.status == "warn" for check in enterprise_checks):
        recommendations.append("Review warning enterprise readiness checks before scaling traffic.")
    recommendations.append("Run pytest before demo or deployment.")
    return tuple(recommendations)


def _enterprise_checks(
    config: AppConfig,
    chunk_count: int,
    eval_present: bool,
    eval_case_count: int,
    query_log_present: bool,
    self_healing_suggestions_present: bool,
) -> tuple[ReadinessCheck, ...]:
    return (
        _check(
            "index",
            "pass" if chunk_count > 0 else "fail",
            f"{chunk_count} chunks indexed" if chunk_count > 0 else "no chunks indexed",
        ),
        _check(
            "api_auth",
            "pass" if config.api_security.require_api_key else "warn",
            "API key required" if config.api_security.require_api_key else "API key is not required",
        ),
        _check(
            "audit_logging",
            "pass" if config.audit.enabled else "warn",
            f"audit log path: {config.audit.path}" if config.audit.enabled else "audit logging disabled",
        ),
        _check(
            "vector_index",
            "pass" if config.vector_index.provider != "memory" else "warn",
            f"provider={config.vector_index.provider}, collection={config.vector_index.collection_name}",
        ),
        _check(
            "cache",
            "pass" if config.cache.provider != "memory" else "warn",
            f"provider={config.cache.provider}, query_ttl={config.cache.query_ttl_seconds}s",
        ),
        _check(
            "leases",
            "pass" if config.leases.provider != "memory" else "warn",
            f"provider={config.leases.provider}",
        ),
        _check(
            "provider_resilience",
            "pass" if _provider_resilience_enabled(config) else "warn",
            (
                "LLM/embedding retries or circuit breakers configured"
                if _provider_resilience_enabled(config)
                else "LLM/embedding provider resilience disabled"
            ),
        ),
        _check(
            "eval_coverage",
            "pass" if eval_present and eval_case_count > 0 else "fail",
            f"{eval_case_count} eval cases" if eval_present else "retrieval eval file missing",
        ),
        _check(
            "query_logging",
            "pass" if query_log_present else "warn",
            "query log present" if query_log_present else "query log missing",
        ),
        _check(
            "self_healing",
            "pass" if self_healing_suggestions_present else "warn",
            "evidence suggestions present" if self_healing_suggestions_present else "self-healing suggestions missing",
        ),
        _check(
            "ingestion_policy",
            "pass" if config.ingestion.allowed_source_roots else "warn",
            (
                f"{len(config.ingestion.allowed_source_roots)} allowed roots, "
                f"{len(config.ingestion.allowed_extensions)} extensions"
            )
            if config.ingestion.allowed_source_roots
            else "allowed_source_roots is empty",
        ),
    )


def _provider_resilience_enabled(config: AppConfig) -> bool:
    return any(
        value > 0
        for value in (
            config.llm.max_retries,
            config.llm.circuit_breaker_failure_threshold,
            config.embedding.max_retries,
            config.embedding.circuit_breaker_failure_threshold,
        )
    )


def _check(name: str, status: str, detail: str) -> ReadinessCheck:
    return ReadinessCheck(name=name, status=status, detail=detail)
