from enterprise_rag.evaluation.diagnostics import RetrievalDiagnostics, format_diagnostics
from enterprise_rag.evaluation.retrieval_eval import RetrievalCaseResult, RetrievalEvalReport


def report_with(result: RetrievalCaseResult) -> RetrievalEvalReport:
    return RetrievalEvalReport(k=3, case_results=[result])


def test_diagnostics_identifies_eval_data_or_ingestion_warning() -> None:
    result = RetrievalCaseResult(
        case_id="bad_label",
        query="unknown",
        expected_chunk_ids=set(),
        retrieved_chunk_ids=[],
        recall_at_k=0.0,
        precision_at_k=0.0,
        reciprocal_rank=0.0,
        warnings=("expected_text_contains did not match any chunk: missing text",),
    )

    findings = RetrievalDiagnostics().diagnose(report_with(result))

    assert findings[0].category == "eval_data_or_ingestion"
    assert "missing text" in findings[0].message
    assert "ingestion" in findings[0].suggestion


def test_diagnostics_identifies_missing_expected_configuration() -> None:
    result = RetrievalCaseResult(
        case_id="no_expected",
        query="unknown",
        expected_chunk_ids=set(),
        retrieved_chunk_ids=[],
        recall_at_k=0.0,
        precision_at_k=0.0,
        reciprocal_rank=0.0,
    )

    findings = RetrievalDiagnostics().diagnose(report_with(result))

    assert findings[0].category == "eval_data"
    assert "No expected chunks" in findings[0].message


def test_diagnostics_identifies_retrieval_recall_failure() -> None:
    result = RetrievalCaseResult(
        case_id="missed",
        query="hybrid retrieval",
        expected_chunk_ids={"expected"},
        retrieved_chunk_ids=["other"],
        recall_at_k=0.0,
        precision_at_k=0.0,
        reciprocal_rank=0.0,
    )

    findings = RetrievalDiagnostics().diagnose(report_with(result))

    assert findings[0].category == "retrieval_recall"
    assert "not retrieved" in findings[0].message


def test_diagnostics_identifies_ranking_issue() -> None:
    result = RetrievalCaseResult(
        case_id="low_rank",
        query="hybrid retrieval",
        expected_chunk_ids={"expected"},
        retrieved_chunk_ids=["other", "expected"],
        recall_at_k=1.0,
        precision_at_k=0.5,
        reciprocal_rank=0.5,
    )

    findings = RetrievalDiagnostics().diagnose(report_with(result))

    assert findings[0].category == "ranking"
    assert "not ranked first" in findings[0].message


def test_format_diagnostics_handles_empty_and_non_empty_findings() -> None:
    assert format_diagnostics([]) == "Diagnostics:\n- none"

    result = RetrievalCaseResult(
        case_id="missed",
        query="hybrid retrieval",
        expected_chunk_ids={"expected"},
        retrieved_chunk_ids=["other"],
        recall_at_k=0.0,
        precision_at_k=0.0,
        reciprocal_rank=0.0,
    )
    findings = RetrievalDiagnostics().diagnose(report_with(result))

    formatted = format_diagnostics(findings)

    assert "Diagnostics:" in formatted
    assert "- missed [retrieval_recall]" in formatted
    assert "suggestion:" in formatted

