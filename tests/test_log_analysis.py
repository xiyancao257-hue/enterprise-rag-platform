import json

from enterprise_rag.observability.log_analysis import analyze_query_log, format_log_analysis_report


def test_analyze_query_log_counts_failure_patterns(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    records = [
        {
            "query": "What does AUTH-429 affect?",
            "retrieved_chunk_ids": ["policy"],
            "final_chunk_ids": ["policy"],
            "insufficient_evidence": False,
            "enable_graph": True,
        },
        {
            "query": "missing procedure",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
            "enable_graph": False,
        },
        {
            "query": "compressed away",
            "retrieved_chunk_ids": ["chunk1"],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
            "enable_graph": False,
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    report = analyze_query_log(log_path)

    assert report.total_queries == 3
    assert report.insufficient_evidence_count == 2
    assert report.no_retrieval_hits_count == 1
    assert report.no_final_context_count == 2
    assert report.graph_disabled_count == 2
    assert report.candidate_eval_queries == ("missing procedure", "compressed away")
    assert "Add insufficient-evidence queries" in report.recommendations[0]
    assert any("graph retrieval" in recommendation for recommendation in report.recommendations)


def test_format_log_analysis_report_outputs_summary() -> None:
    from enterprise_rag.observability.log_analysis import LogAnalysisReport

    report = LogAnalysisReport(
        total_queries=2,
        insufficient_evidence_count=1,
        no_retrieval_hits_count=1,
        no_final_context_count=0,
        graph_disabled_count=1,
        candidate_eval_queries=("missing policy",),
        recommendations=("Inspect ingestion, chunking, and metadata filters for no-hit queries.",),
    )

    formatted = format_log_analysis_report(report)

    assert "Log Analysis" in formatted
    assert "- total queries: 2" in formatted
    assert "Candidate eval queries" in formatted
    assert "- missing policy" in formatted
    assert "Recommendations" in formatted
