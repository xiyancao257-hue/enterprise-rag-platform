import json

from enterprise_rag.evaluation.self_healing_workflow import (
    format_self_healing_workflow_report,
    run_self_healing_workflow,
)
from enterprise_rag.models import Chunk


def test_run_self_healing_workflow_generates_draft_and_suggestions(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    workdir = tmp_path / "self_healing"
    records = [
        {
            "query": "What is AUTH-429?",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
            "enable_graph": False,
        },
        {
            "query": "working query",
            "retrieved_chunk_ids": ["chunk1"],
            "final_chunk_ids": ["chunk1"],
            "insufficient_evidence": False,
            "enable_graph": True,
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    chunks = [
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
            metadata={"source_path": "data/raw/kg_demo.md"},
        )
    ]

    report = run_self_healing_workflow(log_path, chunks, workdir, limit=10, suggestion_top_k=1)

    assert report.analysis.total_queries == 2
    assert report.analysis.insufficient_evidence_count == 1
    assert report.generated_case_count == 1
    assert report.draft_path.exists()
    assert report.suggestions_path.exists()

    draft_cases = json.loads(report.draft_path.read_text(encoding="utf-8"))
    suggestion_cases = json.loads(report.suggestions_path.read_text(encoding="utf-8"))
    assert draft_cases[0]["query"] == "What is AUTH-429?"
    assert suggestion_cases[0]["suggested_evidence"][0]["chunk_id"] == "policy"


def test_format_self_healing_workflow_report_includes_human_review_steps(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    workdir = tmp_path / "self_healing"
    log_path.write_text(
        json.dumps(
            {
                "query": "missing query",
                "retrieved_chunk_ids": [],
                "final_chunk_ids": [],
                "insufficient_evidence": True,
                "enable_graph": False,
            }
        ),
        encoding="utf-8",
    )

    report = run_self_healing_workflow(log_path, [], workdir)
    formatted = format_self_healing_workflow_report(report)

    assert "Self-Healing Workflow Report" in formatted
    assert "- generated draft cases: 1" in formatted
    assert "Human review next" in formatted
    assert "approve-suggested-evidence" in formatted
    assert "promote-eval-draft" in formatted
