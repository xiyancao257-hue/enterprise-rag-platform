import json

from enterprise_rag.evaluation.eval_generation import (
    generate_eval_cases_from_logs,
    promote_reviewed_eval_draft,
    write_generated_eval_cases,
)


def test_generate_eval_cases_from_failed_query_logs(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    records = [
        {
            "query": "working query",
            "retrieved_chunk_ids": ["chunk1"],
            "final_chunk_ids": ["chunk1"],
            "insufficient_evidence": False,
        },
        {
            "query": "missing escalation path",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
        },
        {
            "query": "compressed away",
            "retrieved_chunk_ids": ["chunk2"],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    cases = generate_eval_cases_from_logs(log_path)

    assert [case.query for case in cases] == ["missing escalation path", "compressed away"]
    assert cases[0].id.startswith("log_1_missing_escalation_path")
    assert cases[0].expected_text_contains == ()
    assert "manual review" in cases[0].notes


def test_generate_eval_cases_deduplicates_queries_and_respects_limit(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    records = [
        {
            "query": "same failure",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
        },
        {
            "query": "same failure",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
        },
        {
            "query": "second failure",
            "retrieved_chunk_ids": [],
            "final_chunk_ids": [],
            "insufficient_evidence": True,
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    cases = generate_eval_cases_from_logs(log_path, limit=1)

    assert len(cases) == 1
    assert cases[0].query == "same failure"


def test_write_generated_eval_cases_outputs_json(tmp_path) -> None:
    log_path = tmp_path / "query_log.jsonl"
    output_path = tmp_path / "generated_eval.json"
    log_path.write_text(
        json.dumps(
            {
                "query": "missing policy",
                "retrieved_chunk_ids": [],
                "final_chunk_ids": [],
                "insufficient_evidence": True,
            }
        ),
        encoding="utf-8",
    )
    cases = generate_eval_cases_from_logs(log_path)

    write_generated_eval_cases(cases, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == [
        {
            "id": "log_1_missing_policy",
            "query": "missing policy",
            "expected_text_contains": [],
            "notes": cases[0].notes,
        }
    ]


def test_promote_reviewed_eval_draft_keeps_only_cases_with_expected_evidence(tmp_path) -> None:
    draft_path = tmp_path / "generated_eval.json"
    output_path = tmp_path / "regression_eval.json"
    draft_path.write_text(
        json.dumps(
            [
                {
                    "id": "unreviewed",
                    "query": "missing escalation path",
                    "expected_text_contains": [],
                    "notes": "needs review",
                },
                {
                    "id": "reviewed_text",
                    "query": "What is AUTH-429?",
                    "expected_text_contains": ["Rate Limit Policy defines AUTH-429."],
                    "notes": "reviewed",
                },
                {
                    "id": "reviewed_chunk",
                    "query": "Which chunk explains hybrid retrieval?",
                    "expected_chunk_ids": ["chunk_hybrid"],
                    "notes": "reviewed",
                },
            ]
        ),
        encoding="utf-8",
    )

    report = promote_reviewed_eval_draft(draft_path, output_path)

    assert report.promoted_count == 2
    assert report.skipped_count == 1
    assert report.skipped_ids == ("unreviewed",)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == [
        {
            "id": "reviewed_text",
            "query": "What is AUTH-429?",
            "expected_text_contains": ["Rate Limit Policy defines AUTH-429."],
        },
        {
            "id": "reviewed_chunk",
            "query": "Which chunk explains hybrid retrieval?",
            "expected_chunk_ids": ["chunk_hybrid"],
        },
    ]
