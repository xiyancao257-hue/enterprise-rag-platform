import json

import pytest

from enterprise_rag.evaluation.evidence_suggestion import approve_suggested_evidence, suggest_evidence_for_eval_draft
from enterprise_rag.models import Chunk


def test_suggest_evidence_for_eval_draft_adds_candidates_without_promoting(tmp_path) -> None:
    draft_path = tmp_path / "generated_eval.json"
    output_path = tmp_path / "generated_with_suggestions.json"
    draft_path.write_text(
        json.dumps(
            [
                {
                    "id": "log_1_auth_429",
                    "query": "What is AUTH-429?",
                    "expected_text_contains": [],
                    "notes": "needs review",
                }
            ]
        ),
        encoding="utf-8",
    )
    chunks = [
        Chunk(
            id="policy",
            document_id="doc1",
            text="Rate Limit Policy defines AUTH-429.",
            heading_path=("Service Dependency Notes",),
            metadata={"source_path": "data/raw/kg_demo.md"},
        ),
        Chunk(
            id="unrelated",
            document_id="doc2",
            text="Hybrid retrieval combines BM25 and vector search.",
        ),
    ]

    suggest_evidence_for_eval_draft(draft_path, chunks, output_path, top_k=1)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["id"] == "log_1_auth_429"
    assert data[0]["expected_text_contains"] == []
    assert data[0]["suggested_evidence"][0]["chunk_id"] == "policy"
    assert data[0]["suggested_evidence"][0]["source_path"] == "data/raw/kg_demo.md"
    assert data[0]["suggested_evidence"][0]["heading_path"] == ["Service Dependency Notes"]
    assert data[0]["suggested_evidence"][0]["text"] == "Rate Limit Policy defines AUTH-429."


def test_suggest_evidence_for_empty_query_returns_no_candidates(tmp_path) -> None:
    draft_path = tmp_path / "generated_eval.json"
    output_path = tmp_path / "generated_with_suggestions.json"
    draft_path.write_text(
        json.dumps(
            [
                {
                    "id": "empty",
                    "query": "",
                    "expected_text_contains": [],
                    "notes": "needs review",
                }
            ]
        ),
        encoding="utf-8",
    )

    suggest_evidence_for_eval_draft(draft_path, [], output_path, top_k=3)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["suggested_evidence"] == []


def test_approve_suggested_evidence_copies_text_to_expected_text_contains(tmp_path) -> None:
    draft_path = tmp_path / "generated_with_suggestions.json"
    output_path = tmp_path / "reviewed_eval.json"
    draft_path.write_text(
        json.dumps(
            [
                {
                    "id": "log_1_auth_429",
                    "query": "What is AUTH-429?",
                    "expected_text_contains": [],
                    "suggested_evidence": [
                        {
                            "chunk_id": "policy",
                            "text": "Rate Limit Policy defines AUTH-429.",
                        }
                    ],
                },
                {
                    "id": "other",
                    "query": "Other query",
                    "expected_text_contains": [],
                    "suggested_evidence": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    approve_suggested_evidence(draft_path, "log_1_auth_429", 0, output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data[0]["expected_text_contains"] == ["Rate Limit Policy defines AUTH-429."]
    assert data[0]["suggested_evidence"][0]["chunk_id"] == "policy"
    assert data[1]["expected_text_contains"] == []


def test_approve_suggested_evidence_rejects_unknown_case_id(tmp_path) -> None:
    draft_path = tmp_path / "generated_with_suggestions.json"
    output_path = tmp_path / "reviewed_eval.json"
    draft_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Case id not found"):
        approve_suggested_evidence(draft_path, "missing", 0, output_path)


def test_approve_suggested_evidence_rejects_out_of_range_index(tmp_path) -> None:
    draft_path = tmp_path / "generated_with_suggestions.json"
    output_path = tmp_path / "reviewed_eval.json"
    draft_path.write_text(
        json.dumps(
            [
                {
                    "id": "log_1_auth_429",
                    "query": "What is AUTH-429?",
                    "suggested_evidence": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(IndexError, match="out of range"):
        approve_suggested_evidence(draft_path, "log_1_auth_429", 0, output_path)
