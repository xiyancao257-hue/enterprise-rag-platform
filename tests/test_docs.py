from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prompting_and_rag_eval_notes_cover_interview_topics() -> None:
    notes = (PROJECT_ROOT / "docs" / "prompting_and_rag_eval.md").read_text()
    normalized_notes = notes.lower()

    expected_terms = [
        "zero-shot",
        "few-shot",
        "chain-of-thought",
        "role-specific",
        "context relevance",
        "faithfulness",
        "answer correctness",
        "raft",
        "retrieval-augmented fine-tuning",
    ]

    for term in expected_terms:
        assert term in normalized_notes
