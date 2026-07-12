from enterprise_rag.observability.feedback import FeedbackRecord, JsonFeedbackStore


def test_json_feedback_store_appends_and_loads_records(tmp_path) -> None:
    store = JsonFeedbackStore(tmp_path / "feedback.jsonl")
    record = FeedbackRecord(
        feedback_id="fb_123",
        request_id="req_query_123",
        tenant_id="acme",
        user_id="user-1",
        query="What is hybrid retrieval?",
        answer="Hybrid retrieval combines lexical and vector retrieval.",
        rating="negative",
        citation_chunk_ids=("chunk1",),
        labels=("wrong_citation", "incomplete_answer"),
        comment="The citation did not support the answer.",
    )

    store.append(record)

    assert store.load() == [record]
