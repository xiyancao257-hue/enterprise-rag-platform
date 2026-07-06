from enterprise_rag.query.engine import QueryEngine


def test_query_plan_normalizes_extra_whitespace() -> None:
    engine = QueryEngine()

    plan = engine.plan("   what   is    hybrid retrieval   ")

    assert plan.original_query == "   what   is    hybrid retrieval   "
    assert plan.normalized_query == "what is hybrid retrieval"
    assert plan.rewritten_queries == ("what is hybrid retrieval", "hybrid retrieval")
    assert plan.corrections == {}


def test_query_plan_corrects_typos_from_vocabulary() -> None:
    engine = QueryEngine(vocabulary={"hybrid", "retrieval", "rerank"})

    plan = engine.plan("hybrid retrival")

    assert plan.normalized_query == "hybrid retrieval"
    assert plan.corrections == {"retrival": "retrieval"}
    assert plan.rewritten_queries == ("hybrid retrieval",)


def test_query_plan_rewrites_comparison_queries() -> None:
    engine = QueryEngine()

    plan = engine.plan("BM25 vs vector search")

    assert plan.normalized_query == "BM25 vs vector search"
    assert plan.rewritten_queries == (
        "BM25 vs vector search",
        "BM25 compare vector search",
    )


def test_query_plan_detects_ambiguous_references() -> None:
    engine = QueryEngine()

    plan = engine.plan("how does it work")

    assert plan.normalized_query == "how does it work"
    assert plan.rewritten_queries == ("how does it work",)
    assert plan.ambiguity_notes == (
        "Query contains references that may need conversation history or clarification.",
    )


def test_query_plan_flags_very_short_queries() -> None:
    engine = QueryEngine()

    plan = engine.plan("retrieval")

    assert plan.normalized_query == "retrieval"
    assert plan.ambiguity_notes == (
        "Query is very short; expansion or clarification may improve recall.",
    )


def test_query_plan_extracts_metadata_filters_and_removes_them_from_query() -> None:
    engine = QueryEngine()

    plan = engine.plan("department:security year:2024 retention policy")

    assert plan.normalized_query == "retention policy"
    assert plan.rewritten_queries == ("retention policy",)
    assert plan.metadata_filters == {"department": "security", "year": "2024"}
