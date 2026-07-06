from enterprise_rag.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from enterprise_rag.evaluation.retrieval_eval import (
    RetrievalEvalCase,
    evaluate_retrieval_results,
    format_retrieval_eval_report,
    load_retrieval_eval_cases,
    run_retrieval_eval,
)
from enterprise_rag.models import Chunk, SearchHit


def hit(chunk_id: str, rank: int) -> SearchHit:
    return SearchHit(
        chunk=Chunk(id=chunk_id, document_id="doc1", text=f"Evidence {chunk_id}"),
        score=1 / rank,
        retriever="test",
        rank=rank,
    )


def test_precision_at_k_counts_correct_results_in_top_k() -> None:
    expected = {"B", "D"}
    retrieved = ["A", "B", "C", "D", "E"]

    assert precision_at_k(expected, retrieved, k=1) == 0.0
    assert precision_at_k(expected, retrieved, k=3) == 1 / 3
    assert precision_at_k(expected, retrieved, k=5) == 2 / 5


def test_recall_at_k_counts_expected_results_found_in_top_k() -> None:
    expected = {"B", "D"}
    retrieved = ["A", "B", "C", "D", "E"]

    assert recall_at_k(expected, retrieved, k=1) == 0.0
    assert recall_at_k(expected, retrieved, k=3) == 1 / 2
    assert recall_at_k(expected, retrieved, k=5) == 1.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    expected = {"B", "D"}

    assert reciprocal_rank(expected, ["B", "A", "D"]) == 1.0
    assert reciprocal_rank(expected, ["A", "B", "D"]) == 1 / 2
    assert reciprocal_rank(expected, ["A", "C", "E"]) == 0.0


def test_evaluate_retrieval_results_calculates_report_and_failures() -> None:
    cases = [
        RetrievalEvalCase(
            id="hybrid_retrieval",
            query="What combines BM25 and vector search?",
            expected_chunk_ids={"chunk_hybrid"},
        ),
        RetrievalEvalCase(
            id="dirty_data_cleaning",
            query="What removes repeated headers?",
            expected_chunk_ids={"chunk_cleaning"},
        ),
    ]
    retrieved_by_case_id = {
        "hybrid_retrieval": [
            hit("chunk_other", rank=1),
            hit("chunk_hybrid", rank=2),
            hit("chunk_extra", rank=3),
        ],
        "dirty_data_cleaning": [
            hit("chunk_other", rank=1),
            hit("chunk_extra", rank=2),
        ],
    }

    report = evaluate_retrieval_results(cases, retrieved_by_case_id, k=3)

    assert report.k == 3
    assert report.recall_at_k == (1.0 + 0.0) / 2
    assert report.precision_at_k == ((1 / 3) + 0.0) / 2
    assert report.mrr == ((1 / 2) + 0.0) / 2
    assert len(report.failures) == 1
    assert report.failures[0].case_id == "dirty_data_cleaning"
    assert report.failures[0].retrieved_chunk_ids == ["chunk_other", "chunk_extra"]


def test_load_retrieval_eval_cases_maps_expected_text_to_chunk_ids(tmp_path) -> None:
    chunks = [
        Chunk(id="chunk_hybrid", document_id="doc1", text="Hybrid retrieval combines BM25 and vector search."),
        Chunk(id="chunk_cleaning", document_id="doc1", text="Dirty data cleaning removes repeated headers."),
    ]
    eval_path = tmp_path / "retrieval_eval.json"
    eval_path.write_text(
        """
        [
          {
            "id": "hybrid",
            "query": "What combines BM25 and vector search?",
            "expected_text_contains": ["Hybrid retrieval combines BM25"]
          }
        ]
        """,
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(eval_path, chunks)

    assert cases == [
        RetrievalEvalCase(
            id="hybrid",
            query="What combines BM25 and vector search?",
            expected_chunk_ids={"chunk_hybrid"},
        )
    ]


def test_load_retrieval_eval_cases_warns_when_expected_text_does_not_match(tmp_path) -> None:
    chunks = [
        Chunk(id="chunk_hybrid", document_id="doc1", text="Hybrid retrieval combines BM25 and vector search."),
    ]
    eval_path = tmp_path / "retrieval_eval.json"
    eval_path.write_text(
        """
        [
          {
            "id": "missing_expected_text",
            "query": "What removes OCR noise?",
            "expected_text_contains": ["Dirty data filtering removes OCR noise"]
          }
        ]
        """,
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(eval_path, chunks)

    assert cases[0].expected_chunk_ids == set()
    assert cases[0].warnings == (
        "expected_text_contains did not match any chunk: Dirty data filtering removes OCR noise",
    )


def test_run_retrieval_eval_uses_real_query_engine_and_hybrid_retriever() -> None:
    chunks = [
        Chunk(id="chunk_hybrid", document_id="doc1", text="Hybrid retrieval combines BM25 keyword search with vector search."),
        Chunk(id="chunk_cleaning", document_id="doc1", text="Dirty data cleaning removes repeated headers and OCR noise."),
    ]
    cases = [
        RetrievalEvalCase(
            id="hybrid",
            query="hybrid retrival",
            expected_chunk_ids={"chunk_hybrid"},
        )
    ]

    report = run_retrieval_eval(cases, chunks, k=2)

    assert report.recall_at_k == 1.0
    assert report.failures == []


def test_run_retrieval_eval_can_enable_graph_retrieval() -> None:
    chunks = [
        Chunk(id="product", document_id="doc1", text="Product Atlas depends on Auth Service."),
        Chunk(id="service", document_id="doc1", text="Auth Service uses Rate Limit Policy."),
        Chunk(id="policy", document_id="doc1", text="Rate Limit Policy defines AUTH-429."),
    ]
    cases = [
        RetrievalEvalCase(
            id="impact",
            query="Which product is affected by AUTH-429?",
            expected_chunk_ids={"product"},
        )
    ]

    report = run_retrieval_eval(cases, chunks, k=3, enable_graph=True, graph_max_hops=3)

    assert report.recall_at_k == 1.0
    assert report.failures == []


def test_format_retrieval_eval_report_summarizes_metrics_and_failures() -> None:
    cases = [
        RetrievalEvalCase(id="missing", query="unknown", expected_chunk_ids={"expected"}),
    ]
    report = evaluate_retrieval_results(cases, {"missing": [hit("other", rank=1)]}, k=1)

    formatted = format_retrieval_eval_report(report)

    assert "Retrieval Eval Report" in formatted
    assert "cases: 1" in formatted
    assert "Recall@1: 0.00" in formatted
    assert "- missing" in formatted
    assert "expected: expected" in formatted
    assert "retrieved: other" in formatted


def test_format_retrieval_eval_report_includes_warnings() -> None:
    cases = [
        RetrievalEvalCase(
            id="bad_eval_case",
            query="unknown",
            expected_chunk_ids=set(),
            warnings=("expected_text_contains did not match any chunk: missing text",),
        ),
    ]
    report = evaluate_retrieval_results(cases, {"bad_eval_case": []}, k=1)

    formatted = format_retrieval_eval_report(report)

    assert "Warnings:" in formatted
    assert "- bad_eval_case: expected_text_contains did not match any chunk: missing text" in formatted
