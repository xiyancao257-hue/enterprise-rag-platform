from enterprise_rag.evaluation.experiments import (
    RetrievalExperimentResult,
    RetrievalExperimentReport,
    format_retrieval_experiment_report,
    run_top_k_experiments,
)
from enterprise_rag.evaluation.retrieval_eval import RetrievalCaseResult, RetrievalEvalCase, RetrievalEvalReport
from enterprise_rag.models import Chunk


def test_experiment_report_recommends_highest_recall_then_mrr_then_precision() -> None:
    weak = RetrievalEvalReport(
        k=1,
        case_results=[
            RetrievalCaseResult(
                case_id="weak",
                query="query",
                expected_chunk_ids={"expected"},
                retrieved_chunk_ids=["other"],
                recall_at_k=0.0,
                precision_at_k=0.0,
                reciprocal_rank=0.0,
            )
        ],
    )
    strong = RetrievalEvalReport(
        k=3,
        case_results=[
            RetrievalCaseResult(
                case_id="strong",
                query="query",
                expected_chunk_ids={"expected"},
                retrieved_chunk_ids=["other", "expected"],
                recall_at_k=1.0,
                precision_at_k=1 / 3,
                reciprocal_rank=1 / 2,
            )
        ],
    )

    report = RetrievalExperimentReport(
        results=[
            RetrievalExperimentResult(name="top_k=1", report=weak),
            RetrievalExperimentResult(name="top_k=3", report=strong),
        ]
    )

    assert report.best is not None
    assert report.best.name == "top_k=3"


def test_run_top_k_experiments_runs_real_eval_for_each_k() -> None:
    chunks = [
        Chunk(id="chunk_hybrid", document_id="doc1", text="Hybrid retrieval combines BM25 keyword search with vector search."),
        Chunk(id="chunk_cleaning", document_id="doc1", text="Dirty data cleaning removes repeated headers and OCR noise."),
    ]
    cases = [
        RetrievalEvalCase(
            id="hybrid",
            query="hybrid retrieval",
            expected_chunk_ids={"chunk_hybrid"},
        )
    ]

    report = run_top_k_experiments(cases, chunks, k_values=[1, 2])

    assert [result.name for result in report.results] == ["top_k=1", "top_k=2"]
    assert report.best is not None
    assert report.best.report.recall_at_k == 1.0


def test_run_top_k_experiments_can_enable_graph_retrieval() -> None:
    chunks = [
        Chunk(id="product", document_id="doc1", text="Product Atlas depends on Auth Service."),
        Chunk(id="service", document_id="doc1", text="Auth Service uses Rate Limit Policy."),
        Chunk(id="policy", document_id="doc1", text="Rate Limit Policy defines AUTH-429."),
    ]
    cases = [
        RetrievalEvalCase(id="impact", query="Which product is affected by AUTH-429?", expected_chunk_ids={"product"}),
    ]

    report = run_top_k_experiments(cases, chunks, k_values=[3], enable_graph=True, graph_max_hops=3)

    assert report.results[0].name == "top_k=3+graph_hops=3"
    assert report.results[0].report.recall_at_k == 1.0


def test_format_retrieval_experiment_report_lists_results_and_recommendation() -> None:
    chunks = [
        Chunk(id="chunk_hybrid", document_id="doc1", text="Hybrid retrieval combines BM25 keyword search with vector search."),
    ]
    cases = [
        RetrievalEvalCase(id="hybrid", query="hybrid retrieval", expected_chunk_ids={"chunk_hybrid"}),
    ]
    report = run_top_k_experiments(cases, chunks, k_values=[1])

    formatted = format_retrieval_experiment_report(report)

    assert "Retrieval Experiment Report" in formatted
    assert "- top_k=1:" in formatted
    assert "Recommended:" in formatted
