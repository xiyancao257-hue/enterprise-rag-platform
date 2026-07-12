from enterprise_rag.evaluation.llm_judge import (
    JudgeCase,
    JudgeRubric,
    ReferenceOverlapJudge,
    blind_judge_case,
    format_judge_report,
    run_llm_judge_eval,
)


def test_reference_overlap_judge_passes_supported_answer() -> None:
    case = JudgeCase(
        case_id="case_1",
        question="What is hybrid retrieval?",
        answer="Hybrid retrieval combines BM25 and vector search.",
        reference_answer="Hybrid retrieval combines BM25 keyword search and vector search.",
    )

    result = ReferenceOverlapJudge(pass_threshold=0.5).judge(case)

    assert result.passed is True
    assert result.score >= 0.5
    assert result.notes == ()
    assert result.rubric_scores is not None
    assert result.rubric_scores["reference_overlap"] >= 0.5


def test_reference_overlap_judge_flags_low_overlap() -> None:
    case = JudgeCase(
        case_id="case_1",
        question="What is hybrid retrieval?",
        answer="The system discusses unrelated billing workflows.",
        reference_answer="Hybrid retrieval combines BM25 and vector search.",
    )

    result = ReferenceOverlapJudge(pass_threshold=0.5).judge(case)

    assert result.passed is False
    assert "low_reference_overlap" in result.notes


def test_run_llm_judge_eval_scores_multiple_cases() -> None:
    results = run_llm_judge_eval(
        [
            JudgeCase("case_1", "Q1", "answer one", "answer one"),
            JudgeCase("case_2", "Q2", "", "reference answer"),
        ]
    )

    assert [result.case_id for result in results] == ["case_1", "case_2"]
    assert results[0].passed is True
    assert results[1].passed is False


def test_blind_judge_case_removes_candidate_identity() -> None:
    case = JudgeCase(
        case_id="case_1",
        question="What is hybrid retrieval?",
        answer="Hybrid retrieval combines BM25 and vector search.",
        reference_answer="Hybrid retrieval combines BM25 and vector search.",
        candidate_id="new_reranker_variant",
    )

    blind_case = blind_judge_case(case)
    result = run_llm_judge_eval([case])[0]

    assert not hasattr(blind_case, "candidate_id")
    assert "candidate_id_blinded" in result.bias_controls


def test_reference_overlap_judge_accepts_rubric_threshold() -> None:
    case = JudgeCase(
        case_id="case_1",
        question="What is hybrid retrieval?",
        answer="Hybrid retrieval combines BM25.",
        reference_answer="Hybrid retrieval combines BM25 and vector search.",
    )

    result = ReferenceOverlapJudge(rubric=JudgeRubric(pass_threshold=0.9)).judge(case)

    assert result.passed is False
    assert "low_reference_overlap" in result.notes
    assert result.rubric_scores is not None
    assert result.rubric_scores["answer_non_empty"] == 1.0


def test_format_judge_report_summarizes_bias_controls_and_failures() -> None:
    results = run_llm_judge_eval(
        [
            JudgeCase("pass", "Q1", "answer one", "answer one", candidate_id="baseline"),
            JudgeCase("fail", "Q2", "unrelated", "reference answer", candidate_id="candidate"),
        ]
    )

    report = format_judge_report(results)

    assert "LLM-as-Judge Report" in report
    assert "pass_rate: 0.50" in report
    assert "- candidate_id_blinded" in report
    assert "- fail:" in report
