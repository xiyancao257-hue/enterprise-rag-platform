from enterprise_rag.evaluation.llm_judge import JudgeCase, ReferenceOverlapJudge, run_llm_judge_eval


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
