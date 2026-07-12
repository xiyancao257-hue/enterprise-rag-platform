from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.text import token_set


@dataclass(frozen=True)
class JudgeCase:
    case_id: str
    question: str
    answer: str
    reference_answer: str
    candidate_id: str = ""


@dataclass(frozen=True)
class BlindJudgeCase:
    case_id: str
    question: str
    answer: str
    reference_answer: str


@dataclass(frozen=True)
class JudgeRubric:
    pass_threshold: float = 0.5
    dimensions: tuple[str, ...] = ("reference_overlap", "answer_non_empty")


@dataclass(frozen=True)
class JudgeResult:
    case_id: str
    score: float
    passed: bool
    notes: tuple[str, ...]
    rubric_scores: dict[str, float] | None = None
    bias_controls: tuple[str, ...] = ()


class ReferenceOverlapJudge:
    def __init__(self, pass_threshold: float = 0.5, rubric: JudgeRubric | None = None) -> None:
        self.rubric = rubric or JudgeRubric(pass_threshold=pass_threshold)

    def judge(self, case: JudgeCase | BlindJudgeCase) -> JudgeResult:
        answer_tokens = token_set(case.answer)
        reference_tokens = token_set(case.reference_answer)
        if not reference_tokens:
            return JudgeResult(
                case_id=case.case_id,
                score=0.0,
                passed=False,
                notes=("missing_reference",),
                rubric_scores={"reference_overlap": 0.0, "answer_non_empty": float(bool(answer_tokens))},
                bias_controls=_bias_controls(case),
            )
        score = len(answer_tokens & reference_tokens) / len(reference_tokens)
        notes = []
        if score < self.rubric.pass_threshold:
            notes.append("low_reference_overlap")
        if not answer_tokens:
            notes.append("empty_answer")
        return JudgeResult(
            case_id=case.case_id,
            score=round(score, 4),
            passed=score >= self.rubric.pass_threshold,
            notes=tuple(notes),
            rubric_scores={
                "reference_overlap": round(score, 4),
                "answer_non_empty": float(bool(answer_tokens)),
            },
            bias_controls=_bias_controls(case),
        )


def blind_judge_case(case: JudgeCase) -> BlindJudgeCase:
    return BlindJudgeCase(
        case_id=case.case_id,
        question=case.question,
        answer=case.answer,
        reference_answer=case.reference_answer,
    )


def run_llm_judge_eval(
    cases: list[JudgeCase],
    judge: ReferenceOverlapJudge | None = None,
    blind: bool = True,
) -> list[JudgeResult]:
    active_judge = judge or ReferenceOverlapJudge()
    return [active_judge.judge(blind_judge_case(case) if blind else case) for case in cases]


def format_judge_report(results: list[JudgeResult]) -> str:
    lines = ["LLM-as-Judge Report", f"cases: {len(results)}"]
    if not results:
        return "\n".join(lines)
    pass_count = sum(1 for result in results if result.passed)
    average_score = sum(result.score for result in results) / len(results)
    lines.extend(
        [
            f"pass_rate: {pass_count / len(results):.2f}",
            f"average_score: {average_score:.2f}",
            "bias_controls:",
        ]
    )
    controls = sorted({control for result in results for control in result.bias_controls})
    lines.extend(f"- {control}" for control in controls)
    lines.append("failures:")
    failures = [result for result in results if not result.passed]
    if not failures:
        lines.append("- none")
    else:
        for failure in failures:
            lines.append(f"- {failure.case_id}: score={failure.score:.2f}, notes={', '.join(failure.notes)}")
    return "\n".join(lines)


def _bias_controls(case: JudgeCase | BlindJudgeCase) -> tuple[str, ...]:
    controls = ["rubric_scores_recorded"]
    if isinstance(case, BlindJudgeCase):
        controls.append("candidate_id_blinded")
    return tuple(controls)
