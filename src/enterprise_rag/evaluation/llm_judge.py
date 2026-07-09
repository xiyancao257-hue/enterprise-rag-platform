from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.text import token_set


@dataclass(frozen=True)
class JudgeCase:
    case_id: str
    question: str
    answer: str
    reference_answer: str


@dataclass(frozen=True)
class JudgeResult:
    case_id: str
    score: float
    passed: bool
    notes: tuple[str, ...]


class ReferenceOverlapJudge:
    def __init__(self, pass_threshold: float = 0.5) -> None:
        self.pass_threshold = pass_threshold

    def judge(self, case: JudgeCase) -> JudgeResult:
        answer_tokens = token_set(case.answer)
        reference_tokens = token_set(case.reference_answer)
        if not reference_tokens:
            return JudgeResult(case_id=case.case_id, score=0.0, passed=False, notes=("missing_reference",))
        score = len(answer_tokens & reference_tokens) / len(reference_tokens)
        notes = []
        if score < self.pass_threshold:
            notes.append("low_reference_overlap")
        if not answer_tokens:
            notes.append("empty_answer")
        return JudgeResult(
            case_id=case.case_id,
            score=round(score, 4),
            passed=score >= self.pass_threshold,
            notes=tuple(notes),
        )


def run_llm_judge_eval(cases: list[JudgeCase], judge: ReferenceOverlapJudge | None = None) -> list[JudgeResult]:
    active_judge = judge or ReferenceOverlapJudge()
    return [active_judge.judge(case) for case in cases]
