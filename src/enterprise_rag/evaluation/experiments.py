from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.evaluation.retrieval_eval import RetrievalEvalCase, RetrievalEvalReport, run_retrieval_eval
from enterprise_rag.models import Chunk


@dataclass(frozen=True)
class RetrievalExperimentResult:
    name: str
    report: RetrievalEvalReport


@dataclass(frozen=True)
class RetrievalExperimentReport:
    results: list[RetrievalExperimentResult]

    @property
    def best(self) -> RetrievalExperimentResult | None:
        if not self.results:
            return None
        return max(
            self.results,
            key=lambda result: (
                result.report.recall_at_k,
                result.report.mrr,
                result.report.precision_at_k,
                -result.report.k,
            ),
        )


def run_top_k_experiments(
    cases: list[RetrievalEvalCase],
    chunks: list[Chunk],
    k_values: list[int],
    enable_graph: bool = False,
    graph_max_hops: int = 2,
) -> RetrievalExperimentReport:
    results = []
    for k in k_values:
        name = f"top_k={k}"
        if enable_graph:
            name += f"+graph_hops={graph_max_hops}"
        results.append(
            RetrievalExperimentResult(
                name=name,
                report=run_retrieval_eval(
                    cases,
                    chunks,
                    k=k,
                    enable_graph=enable_graph,
                    graph_max_hops=graph_max_hops,
                ),
            )
        )
    return RetrievalExperimentReport(results=results)


def format_retrieval_experiment_report(report: RetrievalExperimentReport) -> str:
    lines = ["Retrieval Experiment Report"]
    if not report.results:
        lines.append("- no experiments")
        return "\n".join(lines)

    lines.append("results:")
    for result in report.results:
        lines.append(
            f"- {result.name}: "
            f"Recall@{result.report.k}={result.report.recall_at_k:.2f}, "
            f"Precision@{result.report.k}={result.report.precision_at_k:.2f}, "
            f"MRR={result.report.mrr:.2f}, "
            f"failures={len(result.report.failures)}"
        )

    best = report.best
    if best is not None:
        lines.extend(
            [
                "",
                "Recommended:",
                (
                    f"- {best.name} "
                    f"(Recall@{best.report.k}={best.report.recall_at_k:.2f}, "
                    f"Precision@{best.report.k}={best.report.precision_at_k:.2f}, "
                    f"MRR={best.report.mrr:.2f})"
                ),
            ]
        )
    return "\n".join(lines)
