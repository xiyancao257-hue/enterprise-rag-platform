from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    traffic_weight: int
    retrieval_profile: dict[str, object]


@dataclass(frozen=True)
class ExperimentAssignment:
    experiment_name: str
    variant_name: str
    assignment_key: str
    bucket: int
    retrieval_profile: dict[str, object]


class ExperimentAssigner:
    def __init__(self, experiment_name: str, variants: tuple[ExperimentVariant, ...]) -> None:
        if not variants:
            raise ValueError("Experiment must define at least one variant.")
        if any(variant.traffic_weight <= 0 for variant in variants):
            raise ValueError("Experiment variant weights must be positive.")
        self.experiment_name = experiment_name
        self.variants = variants
        self.total_weight = sum(variant.traffic_weight for variant in variants)

    def assign(self, assignment_key: str) -> ExperimentAssignment:
        bucket = _stable_bucket(f"{self.experiment_name}:{assignment_key}", self.total_weight)
        cursor = 0
        for variant in self.variants:
            cursor += variant.traffic_weight
            if bucket < cursor:
                return ExperimentAssignment(
                    experiment_name=self.experiment_name,
                    variant_name=variant.name,
                    assignment_key=assignment_key,
                    bucket=bucket,
                    retrieval_profile=variant.retrieval_profile,
                )
        variant = self.variants[-1]
        return ExperimentAssignment(
            experiment_name=self.experiment_name,
            variant_name=variant.name,
            assignment_key=assignment_key,
            bucket=bucket,
            retrieval_profile=variant.retrieval_profile,
        )


def assignment_key(*, tenant_id: str | None, user_id: str | None, query: str) -> str:
    tenant = tenant_id or "public"
    user = user_id or "anonymous"
    normalized_query = " ".join(query.lower().split())
    return f"{tenant}:{user}:{normalized_query}"


def format_assignment(assignment: ExperimentAssignment) -> str:
    profile = ", ".join(f"{key}={value}" for key, value in sorted(assignment.retrieval_profile.items()))
    return (
        f"Experiment assignment: {assignment.experiment_name} -> {assignment.variant_name} "
        f"(bucket={assignment.bucket}, key={assignment.assignment_key}, profile={profile})"
    )


def _stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo
