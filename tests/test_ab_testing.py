import pytest

from enterprise_rag.evaluation.ab_testing import (
    ExperimentAssigner,
    ExperimentVariant,
    assignment_key,
    format_assignment,
)


def test_experiment_assigner_is_deterministic_for_same_key() -> None:
    assigner = ExperimentAssigner(
        "retrieval_profile",
        (
            ExperimentVariant(
                name="baseline",
                traffic_weight=50,
                retrieval_profile={"top_k": 5, "enable_graph": False},
            ),
            ExperimentVariant(
                name="graph",
                traffic_weight=50,
                retrieval_profile={"top_k": 5, "enable_graph": True},
            ),
        ),
    )
    key = assignment_key(tenant_id="acme", user_id="u1", query="What does AUTH-429 affect?")

    first = assigner.assign(key)
    second = assigner.assign(key)

    assert first == second
    assert first.variant_name in {"baseline", "graph"}
    assert first.assignment_key == "acme:u1:what does auth-429 affect?"


def test_experiment_assigner_respects_weight_boundaries() -> None:
    assigner = ExperimentAssigner(
        "all_baseline",
        (
            ExperimentVariant(name="baseline", traffic_weight=100, retrieval_profile={"top_k": 5}),
            ExperimentVariant(name="candidate", traffic_weight=1, retrieval_profile={"top_k": 8}),
        ),
    )

    assignment = assigner.assign("known-key")

    assert assignment.variant_name in {"baseline", "candidate"}
    assert 0 <= assignment.bucket < 101


def test_assignment_key_normalizes_query_text() -> None:
    first = assignment_key(tenant_id=None, user_id=None, query=" Hybrid   Retrieval ")
    second = assignment_key(tenant_id=None, user_id=None, query="hybrid retrieval")

    assert first == second
    assert first == "public:anonymous:hybrid retrieval"


def test_experiment_assigner_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="at least one variant"):
        ExperimentAssigner("empty", ())

    with pytest.raises(ValueError, match="weights must be positive"):
        ExperimentAssigner(
            "bad_weight",
            (ExperimentVariant(name="disabled", traffic_weight=0, retrieval_profile={}),),
        )


def test_format_assignment_is_readable() -> None:
    assigner = ExperimentAssigner(
        "retrieval_profile",
        (ExperimentVariant(name="baseline", traffic_weight=1, retrieval_profile={"top_k": 5}),),
    )

    formatted = format_assignment(assigner.assign("tenant:user:query"))

    assert "Experiment assignment: retrieval_profile -> baseline" in formatted
    assert "top_k=5" in formatted
