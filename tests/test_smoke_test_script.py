import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path("scripts/smoke_test.py")
    spec = importlib.util.spec_from_file_location("smoke_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_test_skips_when_api_key_is_missing() -> None:
    smoke_test = _load_module()
    args = argparse.Namespace(
        base_url="http://localhost:8000",
        tenant="acme",
        api_key="",
        allow_missing_api_key=False,
        require_openai=False,
    )

    env_status = smoke_test.build_env_status(args)
    reasons = smoke_test.skip_reasons(args, env_status)

    assert reasons == ["Missing ENTERPRISE_RAG_API_KEY or --api-key."]


def test_query_response_validation_requires_answer_citations_and_cost() -> None:
    smoke_test = _load_module()

    assert (
        smoke_test.validate_query_response(
            json.dumps({"answer": "ok", "citations": [], "cost": {"estimated_cost_usd": 0.0}})
        )
        == ""
    )
    assert smoke_test.validate_query_response(json.dumps({"citations": [], "cost": {}})) == (
        "Query response did not include an answer."
    )
    assert smoke_test.validate_query_response(json.dumps({"answer": "ok", "citations": "bad", "cost": {}})) == (
        "Query response citations were not a list."
    )
    assert smoke_test.validate_query_response(json.dumps({"answer": "ok", "citations": []})) == (
        "Query response did not include cost."
    )


def test_metrics_validation_requires_core_metrics() -> None:
    smoke_test = _load_module()
    valid_metrics = "\n".join(
        [
            "enterprise_rag_query_requests_total 1",
            "enterprise_rag_query_latency_ms_count 1",
            "enterprise_rag_query_estimated_cost_usd_sum 0.0",
        ]
    )

    assert smoke_test.validate_metrics_response(valid_metrics) == ""
    assert "enterprise_rag_query_latency_ms_count" in smoke_test.validate_metrics_response(
        "enterprise_rag_query_requests_total 1"
    )


def test_build_report_serializes_check_results() -> None:
    smoke_test = _load_module()
    report = smoke_test.build_report(
        status="PASS",
        env_status={"api_key_present": True},
        checks=[smoke_test.SmokeCheck("query", "PASS", 12.34567, "HTTP 200")],
        skipped_reasons=[],
    )

    assert report["status"] == "PASS"
    assert report["checks"] == [
        {
            "name": "query",
            "status": "PASS",
            "latency_ms": 12.3457,
            "detail": "HTTP 200",
        }
    ]
    assert "query: PASS" in smoke_test.format_report(report)


def test_smoke_test_script_exposes_real_provider_options() -> None:
    script = Path("scripts/smoke_test.py").read_text(encoding="utf-8")

    assert "--require-openai" in script
    assert "OPENAI_API_KEY" in script
    assert "ENTERPRISE_RAG_API_KEY" in script
    assert "/readiness" in script
    assert "/metrics" in script
