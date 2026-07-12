import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path("scripts/load_test.py")
    spec = importlib.util.spec_from_file_location("load_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_report_summarizes_latency_cost_and_citations() -> None:
    load_test = _load_module()
    results = [
        load_test.RequestResult(True, 200, 10.0, estimated_cost_usd=0.01, citation_count=1),
        load_test.RequestResult(True, 200, 20.0, estimated_cost_usd=0.03, citation_count=3),
        load_test.RequestResult(False, 500, 30.0, error="HTTP 500"),
    ]

    report = load_test.build_report(results, duration_seconds=2.0)

    assert report["requests"] == 3
    assert report["success"] == 2
    assert report["failures"] == 1
    assert report["error_rate"] == 1 / 3
    assert report["rps"] == 1.5
    assert report["latency_ms"]["p50"] == 10.0
    assert report["latency_ms"]["p95"] == 20.0
    assert report["cost_usd"]["average"] == 0.02
    assert report["cost_usd"]["total"] == 0.04
    assert report["avg_citations"] == 2.0
    assert report["failure_samples"] == ["HTTP 500"]


def test_evaluate_gates_reports_latency_cost_and_error_violations() -> None:
    load_test = _load_module()
    report = {
        "error_rate": 0.2,
        "latency_ms": {"p95": 3500.0},
        "cost_usd": {"average": 0.05, "total": 2.0},
    }

    violations = load_test.evaluate_gates(
        report,
        max_error_rate=0.01,
        p95_threshold_ms=3000.0,
        avg_cost_threshold_usd=0.01,
        total_cost_threshold_usd=1.0,
        max_p95_regression_pct=None,
    )

    assert len(violations) == 4
    assert any("error_rate" in violation for violation in violations)
    assert any("p95 latency" in violation for violation in violations)
    assert any("average cost" in violation for violation in violations)
    assert any("total cost" in violation for violation in violations)


def test_build_report_compares_baseline_p95(tmp_path) -> None:
    load_test = _load_module()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"latency_ms": {"p95": 100.0}}), encoding="utf-8")

    report = load_test.build_report(
        [load_test.RequestResult(True, 200, 130.0)],
        duration_seconds=1.0,
        baseline_path=baseline_path,
    )

    assert report["baseline"]["p95_ms"] == 100.0
    assert report["baseline"]["p95_regression_pct"] == 30.0

    violations = load_test.evaluate_gates(
        report,
        max_error_rate=0.01,
        p95_threshold_ms=None,
        avg_cost_threshold_usd=None,
        total_cost_threshold_usd=None,
        max_p95_regression_pct=20.0,
    )
    assert violations == ["p95 regression 30.00% exceeded threshold 20.00%"]


def test_load_test_script_is_documented_in_deployment_examples() -> None:
    script = Path("scripts/load_test.py").read_text(encoding="utf-8")

    assert "--p95-threshold-ms" in script
    assert "--avg-cost-threshold-usd" in script
    assert "--total-cost-threshold-usd" in script
    assert "--max-p95-regression-pct" in script
