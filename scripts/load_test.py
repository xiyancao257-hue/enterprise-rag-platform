#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    status_code: int | None
    latency_ms: float
    estimated_cost_usd: float = 0.0
    citation_count: int = 0
    error: str = ""


def main() -> int:
    args = _parse_args()
    queries = _load_queries(args)
    started_at = time.perf_counter()
    results = _run_load_test(args, queries)
    duration_seconds = time.perf_counter() - started_at
    report = build_report(results, duration_seconds=duration_seconds, baseline_path=args.baseline)
    violations = evaluate_gates(
        report,
        max_error_rate=args.max_error_rate,
        p95_threshold_ms=args.p95_threshold_ms,
        avg_cost_threshold_usd=args.avg_cost_threshold_usd,
        total_cost_threshold_usd=args.total_cost_threshold_usd,
        max_p95_regression_pct=args.max_p95_regression_pct,
    )
    report["status"] = "FAIL" if violations else "PASS"
    report["violations"] = violations

    print(format_report(report))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if violations else 0


def build_report(
    results: list[RequestResult],
    *,
    duration_seconds: float,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    total = len(results)
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_ms for result in successes]
    costs = [result.estimated_cost_usd for result in successes]
    citation_counts = [result.citation_count for result in successes]

    report: dict[str, Any] = {
        "requests": total,
        "success": len(successes),
        "failures": len(failures),
        "error_rate": (len(failures) / total) if total else 0.0,
        "duration_seconds": round(duration_seconds, 4),
        "rps": round(total / duration_seconds, 4) if duration_seconds > 0 else 0.0,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 4),
            "p95": round(percentile(latencies, 95), 4),
            "p99": round(percentile(latencies, 99), 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "cost_usd": {
            "average": round((sum(costs) / len(costs)) if costs else 0.0, 8),
            "total": round(sum(costs), 8),
        },
        "avg_citations": round((sum(citation_counts) / len(citation_counts)) if citation_counts else 0.0, 4),
        "failure_samples": [result.error for result in failures[:5] if result.error],
    }
    if baseline_path is not None and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_p95 = float(baseline.get("latency_ms", {}).get("p95", 0.0))
        current_p95 = float(report["latency_ms"]["p95"])
        regression_pct = ((current_p95 - baseline_p95) / baseline_p95 * 100) if baseline_p95 > 0 else 0.0
        report["baseline"] = {
            "path": str(baseline_path),
            "p95_ms": baseline_p95,
            "p95_regression_pct": round(regression_pct, 4),
        }
    return report


def evaluate_gates(
    report: dict[str, Any],
    *,
    max_error_rate: float,
    p95_threshold_ms: float | None,
    avg_cost_threshold_usd: float | None,
    total_cost_threshold_usd: float | None,
    max_p95_regression_pct: float | None,
) -> list[str]:
    violations = []
    error_rate = float(report["error_rate"])
    p95 = float(report["latency_ms"]["p95"])
    avg_cost = float(report["cost_usd"]["average"])
    total_cost = float(report["cost_usd"]["total"])

    if error_rate > max_error_rate:
        violations.append(f"error_rate {error_rate:.4f} exceeded threshold {max_error_rate:.4f}")
    if p95_threshold_ms is not None and p95 > p95_threshold_ms:
        violations.append(f"p95 latency {p95:.2f} ms exceeded threshold {p95_threshold_ms:.2f} ms")
    if avg_cost_threshold_usd is not None and avg_cost > avg_cost_threshold_usd:
        violations.append(f"average cost ${avg_cost:.8f} exceeded threshold ${avg_cost_threshold_usd:.8f}")
    if total_cost_threshold_usd is not None and total_cost > total_cost_threshold_usd:
        violations.append(f"total cost ${total_cost:.8f} exceeded threshold ${total_cost_threshold_usd:.8f}")
    baseline = report.get("baseline")
    if baseline and max_p95_regression_pct is not None:
        regression_pct = float(baseline["p95_regression_pct"])
        if regression_pct > max_p95_regression_pct:
            violations.append(f"p95 regression {regression_pct:.2f}% exceeded threshold {max_p95_regression_pct:.2f}%")
    return violations


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil((pct / 100) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def format_report(report: dict[str, Any]) -> str:
    latency = report["latency_ms"]
    cost = report["cost_usd"]
    lines = [
        "Load Test Report",
        f"requests: {report['requests']}",
        f"success: {report['success']}",
        f"failures: {report['failures']}",
        f"error_rate: {report['error_rate']:.4f}",
        f"rps: {report['rps']:.4f}",
        "latency_ms:",
        f"  p50: {latency['p50']:.4f}",
        f"  p95: {latency['p95']:.4f}",
        f"  p99: {latency['p99']:.4f}",
        f"  max: {latency['max']:.4f}",
        f"avg_citations: {report['avg_citations']:.4f}",
        "cost_usd:",
        f"  average: {cost['average']:.8f}",
        f"  total: {cost['total']:.8f}",
    ]
    if "baseline" in report:
        baseline = report["baseline"]
        lines.extend(
            [
                "baseline:",
                f"  path: {baseline['path']}",
                f"  p95_ms: {baseline['p95_ms']:.4f}",
                f"  p95_regression_pct: {baseline['p95_regression_pct']:.4f}",
            ]
        )
    if report.get("violations"):
        lines.append("violations:")
        lines.extend(f"  - {violation}" for violation in report["violations"])
    lines.append(f"status: {report.get('status', 'PASS')}")
    return "\n".join(lines)


def _run_load_test(args: argparse.Namespace, queries: list[str]) -> list[RequestResult]:
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_send_query, args, queries[index % len(queries)]) for index in range(args.requests)]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _send_query(args: argparse.Namespace, query: str) -> RequestResult:
    payload: dict[str, Any] = {"query": query}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.include_trace:
        payload["include_trace"] = True

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    if args.tenant:
        headers["X-Tenant-ID"] = args.tenant

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(args.url, data=body, headers=headers, method="POST")
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            latency_ms = (time.perf_counter() - started_at) * 1000
            data = json.loads(response.read().decode("utf-8"))
            return RequestResult(
                ok=200 <= response.status < 300,
                status_code=response.status,
                latency_ms=latency_ms,
                estimated_cost_usd=float(data.get("cost", {}).get("estimated_cost_usd", 0.0)),
                citation_count=len(data.get("citations", [])),
            )
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        detail = exc.read().decode("utf-8", errors="replace")
        return RequestResult(False, exc.code, latency_ms, error=f"HTTP {exc.code}: {detail[:300]}")
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return RequestResult(False, None, latency_ms, error=repr(exc))


def _load_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.query)
    if args.queries_file:
        file_queries = [
            line.strip()
            for line in args.queries_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        queries.extend(file_queries)
    if not queries:
        raise SystemExit("Provide at least one --query or --queries-file.")
    return queries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight load test against the Enterprise RAG API.")
    parser.add_argument("--url", default="http://localhost:8000/query", help="Query endpoint URL.")
    parser.add_argument("--query", action="append", default=[], help="Query text. Can be provided multiple times.")
    parser.add_argument("--queries-file", type=Path, help="Text file with one query per line.")
    parser.add_argument("--requests", type=int, default=20, help="Total number of query requests.")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="Per-request timeout.")
    parser.add_argument("--api-key", default="", help="API key for Authorization: Bearer.")
    parser.add_argument("--tenant", default="", help="Tenant id for X-Tenant-ID.")
    parser.add_argument("--top-k", type=int, help="Optional top_k request override.")
    parser.add_argument("--include-trace", action="store_true", help="Request retrieval trace in API responses.")
    parser.add_argument("--output-json", type=Path, help="Write the load test report as JSON.")
    parser.add_argument("--baseline", type=Path, help="Previous JSON report for p95 regression comparison.")
    parser.add_argument("--max-error-rate", type=float, default=0.01, help="Fail when error rate is above this value.")
    parser.add_argument("--p95-threshold-ms", type=float, help="Fail when p95 latency exceeds this threshold.")
    parser.add_argument("--avg-cost-threshold-usd", type=float, help="Fail when average estimated cost exceeds this.")
    parser.add_argument("--total-cost-threshold-usd", type=float, help="Fail when total estimated cost exceeds this.")
    parser.add_argument("--max-p95-regression-pct", type=float, help="Fail when p95 regresses versus baseline.")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


if __name__ == "__main__":
    sys.exit(main())
