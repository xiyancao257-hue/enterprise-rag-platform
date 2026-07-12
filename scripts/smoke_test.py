#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    status: str
    latency_ms: float = 0.0
    detail: str = ""


def main() -> int:
    args = _parse_args()
    env_status = build_env_status(args)
    skipped_reasons = skip_reasons(args, env_status)
    if skipped_reasons:
        report = build_report(
            status="SKIPPED",
            env_status=env_status,
            checks=[],
            skipped_reasons=skipped_reasons,
        )
        _emit_report(report, args.output_json)
        return 0

    checks = run_smoke_checks(args)
    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    report = build_report(status=status, env_status=env_status, checks=checks, skipped_reasons=[])
    _emit_report(report, args.output_json)
    return 0 if status == "PASS" else 1


def build_env_status(args: argparse.Namespace) -> dict[str, object]:
    return {
        "base_url": args.base_url,
        "tenant": args.tenant,
        "api_key_present": bool(args.api_key),
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "enterprise_rag_config": os.getenv("ENTERPRISE_RAG_CONFIG", ""),
    }


def skip_reasons(args: argparse.Namespace, env_status: dict[str, object]) -> list[str]:
    reasons = []
    if not args.api_key and not args.allow_missing_api_key:
        reasons.append("Missing ENTERPRISE_RAG_API_KEY or --api-key.")
    if args.require_openai and not env_status["openai_api_key_present"]:
        reasons.append("Missing OPENAI_API_KEY for real provider smoke test.")
    return reasons


def run_smoke_checks(args: argparse.Namespace) -> list[SmokeCheck]:
    headers = auth_headers(args)
    query_headers = {**headers, "Content-Type": "application/json"}
    query_payload = {
        "query": args.query,
        "top_k": args.top_k,
        "include_trace": args.include_trace,
    }
    return [
        request_check(
            "health",
            "GET",
            f"{args.base_url}/health",
            headers=headers,
            timeout_seconds=args.timeout_seconds,
        ),
        request_check(
            "readiness",
            "GET",
            f"{args.base_url}/readiness",
            headers=headers,
            timeout_seconds=args.timeout_seconds,
        ),
        request_check(
            "query",
            "POST",
            f"{args.base_url}/query",
            headers=query_headers,
            payload=query_payload,
            timeout_seconds=args.timeout_seconds,
            validate=validate_query_response,
        ),
        request_check(
            "metrics",
            "GET",
            f"{args.base_url}/metrics",
            headers=headers,
            timeout_seconds=args.timeout_seconds,
            validate=validate_metrics_response,
        ),
    ]


def request_check(
    name: str,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    payload: dict[str, object] | None = None,
    validate: Any | None = None,
) -> SmokeCheck:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            latency_ms = _elapsed_ms(started_at)
            raw = response.read().decode("utf-8", errors="replace")
            if not 200 <= response.status < 300:
                return SmokeCheck(name, "FAIL", latency_ms, f"HTTP {response.status}")
            if validate is not None:
                validation_error = validate(raw)
                if validation_error:
                    return SmokeCheck(name, "FAIL", latency_ms, validation_error)
            return SmokeCheck(name, "PASS", latency_ms, f"HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        latency_ms = _elapsed_ms(started_at)
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return SmokeCheck(name, "FAIL", latency_ms, f"HTTP {exc.code}: {detail}")
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        return SmokeCheck(name, "FAIL", _elapsed_ms(started_at), repr(exc))


def validate_query_response(raw: str) -> str:
    data = json.loads(raw)
    if not data.get("answer"):
        return "Query response did not include an answer."
    if not isinstance(data.get("citations", []), list):
        return "Query response citations were not a list."
    if "cost" not in data:
        return "Query response did not include cost."
    return ""


def validate_metrics_response(raw: str) -> str:
    required_metrics = [
        "enterprise_rag_query_requests_total",
        "enterprise_rag_query_latency_ms_count",
        "enterprise_rag_query_estimated_cost_usd_sum",
    ]
    missing = [metric for metric in required_metrics if metric not in raw]
    if missing:
        return f"Metrics response missing: {', '.join(missing)}"
    return ""


def auth_headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    if args.tenant:
        headers["X-Tenant-ID"] = args.tenant
    return headers


def build_report(
    *,
    status: str,
    env_status: dict[str, object],
    checks: list[SmokeCheck],
    skipped_reasons: list[str],
) -> dict[str, object]:
    return {
        "status": status,
        "env": env_status,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "latency_ms": round(check.latency_ms, 4),
                "detail": check.detail,
            }
            for check in checks
        ],
        "skipped_reasons": skipped_reasons,
    }


def format_report(report: dict[str, object]) -> str:
    lines = ["Smoke Test Report", f"status: {report['status']}"]
    skipped_reasons = report.get("skipped_reasons", [])
    if skipped_reasons:
        lines.append("skipped_reasons:")
        lines.extend(f"  - {reason}" for reason in skipped_reasons)
    checks = report.get("checks", [])
    if checks:
        lines.append("checks:")
        for check in checks:
            lines.append(f"  - {check['name']}: {check['status']} ({check['latency_ms']:.4f} ms) {check['detail']}")
    return "\n".join(lines)


def _emit_report(report: dict[str, object], output_json: Path | None) -> None:
    print(format_report(report))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a production smoke test against the Enterprise RAG API.")
    parser.add_argument("--base-url", default=os.getenv("ENTERPRISE_RAG_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("ENTERPRISE_RAG_API_KEY", ""))
    parser.add_argument("--tenant", default=os.getenv("ENTERPRISE_RAG_TENANT", "acme"))
    parser.add_argument("--query", default="What does AUTH-429 affect?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path, help="Write smoke test report JSON.")
    parser.add_argument("--require-openai", action="store_true", help="Skip unless OPENAI_API_KEY is configured.")
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Allow smoke tests against local unsecured API config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
