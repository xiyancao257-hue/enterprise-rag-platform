import json
from pathlib import Path

from enterprise_rag.config import load_config


def test_production_example_config_is_parseable() -> None:
    config = load_config(Path("config/production.example.json"))

    assert config.api_security.require_api_key is True
    assert config.vector_index.provider == "qdrant"
    assert config.cache.provider == "redis"
    assert config.leases.provider == "redis"
    assert config.llm.max_retries == 2
    assert config.embedding.circuit_breaker_failure_threshold == 5


def test_env_example_uses_placeholders_not_real_secrets() -> None:
    content = Path(".env.example").read_text(encoding="utf-8")

    assert "ENTERPRISE_RAG_API_KEY=replace-with-real-api-key" in content
    assert "OPENAI_API_KEY=" in content
    assert "sk-" not in content


def test_prometheus_config_scrapes_api_metrics_with_env_auth() -> None:
    content = Path("monitoring/prometheus.yml").read_text(encoding="utf-8")

    assert "job_name: enterprise-rag-api" in content
    assert "metrics_path: /metrics" in content
    assert "api:8000" in content
    assert "credentials: ${ENTERPRISE_RAG_API_KEY}" in content
    assert "/etc/prometheus/alerts.yml" in content


def test_prod_compose_wires_prometheus_monitoring() -> None:
    content = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "prometheus:" in content
    assert "prom/prometheus" in content
    assert "--config.expand-env" in content
    assert "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in content
    assert "./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro" in content
    assert "9090:9090" in content


def test_prod_compose_wires_grafana_dashboard() -> None:
    content = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "grafana:" in content
    assert "grafana/grafana" in content
    assert "3000:3000" in content
    assert "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" in content
    assert "./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro" in content
    assert "grafana_data:" in content


def test_prometheus_alert_rules_cover_core_rag_failures() -> None:
    content = Path("monitoring/alerts.yml").read_text(encoding="utf-8")

    assert "EnterpriseRagQueryFailures" in content
    assert "EnterpriseRagHighAverageQueryLatency" in content
    assert "EnterpriseRagIngestJobFailures" in content


def test_grafana_dashboard_is_valid_and_covers_rag_operations() -> None:
    dashboard = json.loads(Path("monitoring/grafana/dashboards/enterprise-rag.json").read_text(encoding="utf-8"))
    expressions = json.dumps(dashboard)

    assert dashboard["title"] == "Enterprise RAG Operations"
    assert dashboard["uid"] == "enterprise-rag-ops"
    assert len(dashboard["panels"]) >= 8
    assert "enterprise_rag_query_latency_ms_sum" in expressions
    assert "enterprise_rag_query_failures_total" in expressions
    assert "enterprise_rag_query_cache_hits_total" in expressions
    assert "enterprise_rag_provider_latency_ms_sum" in expressions
    assert "enterprise_rag_query_estimated_cost_usd_sum" in expressions
    assert "enterprise_rag_ingest_job_failures_total" in expressions
    assert "enterprise_rag_feedback_total" in expressions


def test_grafana_provisioning_points_to_prometheus_and_dashboard() -> None:
    datasource = Path("monitoring/grafana/provisioning/datasources/prometheus.yml").read_text(encoding="utf-8")
    provider = Path("monitoring/grafana/provisioning/dashboards/enterprise-rag.yml").read_text(encoding="utf-8")

    assert "url: http://prometheus:9090" in datasource
    assert "isDefault: true" in datasource
    assert "path: /var/lib/grafana/dashboards" in provider
