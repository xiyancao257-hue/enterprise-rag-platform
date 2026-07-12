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


def test_prometheus_alert_rules_cover_core_rag_failures() -> None:
    content = Path("monitoring/alerts.yml").read_text(encoding="utf-8")

    assert "EnterpriseRagQueryFailures" in content
    assert "EnterpriseRagHighAverageQueryLatency" in content
    assert "EnterpriseRagIngestJobFailures" in content
