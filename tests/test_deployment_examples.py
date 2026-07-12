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
