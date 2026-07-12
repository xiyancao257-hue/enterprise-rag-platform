from pathlib import Path

from enterprise_rag.config import AppConfig, load_config, load_config_from_env, parse_config


def test_load_config_without_path_uses_defaults() -> None:
    config = load_config()

    assert config == AppConfig()
    assert config.retrieval.top_k == 5
    assert config.retrieval.enable_graph is False
    assert config.retrieval.graph_max_hops == 2
    assert config.security.default_user_groups == ()
    assert config.vector_index.provider == "memory"


def test_parse_config_loads_retrieval_and_security_settings() -> None:
    config = parse_config(
        {
            "retrieval": {
                "top_k": 8,
                "enable_graph": True,
                "graph_max_hops": 3,
                "experiment_k_values": [2, 4, 6],
            },
            "security": {
                "default_user_groups": ["engineering", "support"],
            },
            "api_security": {
                "require_api_key": True,
                "api_key_env_var": "ENTERPRISE_RAG_TEST_KEYS",
                "api_key_hashes": ["abc123"],
                "rate_limit_requests": 10,
                "rate_limit_window_seconds": 30,
                "api_keys": [
                    {
                        "key_hash": "def456",
                        "allowed_tenants": ["acme", "globex"],
                    }
                ],
            },
            "vector_index": {
                "provider": "qdrant",
                "collection_name": "chunks",
                "url": "http://qdrant:6333",
            },
            "jobs": {
                "running_timeout_seconds": 900,
                "worker_poll_seconds": 2.5,
            },
            "ingestion": {
                "allowed_source_roots": ["data/raw", "/mnt/shared/rag"],
                "allowed_extensions": [".md", ".txt", ".html"],
                "max_file_bytes": 2048,
            },
            "ocr": {
                "provider": "tesseract",
                "tesseract_cmd": "/opt/bin/tesseract",
                "tesseract_timeout_seconds": 12.5,
                "pdf_renderer_cmd": "/opt/bin/pdftoppm",
                "pdf_render_dpi": 300,
                "pdf_render_timeout_seconds": 45.0,
                "aws_region": "us-west-2",
                "azure_endpoint_env_var": "AZURE_DI_ENDPOINT",
                "azure_key_env_var": "AZURE_DI_KEY",
            },
            "chunking": {
                "default": {
                    "target_tokens": 200,
                    "max_tokens": 320,
                },
                "by_extension": {
                    ".md": {
                        "target_tokens": 260,
                        "max_tokens": 420,
                    }
                },
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-test",
                "input_cost_per_1k_tokens": 0.1,
                "output_cost_per_1k_tokens": 0.2,
                "timeout_seconds": 12.0,
                "max_retries": 2,
                "retry_backoff_seconds": 0.5,
                "circuit_breaker_failure_threshold": 3,
                "circuit_breaker_reset_seconds": 45.0,
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-test",
                "dimensions": 1024,
                "timeout_seconds": 8.0,
                "max_retries": 1,
                "retry_backoff_seconds": 0.25,
                "circuit_breaker_failure_threshold": 2,
                "circuit_breaker_reset_seconds": 20.0,
            },
            "guardrails": {
                "min_citations": 2,
                "min_top_score": 0.25,
                "min_evidence_tokens": 12,
                "max_estimated_cost_usd": 0.05,
                "max_latency_ms": 1500,
                "sensitive_terms": ["legal", "finance"],
            },
            "audit": {
                "enabled": True,
                "path": "data/audit/test.jsonl",
            },
            "leases": {
                "provider": "redis",
                "url": "redis://redis:6379/2",
                "prefix": "test-lease",
            },
            "cache": {
                "provider": "redis",
                "url": "redis://redis:6379/1",
                "prefix": "test-rag",
                "query_ttl_seconds": 120,
                "embedding_ttl_seconds": 3600,
            },
            "experiments": {
                "enabled": True,
                "name": "retrieval_profile",
                "variants": [
                    {
                        "name": "baseline",
                        "traffic_weight": 50,
                        "retrieval_profile": {"top_k": 5, "enable_graph": False},
                    },
                    {
                        "name": "graph_candidate",
                        "traffic_weight": 50,
                        "retrieval_profile": {"top_k": 5, "enable_graph": True},
                    },
                ],
            },
        }
    )

    assert config.retrieval.top_k == 8
    assert config.retrieval.enable_graph is True
    assert config.retrieval.graph_max_hops == 3
    assert config.retrieval.experiment_k_values == (2, 4, 6)
    assert config.security.default_user_groups == ("engineering", "support")
    assert config.api_security.require_api_key is True
    assert config.api_security.api_key_env_var == "ENTERPRISE_RAG_TEST_KEYS"
    assert config.api_security.api_key_hashes == ("abc123",)
    assert config.api_security.rate_limit_requests == 10
    assert config.api_security.rate_limit_window_seconds == 30
    assert config.api_security.api_keys[0].key_hash == "def456"
    assert config.api_security.api_keys[0].allowed_tenants == ("acme", "globex")
    assert config.vector_index.provider == "qdrant"
    assert config.vector_index.collection_name == "chunks"
    assert config.vector_index.url == "http://qdrant:6333"
    assert config.jobs.running_timeout_seconds == 900
    assert config.jobs.worker_poll_seconds == 2.5
    assert config.ingestion.allowed_source_roots == ("data/raw", "/mnt/shared/rag")
    assert config.ingestion.allowed_extensions == (".md", ".txt", ".html")
    assert config.ingestion.max_file_bytes == 2048
    assert config.ocr.provider == "tesseract"
    assert config.ocr.tesseract_cmd == "/opt/bin/tesseract"
    assert config.ocr.tesseract_timeout_seconds == 12.5
    assert config.ocr.pdf_renderer_cmd == "/opt/bin/pdftoppm"
    assert config.ocr.pdf_render_dpi == 300
    assert config.ocr.pdf_render_timeout_seconds == 45.0
    assert config.ocr.aws_region == "us-west-2"
    assert config.ocr.azure_endpoint_env_var == "AZURE_DI_ENDPOINT"
    assert config.ocr.azure_key_env_var == "AZURE_DI_KEY"
    assert config.chunking.default.target_tokens == 200
    assert config.chunking.default.max_tokens == 320
    assert config.chunking.by_extension[".md"].target_tokens == 260
    assert config.chunking.by_extension[".md"].max_tokens == 420
    assert config.chunking.profile_for_extension(".txt").target_tokens == 200
    assert config.chunking.profile_for_extension(".MD").target_tokens == 260
    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-test"
    assert config.llm.input_cost_per_1k_tokens == 0.1
    assert config.llm.output_cost_per_1k_tokens == 0.2
    assert config.llm.timeout_seconds == 12.0
    assert config.llm.max_retries == 2
    assert config.llm.retry_backoff_seconds == 0.5
    assert config.llm.circuit_breaker_failure_threshold == 3
    assert config.llm.circuit_breaker_reset_seconds == 45.0
    assert config.embedding.provider == "openai"
    assert config.embedding.model == "text-embedding-test"
    assert config.embedding.dimensions == 1024
    assert config.embedding.timeout_seconds == 8.0
    assert config.embedding.max_retries == 1
    assert config.embedding.retry_backoff_seconds == 0.25
    assert config.embedding.circuit_breaker_failure_threshold == 2
    assert config.embedding.circuit_breaker_reset_seconds == 20.0
    assert config.guardrails.min_citations == 2
    assert config.guardrails.min_top_score == 0.25
    assert config.guardrails.min_evidence_tokens == 12
    assert config.guardrails.max_estimated_cost_usd == 0.05
    assert config.guardrails.max_latency_ms == 1500
    assert config.guardrails.sensitive_terms == ("legal", "finance")
    assert config.audit.enabled is True
    assert config.audit.path == "data/audit/test.jsonl"
    assert config.leases.provider == "redis"
    assert config.leases.url == "redis://redis:6379/2"
    assert config.leases.prefix == "test-lease"
    assert config.cache.provider == "redis"
    assert config.cache.url == "redis://redis:6379/1"
    assert config.cache.prefix == "test-rag"
    assert config.cache.query_ttl_seconds == 120
    assert config.cache.embedding_ttl_seconds == 3600
    assert config.experiments.enabled is True
    assert config.experiments.name == "retrieval_profile"
    assert config.experiments.variants[0].name == "baseline"
    assert config.experiments.variants[0].retrieval_profile == {"top_k": 5, "enable_graph": False}


def test_load_config_from_json_file(tmp_path) -> None:
    config_path = tmp_path / "enterprise-rag.json"
    config_path.write_text(
        """
        {
          "retrieval": {
            "top_k": 3,
            "enable_graph": true
          },
          "security": {
            "default_user_groups": ["admin"]
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retrieval.top_k == 3
    assert config.retrieval.enable_graph is True
    assert config.retrieval.graph_max_hops == 2
    assert config.security.default_user_groups == ("admin",)


def test_load_config_from_env_uses_config_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "enterprise-rag.json"
    config_path.write_text('{"retrieval": {"top_k": 9}}', encoding="utf-8")
    monkeypatch.setenv("ENTERPRISE_RAG_CONFIG", str(config_path))

    config = load_config_from_env()

    assert config.retrieval.top_k == 9


def test_production_example_config_loads() -> None:
    config = load_config(Path("config/production.example.json"))

    assert config.api_security.require_api_key is True
    assert config.vector_index.provider == "qdrant"
    assert config.cache.provider == "redis"
    assert config.leases.provider == "redis"
    assert config.audit.enabled is True
    assert config.guardrails.max_estimated_cost_usd == 0.02
    assert config.ingestion.allowed_source_roots
    assert config.ocr.provider == "disabled"
    assert config.embedding.provider == "hashing"
