from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    enable_graph: bool = False
    graph_max_hops: int = 2
    experiment_k_values: tuple[int, ...] = (1, 3, 5, 8)


@dataclass(frozen=True)
class SecurityConfig:
    default_user_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApiKeyCredential:
    key_hash: str
    allowed_tenants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApiSecurityConfig:
    require_api_key: bool = False
    api_key_env_var: str = "ENTERPRISE_RAG_API_KEYS"
    api_key_hashes: tuple[str, ...] = ()
    api_keys: tuple[ApiKeyCredential, ...] = ()
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60


@dataclass(frozen=True)
class VectorIndexConfig:
    provider: str = "memory"
    collection_name: str = "enterprise_rag_chunks"
    url: str = "http://localhost:6333"


@dataclass(frozen=True)
class JobsConfig:
    running_timeout_seconds: int = 1800
    worker_poll_seconds: float = 5.0


@dataclass(frozen=True)
class IngestionConfig:
    allowed_source_roots: tuple[str, ...] = ()
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".csv", ".pdf")
    max_file_bytes: int = 10_000_000


@dataclass(frozen=True)
class OcrConfig:
    provider: str = "disabled"
    tesseract_cmd: str = "tesseract"
    tesseract_timeout_seconds: float = 30.0
    pdf_renderer_cmd: str = "pdftoppm"
    pdf_render_dpi: int = 200
    pdf_render_timeout_seconds: float = 60.0
    aws_region: str = ""
    azure_endpoint_env_var: str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
    azure_key_env_var: str = "AZURE_DOCUMENT_INTELLIGENCE_KEY"


@dataclass(frozen=True)
class ChunkingProfileConfig:
    target_tokens: int = 220
    max_tokens: int = 360


@dataclass(frozen=True)
class ChunkingConfig:
    default: ChunkingProfileConfig = field(default_factory=ChunkingProfileConfig)
    by_extension: dict[str, ChunkingProfileConfig] = field(default_factory=dict)

    def profile_for_extension(self, extension: str | None) -> ChunkingProfileConfig:
        if extension is None:
            return self.default
        return self.by_extension.get(extension.lower(), self.default)


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "stub"
    model: str = "gpt-4.1-mini"
    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hashing"
    model: str = "text-embedding-3-small"
    dimensions: int = 384


@dataclass(frozen=True)
class GuardrailsConfig:
    min_citations: int = 1
    min_top_score: float = 0.01
    min_evidence_tokens: int = 5
    max_estimated_cost_usd: float = 0.0
    max_latency_ms: float = 0.0
    sensitive_terms: tuple[str, ...] = ("legal", "medical", "finance", "security", "compliance")


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = False
    path: str = "data/audit/audit.jsonl"


@dataclass(frozen=True)
class LeaseConfig:
    provider: str = "memory"
    url: str = "redis://localhost:6379/0"
    prefix: str = "enterprise-rag"


@dataclass(frozen=True)
class CacheConfig:
    provider: str = "memory"
    url: str = "redis://localhost:6379/0"
    prefix: str = "enterprise-rag"
    query_ttl_seconds: int = 300
    embedding_ttl_seconds: int = 86_400


@dataclass(frozen=True)
class AppConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api_security: ApiSecurityConfig = field(default_factory=ApiSecurityConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    leases: LeaseConfig = field(default_factory=LeaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    return parse_config(data)


def load_config_from_env(env_var: str = "ENTERPRISE_RAG_CONFIG") -> AppConfig:
    config_path = os.environ.get(env_var)
    if not config_path:
        return load_config()
    return load_config(Path(config_path))


def parse_config(data: dict[str, Any]) -> AppConfig:
    retrieval_data = _section(data, "retrieval")
    security_data = _section(data, "security")
    api_security_data = _section(data, "api_security")
    vector_index_data = _section(data, "vector_index")
    jobs_data = _section(data, "jobs")
    ingestion_data = _section(data, "ingestion")
    ocr_data = _section(data, "ocr")
    chunking_data = _section(data, "chunking")
    llm_data = _section(data, "llm")
    embedding_data = _section(data, "embedding")
    guardrails_data = _section(data, "guardrails")
    audit_data = _section(data, "audit")
    leases_data = _section(data, "leases")
    cache_data = _section(data, "cache")

    return AppConfig(
        retrieval=RetrievalConfig(
            top_k=int(retrieval_data.get("top_k", RetrievalConfig.top_k)),
            enable_graph=bool(retrieval_data.get("enable_graph", RetrievalConfig.enable_graph)),
            graph_max_hops=int(retrieval_data.get("graph_max_hops", RetrievalConfig.graph_max_hops)),
            experiment_k_values=tuple(
                int(value)
                for value in retrieval_data.get(
                    "experiment_k_values",
                    RetrievalConfig.experiment_k_values,
                )
            ),
        ),
        security=SecurityConfig(
            default_user_groups=tuple(str(group) for group in security_data.get("default_user_groups", ())),
        ),
        api_security=ApiSecurityConfig(
            require_api_key=bool(api_security_data.get("require_api_key", ApiSecurityConfig.require_api_key)),
            api_key_env_var=str(api_security_data.get("api_key_env_var", ApiSecurityConfig.api_key_env_var)),
            api_key_hashes=tuple(str(value) for value in api_security_data.get("api_key_hashes", ())),
            api_keys=_parse_api_key_credentials(api_security_data.get("api_keys", [])),
            rate_limit_requests=int(
                api_security_data.get("rate_limit_requests", ApiSecurityConfig.rate_limit_requests)
            ),
            rate_limit_window_seconds=int(
                api_security_data.get(
                    "rate_limit_window_seconds",
                    ApiSecurityConfig.rate_limit_window_seconds,
                )
            ),
        ),
        vector_index=VectorIndexConfig(
            provider=str(vector_index_data.get("provider", VectorIndexConfig.provider)),
            collection_name=str(vector_index_data.get("collection_name", VectorIndexConfig.collection_name)),
            url=str(vector_index_data.get("url", VectorIndexConfig.url)),
        ),
        jobs=JobsConfig(
            running_timeout_seconds=int(jobs_data.get("running_timeout_seconds", JobsConfig.running_timeout_seconds)),
            worker_poll_seconds=float(jobs_data.get("worker_poll_seconds", JobsConfig.worker_poll_seconds)),
        ),
        ingestion=IngestionConfig(
            allowed_source_roots=tuple(
                str(root) for root in ingestion_data.get("allowed_source_roots", IngestionConfig.allowed_source_roots)
            ),
            allowed_extensions=tuple(
                str(extension).lower()
                for extension in ingestion_data.get("allowed_extensions", IngestionConfig.allowed_extensions)
            ),
            max_file_bytes=int(ingestion_data.get("max_file_bytes", IngestionConfig.max_file_bytes)),
        ),
        ocr=OcrConfig(
            provider=str(ocr_data.get("provider", OcrConfig.provider)),
            tesseract_cmd=str(ocr_data.get("tesseract_cmd", OcrConfig.tesseract_cmd)),
            tesseract_timeout_seconds=float(
                ocr_data.get("tesseract_timeout_seconds", OcrConfig.tesseract_timeout_seconds)
            ),
            pdf_renderer_cmd=str(ocr_data.get("pdf_renderer_cmd", OcrConfig.pdf_renderer_cmd)),
            pdf_render_dpi=int(ocr_data.get("pdf_render_dpi", OcrConfig.pdf_render_dpi)),
            pdf_render_timeout_seconds=float(
                ocr_data.get("pdf_render_timeout_seconds", OcrConfig.pdf_render_timeout_seconds)
            ),
            aws_region=str(ocr_data.get("aws_region", OcrConfig.aws_region)),
            azure_endpoint_env_var=str(ocr_data.get("azure_endpoint_env_var", OcrConfig.azure_endpoint_env_var)),
            azure_key_env_var=str(ocr_data.get("azure_key_env_var", OcrConfig.azure_key_env_var)),
        ),
        chunking=_parse_chunking_config(chunking_data),
        llm=LLMConfig(
            provider=str(llm_data.get("provider", LLMConfig.provider)),
            model=str(llm_data.get("model", LLMConfig.model)),
            input_cost_per_1k_tokens=float(
                llm_data.get("input_cost_per_1k_tokens", LLMConfig.input_cost_per_1k_tokens)
            ),
            output_cost_per_1k_tokens=float(
                llm_data.get("output_cost_per_1k_tokens", LLMConfig.output_cost_per_1k_tokens)
            ),
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding_data.get("provider", EmbeddingConfig.provider)),
            model=str(embedding_data.get("model", EmbeddingConfig.model)),
            dimensions=int(embedding_data.get("dimensions", EmbeddingConfig.dimensions)),
        ),
        guardrails=GuardrailsConfig(
            min_citations=int(guardrails_data.get("min_citations", GuardrailsConfig.min_citations)),
            min_top_score=float(guardrails_data.get("min_top_score", GuardrailsConfig.min_top_score)),
            min_evidence_tokens=int(guardrails_data.get("min_evidence_tokens", GuardrailsConfig.min_evidence_tokens)),
            max_estimated_cost_usd=float(
                guardrails_data.get("max_estimated_cost_usd", GuardrailsConfig.max_estimated_cost_usd)
            ),
            max_latency_ms=float(guardrails_data.get("max_latency_ms", GuardrailsConfig.max_latency_ms)),
            sensitive_terms=tuple(
                str(term).lower() for term in guardrails_data.get("sensitive_terms", GuardrailsConfig.sensitive_terms)
            ),
        ),
        audit=AuditConfig(
            enabled=bool(audit_data.get("enabled", AuditConfig.enabled)),
            path=str(audit_data.get("path", AuditConfig.path)),
        ),
        leases=LeaseConfig(
            provider=str(leases_data.get("provider", LeaseConfig.provider)),
            url=str(leases_data.get("url", LeaseConfig.url)),
            prefix=str(leases_data.get("prefix", LeaseConfig.prefix)),
        ),
        cache=CacheConfig(
            provider=str(cache_data.get("provider", CacheConfig.provider)),
            url=str(cache_data.get("url", CacheConfig.url)),
            prefix=str(cache_data.get("prefix", CacheConfig.prefix)),
            query_ttl_seconds=int(cache_data.get("query_ttl_seconds", CacheConfig.query_ttl_seconds)),
            embedding_ttl_seconds=int(cache_data.get("embedding_ttl_seconds", CacheConfig.embedding_ttl_seconds)),
        ),
    )


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{key}` must be a JSON object.")
    return value


def _parse_chunking_config(data: dict[str, Any]) -> ChunkingConfig:
    default_data = data.get("default", {})
    if not isinstance(default_data, dict):
        raise ValueError("Config section `chunking.default` must be a JSON object.")

    by_extension_data = data.get("by_extension", {})
    if not isinstance(by_extension_data, dict):
        raise ValueError("Config section `chunking.by_extension` must be a JSON object.")

    return ChunkingConfig(
        default=_parse_chunking_profile(default_data),
        by_extension={
            str(extension).lower(): _parse_chunking_profile(value) for extension, value in by_extension_data.items()
        },
    )


def _parse_chunking_profile(value: Any) -> ChunkingProfileConfig:
    if not isinstance(value, dict):
        raise ValueError("Chunking profile values must be JSON objects.")
    return ChunkingProfileConfig(
        target_tokens=int(value.get("target_tokens", ChunkingProfileConfig.target_tokens)),
        max_tokens=int(value.get("max_tokens", ChunkingProfileConfig.max_tokens)),
    )


def _parse_api_key_credentials(value: Any) -> tuple[ApiKeyCredential, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Config field `api_security.api_keys` must be a list.")

    credentials = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each `api_security.api_keys` entry must be a JSON object.")
        credentials.append(
            ApiKeyCredential(
                key_hash=str(item.get("key_hash", "")),
                allowed_tenants=tuple(str(tenant) for tenant in item.get("allowed_tenants", ())),
            )
        )
    return tuple(credentials)
