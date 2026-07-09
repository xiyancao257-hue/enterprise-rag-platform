from __future__ import annotations

import json
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
class ApiSecurityConfig:
    require_api_key: bool = False
    api_key_env_var: str = "ENTERPRISE_RAG_API_KEYS"
    api_key_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorIndexConfig:
    provider: str = "memory"
    collection_name: str = "enterprise_rag_chunks"
    url: str = "http://localhost:6333"


@dataclass(frozen=True)
class AppConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api_security: ApiSecurityConfig = field(default_factory=ApiSecurityConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    return parse_config(data)


def parse_config(data: dict[str, Any]) -> AppConfig:
    retrieval_data = _section(data, "retrieval")
    security_data = _section(data, "security")
    api_security_data = _section(data, "api_security")
    vector_index_data = _section(data, "vector_index")

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
        ),
        vector_index=VectorIndexConfig(
            provider=str(vector_index_data.get("provider", VectorIndexConfig.provider)),
            collection_name=str(vector_index_data.get("collection_name", VectorIndexConfig.collection_name)),
            url=str(vector_index_data.get("url", VectorIndexConfig.url)),
        ),
    )


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{key}` must be a JSON object.")
    return value
