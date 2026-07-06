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
class AppConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


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
    )


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section `{key}` must be a JSON object.")
    return value
